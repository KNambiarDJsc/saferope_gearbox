"""

SafeRoPE Gearbox — Core Implementation

Commutative Manifold Injection via RoPE Hijacking

 

Author: Karthik Nambiar

Paper: "Geometric Governance of LLM Latent Space via Commutative Manifold Injection"

"""

 

import torch

import torch.nn as nn

import numpy as np

from typing import Optional, Tuple, Dict, List

from dataclasses import dataclass

import logging

 

logging.basicConfig(level=logging.INFO)

log = logging.getLogger("SafeRoPE")

 

 

# ─────────────────────────────────────────────

# 1. CONFIGURATION

# ─────────────────────────────────────────────

 

@dataclass

class GearboxConfig:

    """

    All hyperparameters for the SafeRoPE Gearbox.

 

    energy_epsilon: Residual Energy Anchor tolerance.

        If steered norm deviates from original by more than this fraction,

        we fall back to identity (R → I). Prevents OOD glitch.

 

    steering_intensity: Scale factor [0, 1] for the intervention.

        0 = no steering (R = I), 1 = full projection.

        Use 0.3-0.7 for the helpfulness-alignment sweep.

 

    target_layers: Which transformer layers to apply gearbox to.

        None = all layers.

 

    head_subset: Which attention heads to steer.

        None = all heads.

 

    rope_pairing: RoPE dimension pairing scheme.

        "split"       -- Llama style: pairs (i, i + d//2).

        "interleaved" -- Gemma-2 style: pairs (2i, 2i+1).

        "auto"        -- detect from model config (default).

 

    model_arch: Architecture hint for hook attachment paths.

        "auto" detects from model config (recommended).

 

    verbose: Print commutativity errors and energy anchor triggers.

    """

    energy_epsilon: float = 0.1

    steering_intensity: float = 0.5

    target_layers: Optional[List[int]] = None

    head_subset: Optional[List[int]] = None

    commutativity_tol: float = 1e-3

    rope_pairing: str = "auto"   # "split" | "interleaved" | "auto"

    model_arch: str = "auto"     # "llama" | "gemma2" | "auto"

    verbose: bool = True

 

 

def detect_model_arch(model: "nn.Module") -> Tuple[str, str]:

    """

    Auto-detect model architecture and RoPE pairing from model config.

 

    Returns:

        (arch, rope_pairing) where:

            arch         = "gemma2" | "llama" | "unknown"

            rope_pairing = "interleaved" | "split"

    """

    model_type = getattr(model.config, "model_type", "").lower()

    arch_name = getattr(model.config, "_name_or_path", "").lower()

 

    if "gemma2" in model_type or "gemma-2" in arch_name or "gemma_2" in arch_name:

        return "gemma2", "interleaved"

    elif "gemma" in model_type or "gemma" in arch_name:

        return "gemma", "split"

    elif "llama" in model_type or "llama" in arch_name:

        return "llama", "split"

    elif "mistral" in model_type or "mistral" in arch_name:

        return "mistral", "split"

    else:

        log.warning(

            f"[Gearbox] Unknown model type '{model_type}'. "

            f"Defaulting to split RoPE. "

            f"If results are poor, try rope_pairing='interleaved'."

        )

        return "unknown", "split"

 

 

# ─────────────────────────────────────────────

# 2. SUBSPACE EXTRACTION (SVD-based)

# ─────────────────────────────────────────────

 

class HarmSubspaceExtractor:

    """

    Extracts the 'harm direction' from a trained linear probe via SVD.

   

    Given probe weight matrix W ∈ ℝ^{1 × d_head}, we compute its SVD

    to get the dominant singular vector — this IS the harm direction in

    the attention head's representation space.

   

    For rank-k extraction (k>1), we return the top-k left singular vectors,

    forming the basis U for the harmful subspace.

    """

   

    def __init__(self, rank: int = 1):

        self.rank = rank

        self.harm_basis: Optional[torch.Tensor] = None  # shape: (d, rank)

   

    def fit(self, probe_weights: torch.Tensor) -> "HarmSubspaceExtractor":

        """

        Args:

            probe_weights: Linear probe weight tensor, shape (1, d) or (d,).

                           Can also be a matrix (k, d) from multi-class probe.

        """

        W = probe_weights.float()

        if W.dim() == 1:

            W = W.unsqueeze(0)  # (1, d)

       

        # SVD: W = U Σ Vᵀ

        # V columns are right singular vectors — directions in input space

        # For a probe W, the top right singular vector = harm direction

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

       

        # Vh is (k, d), rows are right singular vectors

        # Top-k rows of Vh = top-k harm directions in representation space

        harm_directions = Vh[:self.rank].T  # shape: (d, rank)

       

        # Orthonormalize (already orthonormal from SVD, but numerical safety)

        self.harm_basis, _ = torch.linalg.qr(harm_directions)

       

        log.info(f"[SVD] Extracted rank-{self.rank} harm subspace. "

                 f"Top singular value: {S[0].item():.4f}")

        return self

   

    def fit_from_contrast(

        self,

        harmful_activations: torch.Tensor,

        safe_activations: torch.Tensor

    ) -> "HarmSubspaceExtractor":

        """

        Alternative: extract harm direction as mean difference

        between harmful and safe activation clusters (PCA-like).

       

        Args:

            harmful_activations: shape (N_harm, d)

            safe_activations:    shape (N_safe, d)

        """

        harm_mean = harmful_activations.float().mean(0)

        safe_mean = safe_activations.float().mean(0)

       

        diff = (harm_mean - safe_mean).unsqueeze(0)  # (1, d)

        return self.fit(diff)

 

 

# ─────────────────────────────────────────────

# 3. COMMUTATIVE ROTATION MATRIX (The Gearbox)

# ─────────────────────────────────────────────

 

class CommutativeProjector:

    """

    Builds the steering matrix R that:

    (a) Projects out the harm subspace (suppresses harmful directions)

    (b) Commutes with the RoPE rotation operator W_θ

   

    Construction:

        R = I - α * (U Uᵀ)     [Oblique projection]

   

    where:

        U = harm basis from SVD (d, rank)

        α = steering_intensity ∈ [0, 2]

            α=0  → no effect (R=I)

            α=1  → projection (removes component in harm subspace)

            α=2  → reflection (Householder, reverses harm direction)

   

    Commutativity Enforcement:

        RoPE rotates pairs of dimensions. R must be block-diagonal with

        2×2 blocks aligned to RoPE's rotation pairs. We enforce this by

        zeroing out cross-block entries after construction.

    """

   

    def __init__(

        self,

        harm_basis: torch.Tensor,

        head_dim: int,

        config: GearboxConfig,

        rope_pairing: str = "split",

    ):

        self.config = config

        self.head_dim = head_dim

        d = harm_basis.shape[0]

 

        assert d == head_dim, (

            f"Harm basis dim {d} must match head_dim {head_dim}"

        )

 

        U = harm_basis.float()  # (d, rank)

        alpha = config.steering_intensity

 

        # Build projection matrix: P = U Uᵀ (projects onto harm subspace)

        P = U @ U.T  # (d, d)

 

        # Steering matrix: R = I - α * P

        R_raw = torch.eye(d) - alpha * P

 

        # Enforce block-diagonal structure aligned to RoPE pairs

        self.R = self._enforce_rope_commutativity(R_raw, rope_pairing)

 

        # Verify commutativity using the same pairing

        self._verify_commutativity(rope_pairing)

   

    def _enforce_rope_commutativity(

        self, R: torch.Tensor, rope_pairing: str = "split"

    ) -> torch.Tensor:

        """

        Zero out entries that violate RoPE block structure.

 

        Two pairing schemes exist across model families:

 

        SPLIT (Llama, Mistral, Gemma-1):

            Pairs dimension i with i + d//2.

            cos/sin applied as: [x[:d//2]*cos - x[d//2:]*sin,

                                  x[:d//2]*sin + x[d//2:]*cos]

 

        INTERLEAVED (Gemma-2, some Phi variants):

            Pairs dimension 2i with 2i+1.

            cos/sin applied as: [x[0]*cos - x[1]*sin,

                                  x[0]*sin + x[1]*cos, ...]

 

        Using the wrong pairing → large commutator error →

        positional context corrupted → grammar degrades.

        The commutativity test in _verify_commutativity catches this.

        """

        d = R.shape[0]

        R_comm = torch.zeros_like(R)

 

        if rope_pairing == "interleaved":

            # Gemma-2 style: pair (2i, 2i+1)

            for i in range(0, d, 2):

                j = i + 1

                if j >= d:

                    break

                R_comm[i, i] = R[i, i]

                R_comm[i, j] = R[i, j]

                R_comm[j, i] = R[j, i]

                R_comm[j, j] = R[j, j]

        else:

            # Llama / split style: pair (i, i + d//2)

            half = d // 2

            for i in range(half):

                j = i + half

                R_comm[i, i] = R[i, i]

                R_comm[i, j] = R[i, j]

                R_comm[j, i] = R[j, i]

                R_comm[j, j] = R[j, j]

 

        return R_comm

   

    def _verify_commutativity(self, rope_pairing: str = "split"):

        """

        Test R W_theta ~ W_theta R for a sample RoPE rotation.

        Builds W using the same pairing scheme as _enforce_rope_commutativity

        so the test is consistent with the construction.

        """

        d = self.head_dim

        theta = 0.5

 

        W = torch.zeros(d, d)

        if rope_pairing == "interleaved":

            for i in range(0, d, 2):

                j = i + 1

                if j >= d:

                    break

                freq = theta * (i // 2 + 1)

                W[i, i]   = np.cos(freq)

                W[i, j]   = -np.sin(freq)

                W[j, i]   = np.sin(freq)

                W[j, j]   = np.cos(freq)

        else:

            half = d // 2

            for i in range(half):

                freq = theta * (i + 1)

                W[i, i]        = np.cos(freq)

                W[i, i + half] = -np.sin(freq)

                W[i + half, i] = np.sin(freq)

                W[i + half, i + half] = np.cos(freq)

 

        commutator = self.R @ W - W @ self.R

        error = commutator.norm(p='fro').item()

 

        if self.config.verbose:

            log.info(

                f"[Commutativity] pairing={rope_pairing} "

                f"||[R, W_theta]||_F = {error:.6f} "

                f"(tol={self.config.commutativity_tol})"

            )

 

        if error > self.config.commutativity_tol:

            log.warning(

                f"[Commutativity] VIOLATION: error {error:.4f} > "

                f"tol {self.config.commutativity_tol}. "

                f"RoPE fidelity not guaranteed. "

                f"Try switching rope_pairing to the other scheme."

            )

   

    def get_matrix(self) -> torch.Tensor:

        return self.R

 

 

# ─────────────────────────────────────────────

# 4. RESIDUAL ENERGY ANCHOR

# ─────────────────────────────────────────────

 

class ResidualEnergyAnchor:

    """

    Gate the steering matrix: if applying R changes the

    L2 norm of the activation beyond epsilon, fall back to I.

   

    This prevents "OOD Glitch" — activations drifting outside

    the model's training distribution due to over-steering.

   

    Per the manuscript:

        If ‖x_steered‖₂ / ‖x_orig‖₂ ∉ [1-ε, 1+ε]:

            R → I  (identity, no intervention)

    """

   

    def __init__(self, epsilon: float = 0.1):

        self.epsilon = epsilon

        self.trigger_count = 0

        self.total_count = 0

   

    def apply(

        self,

        x_orig: torch.Tensor,

        x_steered: torch.Tensor,

        R: torch.Tensor

    ) -> torch.Tensor:

        """

        Args:

            x_orig:    Original activation, shape (..., d)

            x_steered: Steered activation R @ x_orig, shape (..., d)

            R:         The steering matrix (for fallback logic)

       

        Returns:

            x_steered if energy ratio is in bounds, else x_orig

        """

        self.total_count += 1

       

        norm_orig = x_orig.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        norm_steered = x_steered.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        ratio = norm_steered / norm_orig

       

        # Check if any token in the batch is out of bounds

        out_of_bounds = (ratio < (1 - self.epsilon)) | (ratio > (1 + self.epsilon))

       

        if out_of_bounds.any():

            self.trigger_count += 1

            # Per-token fallback: use x_orig where OOD, x_steered where safe

            return torch.where(out_of_bounds.expand_as(x_orig), x_orig, x_steered)

       

        return x_steered

   

    def report(self) -> Dict:

        rate = self.trigger_count / max(self.total_count, 1)

        return {

            "anchor_triggers": self.trigger_count,

            "total_calls": self.total_count,

            "trigger_rate": rate

        }

 

 

# ─────────────────────────────────────────────

# 5. GEARBOX HOOK — attaches to the model

# ─────────────────────────────────────────────

 

class GearboxHook:

    """

    PyTorch forward hook that injects the steering matrix

    into a specific attention layer's query/key projections.

   

    Strategy: Hook into q_proj and k_proj output.

    After linear projection but before RoPE is applied,

    we apply R to reshape the representation space.

   

    For Llama-3.2-1B:

        - n_heads = 32, head_dim = 64

        - Projections: q_proj (2048→2048), k_proj (2048→512 for GQA)

    """

   

    def __init__(

        self,

        projector: CommutativeProjector,

        anchor: ResidualEnergyAnchor,

        layer_idx: int,

        verbose: bool = False

    ):

        self.R = projector.get_matrix()

        self.anchor = anchor

        self.layer_idx = layer_idx

        self.verbose = verbose

        self._handles = []

   

    def _make_hook(self, proj_name: str):

        R = self.R

        anchor = self.anchor

        layer_idx = self.layer_idx

        verbose = self.verbose

 

        def hook(module, input, output):

            # output shape: (batch, seq_len, proj_out_dim)

            x_orig = output.float()

            b, s, h = x_orig.shape

 

            head_dim = R.shape[0]

 

            # GQA guard: if h is not evenly divisible by head_dim,

            # the projection (k_proj/v_proj in GQA) uses fewer heads.

            # head_dim is fixed per head; n_heads varies between q and kv.

            if h % head_dim != 0:

                # Dimension mismatch — skip this projection silently.

                # This can happen with certain GQA configs where the

                # kv head_dim differs. Log once then pass through.

                log.debug(

                    f"[Hook L{layer_idx} {proj_name}] "

                    f"dim {h} not divisible by head_dim {head_dim}, "

                    f"skipping intervention."

                )

                return output

 

            n_heads = h // head_dim

            x_heads = x_orig.view(b, s, n_heads, head_dim)

            R_dev = R.to(x_heads.device)

 

            # Apply R to each head: x_steered = x @ R^T (right-multiply)

            x_steered = x_heads @ R_dev.T  # (b, s, n_heads, head_dim)

            x_steered_flat = x_steered.view(b, s, h)

 

            # Apply energy anchor

            result = anchor.apply(x_orig, x_steered_flat, R_dev)

 

            if verbose and layer_idx <= 2:

                log.debug(

                    f"[Hook L{layer_idx} {proj_name}] "

                    f"norm ratio: {(result.norm() / x_orig.norm()).item():.4f}"

                )

 

            return result.to(output.dtype)

 

        return hook

   

    def attach(self, model, layer_idx: int, arch: str = "llama"):

        """

        Attach hooks to q_proj and k_proj of the specified layer.

 

        Gemma-2 specifics vs Llama:

          - Same module path: model.model.layers[i].self_attn

          - BUT Gemma-2-2B uses GQA: 8 KV heads, 16 query heads

            q_proj output: (batch, seq, num_heads * head_dim)

            k_proj output: (batch, seq, num_kv_heads * head_dim)

            These have DIFFERENT hidden dims so need separate R matrices

            sized to their respective per-head dimensions.

          - We handle this by checking output dim at hook time and

            reshaping accordingly, not assuming num_heads == num_kv_heads.

        """

        # Both Llama and Gemma-2 share the same module path

        layer = model.model.layers[layer_idx]

        attn = layer.self_attn

 

        h1 = attn.q_proj.register_forward_hook(self._make_hook("q_proj"))

        h2 = attn.k_proj.register_forward_hook(self._make_hook("k_proj"))

 

        self._handles.extend([h1, h2])

        log.info(

            f"[Gearbox] Attached to layer {layer_idx} "

            f"q_proj + k_proj (arch={arch})"

        )

   

    def detach(self):

        for h in self._handles:

            h.remove()

        self._handles.clear()

        log.info("[Gearbox] All hooks detached")

 

 

# ─────────────────────────────────────────────

# 6. FULL GEARBOX ORCHESTRATOR

# ─────────────────────────────────────────────

 

class SafeRoPEGearbox:

    """

    Top-level orchestrator. Given a model and probe weights,

    builds all components and manages hook lifecycle.

   

    Usage:

        gearbox = SafeRoPEGearbox(model, probe_weights, config)

        gearbox.install()

        outputs = model.generate(...)  # steered

        gearbox.remove()

        print(gearbox.report())

    """

   

    def __init__(

        self,

        model: nn.Module,

        probe_weights: Dict[int, torch.Tensor],  # layer_idx → weight tensor

        config: GearboxConfig = GearboxConfig(),

    ):

        self.model = model

        self.config = config

        self.hooks: List[GearboxHook] = []

        self.anchors: List[ResidualEnergyAnchor] = []

 

        # Auto-detect architecture and RoPE pairing

        if config.model_arch == "auto" or config.rope_pairing == "auto":

            detected_arch, detected_pairing = detect_model_arch(model)

        arch = detected_arch if config.model_arch == "auto" else config.model_arch

        rope_pairing = detected_pairing if config.rope_pairing == "auto" else config.rope_pairing

 

        # Determine head_dim from model config

        model_cfg = model.config

        self.head_dim = model_cfg.hidden_size // model_cfg.num_attention_heads

 

        log.info(

            f"[Gearbox] arch={arch}, rope_pairing={rope_pairing}, "

            f"head_dim={self.head_dim}, "

            f"n_layers={model_cfg.num_hidden_layers}"

        )

 

        # Gemma-2 GQA info

        num_kv_heads = getattr(model_cfg, "num_key_value_heads", model_cfg.num_attention_heads)

        if num_kv_heads != model_cfg.num_attention_heads:

            log.info(

                f"[Gearbox] GQA detected: {model_cfg.num_attention_heads} Q heads, "

                f"{num_kv_heads} KV heads. k_proj hooks will be GQA-aware."

            )

 

        # Build per-layer components

        target_layers = config.target_layers or list(probe_weights.keys())

 

        for layer_idx in target_layers:

            if layer_idx not in probe_weights:

                log.warning(f"No probe for layer {layer_idx}, skipping.")

                continue

 

            extractor = HarmSubspaceExtractor(rank=1)

            extractor.fit(probe_weights[layer_idx])

 

            projector = CommutativeProjector(

                harm_basis=extractor.harm_basis,

                head_dim=self.head_dim,

                config=config,

                rope_pairing=rope_pairing,

            )

 

            anchor = ResidualEnergyAnchor(epsilon=config.energy_epsilon)

            self.anchors.append(anchor)

 

            hook = GearboxHook(

                projector=projector,

                anchor=anchor,

                layer_idx=layer_idx,

                verbose=config.verbose,

            )

            self.hooks.append((layer_idx, hook, arch))

   

    def install(self):

        """Attach all hooks to the model."""

        for layer_idx, hook, arch in self.hooks:

            hook.attach(self.model, layer_idx, arch)

        log.info(f"[Gearbox] Installed on {len(self.hooks)} layers.")

 

    def remove(self):

        """Detach all hooks."""

        for _, hook, _ in self.hooks:

            hook.detach()

        log.info("[Gearbox] Removed from all layers.")

   

    def report(self) -> Dict:

        """Aggregate stats across all layers."""

        total_triggers = sum(a.trigger_count for a in self.anchors)

        total_calls = sum(a.total_count for a in self.anchors)

        return {

            "layers_instrumented": len(self.hooks),

            "total_anchor_triggers": total_triggers,

            "total_hook_calls": total_calls,

            "overall_trigger_rate": total_triggers / max(total_calls, 1),

            "config": self.config.__dict__

        }

 

    def __enter__(self):

        self.install()

        return self

 

    def __exit__(self, *args):

        self.remove()