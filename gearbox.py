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

        Use 0.3–0.7 for the helpfulness-alignment sweep.

   

    target_layers: Which transformer layers to apply gearbox to.

        None = all layers. Recommend starting with [8, 12, 16] for Llama-1B.

   

    head_subset: Which attention heads to steer.

        None = all heads. Recommend heads identified by high probe accuracy.

   

    verbose: Print commutativity errors and energy anchor triggers.

    """

    energy_epsilon: float = 0.1

    steering_intensity: float = 0.5

    target_layers: Optional[List[int]] = None

    head_subset: Optional[List[int]] = None

    commutativity_tol: float = 1e-3

    verbose: bool = True

 

 

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

        config: GearboxConfig

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

        # RoPE rotates (dim 0, dim d/2), (dim 1, dim d/2+1), etc.

        # We use the "interleaved" pairing: (2i, 2i+1) for i in 0..d/2-1

        self.R = self._enforce_rope_commutativity(R_raw)

       

        # Verify commutativity

        self._verify_commutativity()

   

    def _enforce_rope_commutativity(self, R: torch.Tensor) -> torch.Tensor:

        """

        Zero out entries that violate RoPE block structure.

       

        RoPE uses 2×2 rotation blocks. Our R must have the same

        block-diagonal structure to commute with it.

       

        The key insight: a matrix commutes with all 2×2 rotations

        iff it is itself block-diagonal with 2×2 blocks, and each

        block is a scalar multiple of I or a rotation matrix.

       

        We approximate this by keeping only within-pair entries.

        """

        d = R.shape[0]

        R_comm = torch.zeros_like(R)

       

        # RoPE pairs: (i, i + d//2) for Llama-style RoPE

        half = d // 2

       

        for i in range(half):

            j = i + half

            # Keep the 2×2 block for this pair

            R_comm[i, i] = R[i, i]

            R_comm[i, j] = R[i, j]

            R_comm[j, i] = R[j, i]

            R_comm[j, j] = R[j, j]

       

        return R_comm

   

    def _verify_commutativity(self):

        """

        Test R W_θ ≈ W_θ R for a sample RoPE rotation.

        Reports the Frobenius norm of the commutator [R, W_θ].

        """

        d = self.head_dim

        theta = 0.5  # arbitrary test angle

       

        # Build a sample RoPE rotation (block structure)

        W = torch.zeros(d, d)

        half = d // 2

        for i in range(half):

            W[i, i] = np.cos(theta * (i + 1))

            W[i, i + half] = -np.sin(theta * (i + 1))

            W[i + half, i] = np.sin(theta * (i + 1))

            W[i + half, i + half] = np.cos(theta * (i + 1))

       

        commutator = self.R @ W - W @ self.R

        error = commutator.norm(p='fro').item()

       

        if self.config.verbose:

            log.info(f"[Commutativity] ‖[R, W_θ]‖_F = {error:.6f} "

                     f"(tol={self.config.commutativity_tol})")

       

        if error > self.config.commutativity_tol:

            log.warning(

                f"[Commutativity] VIOLATION: error {error:.4f} > "

                f"tol {self.config.commutativity_tol}. "

                f"RoPE fidelity not guaranteed."

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

            # output shape: (batch, seq_len, hidden)

            x_orig = output.float()

           

            # Reshape to per-head view for head_dim-aligned projection

            # (batch, seq, hidden) → (batch, seq, n_heads, head_dim)

            b, s, h = x_orig.shape

            head_dim = R.shape[0]

            n_heads = h // head_dim

           

            x_heads = x_orig.view(b, s, n_heads, head_dim)  # (b, s, nh, d)

            R_dev = R.to(x_heads.device)

           

            # Apply R to each head: x_steered = x @ Rᵀ (right-multiply)

            # This is equivalent to rotating each head's representation

            x_steered = x_heads @ R_dev.T  # (b, s, nh, d)

           

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

   

    def attach(self, model, layer_idx: int):

        """

        Attach hooks to q_proj and k_proj of the specified layer.

       

        For Llama architecture:

            model.model.layers[i].self_attn.q_proj

            model.model.layers[i].self_attn.k_proj

        """

        layer = model.model.layers[layer_idx]

        attn = layer.self_attn

       

        h1 = attn.q_proj.register_forward_hook(self._make_hook("q_proj"))

        h2 = attn.k_proj.register_forward_hook(self._make_hook("k_proj"))

       

        self._handles.extend([h1, h2])

        log.info(f"[Gearbox] Attached to layer {layer_idx} q_proj + k_proj")

   

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

       

        # Determine head_dim from model config

        model_cfg = model.config

        self.head_dim = model_cfg.hidden_size // model_cfg.num_attention_heads

       

        log.info(

            f"[Gearbox] Model: {model_cfg._name_or_path}, "

            f"head_dim={self.head_dim}, "

            f"layers={model_cfg.num_hidden_layers}"

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

                config=config

            )

           

            anchor = ResidualEnergyAnchor(epsilon=config.energy_epsilon)

            self.anchors.append(anchor)

           

            hook = GearboxHook(

                projector=projector,

                anchor=anchor,

                layer_idx=layer_idx,

                verbose=config.verbose

            )

            self.hooks.append((layer_idx, hook))

   

    def install(self):

        """Attach all hooks to the model."""

        for layer_idx, hook in self.hooks:

            hook.attach(self.model, layer_idx)

        log.info(f"[Gearbox] Installed on {len(self.hooks)} layers.")

   

    def remove(self):

        """Detach all hooks."""

        for _, hook in self.hooks:

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