"""

baselines.py

────────────

Implements two comparison baselines for the paper's Table 1:

 

1. Abliteration (Arditi et al., 2024 / Young, 2025)

   - Find refusal direction via mean difference

   - Permanently modify weight matrices (W_out -= projection)

   - Irreversible, operates on weights not activations

 

2. CAA — Contrastive Activation Addition (Panickssery et al., 2023)

   - Add/subtract a steering vector at inference time

   - Finds vector as mean(harmful_acts) - mean(safe_acts)

   - No commutativity guarantees, no energy anchoring

 

Both are implemented as drop-in wrappers with the same interface

as SafeRoPEGearbox, so eval_runner.py can benchmark all three.

 

Reference:

  - Arditi et al. (2024) "Refusal in LLMs is mediated by a single direction"

  - Panickssery et al. (2023) "Steering LLMs via Contrastive Activation Addition"

  - Young (2025) "Comparative Analysis of LLM Abliteration Methods" (GSM8K -26.5% warning)

"""

 

import torch

import torch.nn as nn

import logging

from typing import Dict, List, Optional, Tuple

from copy import deepcopy

 

log = logging.getLogger("baselines")

 

 

# ─────────────────────────────────────────────

# SHARED UTILITIES

# ─────────────────────────────────────────────

 

def compute_mean_diff_direction(

    harmful_activations: torch.Tensor,   # (N_harm, d)

    safe_activations: torch.Tensor,      # (N_safe, d)

    normalize: bool = True

) -> torch.Tensor:

    """

    The refusal direction as mean difference.

    This is the core of both abliteration and CAA.

   

    Returns: unit vector of shape (d,)

    """

    direction = (

        harmful_activations.float().mean(0)

        - safe_activations.float().mean(0)

    )

    if normalize:

        direction = direction / direction.norm().clamp(min=1e-8)

    return direction

 

 

# ─────────────────────────────────────────────

# BASELINE 1: ABLITERATION

# ─────────────────────────────────────────────

 

class AbliterationBaseline:

    """

    Removes the refusal direction from the model's weight matrices.

   

    Method:

        For each target layer's MLP output projection and attention output:

            W_out ← W_out - (W_out @ d) ⊗ d^T

       

        This permanently removes the component of each output weight

        that points in the harmful direction d.

   

    IMPORTANT: This MODIFIES the model weights in place.

    Call restore() to revert. Or pass a copy of the model.

   

    Warning per Young (2025): GSM8K accuracy can drop up to 26.5%.

    This is the "lobotomy" effect we aim to avoid with SafeRoPE.

    """

   

    def __init__(

        self,

        model: nn.Module,

        probe_weights: Dict[int, torch.Tensor],   # same format as Gearbox

        steering_intensity: float = 1.0,          # α, 1.0 = full abliteration

        target_layers: Optional[List[int]] = None,

    ):

        self.model = model

        self.steering_intensity = steering_intensity

        self.target_layers = target_layers or list(probe_weights.keys())

       

        # Store original weights for restoration

        self._original_weights: Dict[str, torch.Tensor] = {}

       

        # Compute refusal direction per layer from probe weights

        self.directions: Dict[int, torch.Tensor] = {}

        for layer_idx, W in probe_weights.items():

            # Probe weight is (1, d), normalize to get direction

            d = W.squeeze(0).float()

            self.directions[layer_idx] = d / d.norm().clamp(min=1e-8)

   

    def install(self):

        """Modify weight matrices to abliterate the refusal direction."""

        n_modified = 0

       

        for layer_idx in self.target_layers:

            if layer_idx not in self.directions:

                continue

           

            direction = self.directions[layer_idx]  # (d,)

            layer = self.model.model.layers[layer_idx]

           

            # Abliterate from attention output projection

            # and MLP output projection

            targets = [

                ("attn_out", layer.self_attn.o_proj),

                ("mlp_out", layer.mlp.down_proj),

            ]

           

            for name, proj in targets:

                key = f"layer{layer_idx}.{name}"

                W = proj.weight.data.float()  # (d_out, d_in)

 

                # Save original

                self._original_weights[key] = W.clone()

 

                # Standard abliteration (Arditi et al. 2024):

                # Remove the harmful direction from the OUTPUT space of W.

                # W has shape (d_out, d_in). Direction d lives in d_out space.

                # For each row w_i of W: w_i <- w_i - alpha * (w_i . d_norm) * d_norm

                # In matrix form: W_new = W - alpha * (W @ d_norm).unsqueeze(1) * d_norm

                # BUT this requires d_norm in d_in space (columns).

                #

                # Since our probe direction is in hidden_size space = d_out for o_proj,

                # we abliterate from the INPUT space instead — remove the component

                # of each column of W that points along d_in:

                # W_new = W - alpha * d_in * (d_in^T @ W)  [outer product]

                # where d_in matches d_in = W.shape[1].

 

                d_in = direction[:W.shape[1]].to(W.device)   # truncate to d_in

                d_in = d_in / d_in.norm().clamp(min=1e-8)    # renormalise

 

                # (d_in^T @ W^T) = W @ d_in -> shape (d_out,)

                # outer product: d_in.unsqueeze(1) * coeff.unsqueeze(0) -> (d_in, d_out)

                coeff = W @ d_in          # (d_out,)

                W_new = W - self.steering_intensity * coeff.unsqueeze(1) * d_in.unsqueeze(0)

 

                proj.weight.data = W_new.to(proj.weight.dtype)

                n_modified += 1

       

        log.info(

            f"[Abliteration] Modified {n_modified} weight matrices "

            f"across {len(self.target_layers)} layers."

        )

   

    def remove(self):

        """Restore original weights."""

        for layer_idx in self.target_layers:

            layer = self.model.model.layers[layer_idx]

           

            key_attn = f"layer{layer_idx}.attn_out"

            key_mlp = f"layer{layer_idx}.mlp_out"

           

            if key_attn in self._original_weights:

                layer.self_attn.o_proj.weight.data = (

                    self._original_weights[key_attn]

                    .to(layer.self_attn.o_proj.weight.dtype)

                )

            if key_mlp in self._original_weights:

                layer.mlp.down_proj.weight.data = (

                    self._original_weights[key_mlp]

                    .to(layer.mlp.down_proj.weight.dtype)

                )

       

        log.info("[Abliteration] Weights restored.")

   

    def report(self) -> Dict:

        return {

            "method": "abliteration",

            "layers_modified": len(self.target_layers),

            "steering_intensity": self.steering_intensity,

        }

   

    def __enter__(self):

        self.install()

        return self

   

    def __exit__(self, *args):

        self.remove()

 

 

# ─────────────────────────────────────────────

# BASELINE 2: CAA — Contrastive Activation Addition

# ─────────────────────────────────────────────

 

class CAABaseline:

    """

    Contrastive Activation Addition (Panickssery et al., 2023).

   

    At inference time, adds a steering vector to the residual stream.

   

    Method:

        v = mean(harmful_activations) - mean(safe_activations)  [precomputed]

       

        At each token position in each target layer:

            h_l ← h_l - α * v_l

       

    This directly suppresses the harmful direction in the residual stream.

    No commutativity guarantees. No energy anchoring.

    The quadratic degradation from Wolf et al. is expected to appear

    at high α values.

    """

   

    def __init__(

        self,

        model: nn.Module,

        probe_weights: Dict[int, torch.Tensor],

        steering_intensity: float = 15.0,  # CAA typically uses large values (10-30)

        target_layers: Optional[List[int]] = None,

    ):

        self.model = model

        self.alpha = steering_intensity

        self.target_layers = target_layers or list(probe_weights.keys())

        self._handles = []

       

        # CAA steering vectors from probe weights

        self.steering_vectors: Dict[int, torch.Tensor] = {}

        for layer_idx, W in probe_weights.items():

            v = W.squeeze(0).float()

            self.steering_vectors[layer_idx] = v / v.norm().clamp(min=1e-8)

   

    def _make_hook(self, layer_idx: int):

        v = self.steering_vectors[layer_idx]

        alpha = self.alpha

       

        def hook(module, input, output):

            hidden = output[0] if isinstance(output, tuple) else output

            v_dev = v.to(hidden.device)

           

            # Subtract steering vector scaled by alpha

            # hidden shape: (batch, seq, d)

            # v shape: (d,)

            steered = hidden.float() - alpha * v_dev.unsqueeze(0).unsqueeze(0)

           

            if isinstance(output, tuple):

                return (steered.to(hidden.dtype),) + output[1:]

            return steered.to(hidden.dtype)

       

        return hook

   

    def install(self):

        for layer_idx in self.target_layers:

            if layer_idx not in self.steering_vectors:

                continue

            layer = self.model.model.layers[layer_idx]

            h = layer.register_forward_hook(self._make_hook(layer_idx))

            self._handles.append(h)

       

        log.info(f"[CAA] Attached to {len(self._handles)} layers.")

   

    def remove(self):

        for h in self._handles:

            h.remove()

        self._handles.clear()

        log.info("[CAA] Detached.")

   

    def report(self) -> Dict:

        return {

            "method": "CAA",

            "layers": self.target_layers,

            "steering_intensity": self.alpha,

        }

   

    def __enter__(self):

        self.install()

        return self

   

    def __exit__(self, *args):

        self.remove()

 

 

# ─────────────────────────────────────────────

# BASELINE FACTORY — unified interface

# ─────────────────────────────────────────────

 

def get_baseline(

    name: str,

    model: nn.Module,

    probe_weights: Dict[int, torch.Tensor],

    intensity: float,

    target_layers: Optional[List[int]] = None,

):

    """

    Factory for baselines. Returns object with install()/remove() interface.

   

    Args:

        name: "abliteration" | "caa" | "gearbox"

        intensity: steering strength. Each method has different scale:

            - gearbox: [0, 2], typically 0.3–0.7

            - abliteration: [0, 1], typically 1.0

            - caa: [0, 30], typically 10–20

    """

    if name == "abliteration":

        return AbliterationBaseline(

            model, probe_weights, intensity, target_layers

        )

    elif name == "caa":

        return CAABaseline(

            model, probe_weights, intensity, target_layers

        )

    elif name == "gearbox":

        from gearbox import SafeRoPEGearbox, GearboxConfig

        config = GearboxConfig(

            steering_intensity=intensity,

            target_layers=target_layers,

        )

        return SafeRoPEGearbox(model, probe_weights, config)

    else:

        raise ValueError(f"Unknown baseline: {name}. Choose: abliteration, caa, gearbox")

