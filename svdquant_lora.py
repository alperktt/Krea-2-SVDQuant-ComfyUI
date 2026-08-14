"""LoRA support for Krea2 W4A4 quantized models (from ``quantize_krea2.py``).

A normal `LoraLoaderModelOnly` cannot patch these quantized layers correctly: ComfyUI
applies a LoRA by adding `down @ up` onto a module's `.weight`, but here `.weight` is a
`QuantizedTensor` -- patching it that way means dequantize -> add -> requantize, which
throws the 4-bit format away and re-quantizes the LoRA delta along with it.

Krea2 LoRAs (ai-toolkit) target both kinds of layer:

* `diffusion_model.blocks.N.{attn,mlp}.*`  -> quantized, needs a parallel branch
* `diffusion_model.txtfusion.*`            -> ordinary Linear, ComfyUI patches it fine

So this node splits the LoRA: quantized layers get the LoRA attached as an extra
parallel branch (mathematically identical for a linear layer: `(W + BA)x == Wx + B(Ax)`,
so the quantized weight itself is never touched), and everything else is handed to
ComfyUI's normal patching path.

Parsing is ComfyUI's job, not ours: `comfy.lora.load_lora` returns a weight adapter per
layer and already knows every format and naming convention it supports. Plain up/down LoRAs
are folded into the branch above, one pair of GEMMs for the whole stack. Everything else --
LoKr, LoHa, OFT -- runs through ComfyUI's bypass contract `g(f(x) + h(x))`, which is written
for precisely this case: its docstring says "designed for quantized models where weights may
not be accessible". Either way the 4-bit weight is read, never rewritten.

The branch is installed through ``ModelPatcher.add_object_patch`` rather than by mutating
the module. That is what makes the node behave like every other ComfyUI node: `clone()`
shares the underlying `nn.Module`, so writing the LoRA onto the module directly would
leak it into every other branch of the graph that came off the same loader, and would
survive past the sampling run. Object patches are applied in `patch_model` and reverted
afterwards, per patcher.

Stacking works: chaining multiple of these nodes re-applies the whole LoRA stack from
scratch each time, so strengths can change without leftover state from a previous value.
"""

from __future__ import annotations

import copy
import logging
import os
from collections import OrderedDict

import torch

import comfy.lora
import comfy.model_management
import comfy.utils
import comfy.weight_adapter
import folder_paths

from .svdquant_diag import _CATEGORY, quantized_linears
from .svdquant_w4a4 import add_low_rank, has_branch

_PREFIX = "diffusion_model."
_ALT_PREFIX = "transformer."

# What to do with an adapter that cannot fold directly into the low-rank branch (LoKr, LoHa, OFT).
# `bypass` computes it every forward and never touches the 4-bit weight; `svd delta` decomposes
# the weight delta with torch.svd_lowrank and merges it into the low-rank branch (zero per-step
# overhead); `bake` hands it to ComfyUI, which rewrites the weight once and pays nothing per step.
ADAPTER_BYPASS = "bypass (exact, slower)"
ADAPTER_SVD = "svd delta (fast, low-rank SVD, no per-step overhead)"
ADAPTER_BAKE = "bake into the weight (fast, requantizes the delta)"

# Reloading a 300 MB LoRA off disk on every graph execution is pure latency when the
# usual edit is a strength slider. Keyed on mtime+size so editing the file invalidates.
_LORA_CACHE: OrderedDict[str, tuple[tuple, dict]] = OrderedDict()
_LORA_CACHE_MAX = 4


def _load_lora_cached(path: str) -> dict:
    stat = os.stat(path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    hit = _LORA_CACHE.get(path)
    if hit is not None and hit[0] == stamp:
        # Without this the eviction below is insertion-ordered (FIFO), so cycling five LoRAs
        # can drop the one being used every run.
        _LORA_CACHE.move_to_end(path)
        return hit[1]
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if len(_LORA_CACHE) >= _LORA_CACHE_MAX:
        _LORA_CACHE.pop(next(iter(_LORA_CACHE)))
    _LORA_CACHE[path] = (stamp, sd)
    return sd


def _key_map(model):
    """ComfyUI's LoRA-key to model-key map, plus the one naming it does not cover.

    `model_lora_keys_unet` already handles every format and every naming ComfyUI knows --
    diffusers, lycoris, kohya -- so parsing is its job, not ours. The one gap is a diffusers
    *prefix* in front of *native* module names (`transformer.blocks.N.attn.wq`), which some
    PEFT-based extractions write: ComfyUI's `transformer.` entries are keyed on diffusers
    module names (`transformer.transformer_blocks.N.attn.to_q`), so the hybrid matches
    nothing and the LoRA silently does nothing. Aliasing the map rather than special-casing
    the parser means the fix applies to every format, not just plain LoRA.
    """
    key_map = comfy.lora.model_lora_keys_unet(model, {})
    for lora_key, model_key in list(key_map.items()):
        if lora_key.startswith(_PREFIX):
            key_map.setdefault(_ALT_PREFIX + lora_key[len(_PREFIX):], model_key)
    return key_map


def _layer_of(model_key: str, quant_layers: set[str]) -> str | None:
    """`diffusion_model.blocks.0.attn.wq.weight` -> `blocks.0.attn.wq`, if it is quantized."""
    if not model_key.startswith(_PREFIX) or not model_key.endswith(".weight"):
        return None
    layer = model_key[len(_PREFIX):-len(".weight")]
    return layer if layer in quant_layers else None


def _plain_lora_factors(adapter, strength: float, device, dtype):
    """(l1, l2) for a plain up/down LoRA, or None if this adapter is anything else.

    Worth the special case: folded this way an N-LoRA stack costs one pair of GEMMs per step
    (see `_stack_factors`), where the generic adapter path costs a pair *per adapter*. Every
    ordinary Krea2 LoRA takes this route; the generic path is for the rest.
    """
    if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
        return None
    up, down, alpha, mid, dora_scale = adapter.weights[:5]
    reshape = adapter.weights[5] if len(adapter.weights) > 5 else None
    # mid is a Tucker/conv core, dora_scale rescales the *base* weight's norm, reshape means
    # the LoRA was trained against a differently-shaped weight. None of the three survive
    # being folded into a parallel branch, so they go to the generic path instead.
    if mid is not None or dora_scale is not None or reshape is not None or up.ndim != 2:
        return None
    # `if alpha` would treat a legitimate alpha of 0.0 as "no alpha" and silently run the
    # layer at full strength instead of disabling it.
    rank = int(down.shape[0])
    mult = strength * (float(alpha) / rank if alpha is not None else 1.0)
    l1 = up.to(device=device, dtype=dtype)
    l2 = down.to(device=device, dtype=dtype) * mult
    return l1.contiguous(), l2.contiguous()


def _svd_adapter_factors(adapter, strength: float, device, dtype, max_rank: int = 64):
    """Decompose an arbitrary adapter's weight delta into low-rank factors (l1, l2) via SVD.

    For LoKr, LoHa, OFT, GFS, etc. This extracts the weight delta, computes a
    truncated SVD using torch.svd_lowrank, and produces (l1, l2) where l1 @ l2 ≈ delta * strength.
    Running this in the low-rank branch completely eliminates the per-step runtime overhead
    of complex adapter contractions while preserving 4-bit base weights untouched.
    """
    try:
        delta = None
        # Case 1: ComfyUI WeightAdapterBase with calculate_weight
        if hasattr(adapter, "calculate_weight"):
            shape = getattr(adapter, "shape", None)
            if shape is not None and len(shape) == 2:
                zeros = torch.zeros(shape, dtype=torch.float32, device="cpu")
                delta = adapter.calculate_weight(zeros)
        # Case 2: Tuple / Diff / weights directly
        if delta is None and hasattr(adapter, "weights"):
            weights = adapter.weights
            if weights and isinstance(adapter, comfy.weight_adapter.DiffAdapter):
                delta = weights[0]
            elif isinstance(adapter, comfy.weight_adapter.LoKrAdapter) and len(weights) >= 2:
                w1, w2 = weights[0], weights[1]
                if w1 is not None and w2 is not None:
                    delta = torch.kron(w1.to(torch.float32), w2.to(torch.float32))
            elif isinstance(adapter, comfy.weight_adapter.LoHaAdapter) and len(weights) >= 4:
                w1a, w1b, w2a, w2b = weights[:4]
                if all(w is not None for w in (w1a, w1b, w2a, w2b)):
                    delta = (w1a.to(torch.float32) @ w1b.to(torch.float32)) * (w2a.to(torch.float32) @ w2b.to(torch.float32))
        elif isinstance(adapter, tuple) and len(adapter) >= 1 and torch.is_tensor(adapter[0]):
            delta = adapter[0]

        if delta is None or delta.ndim != 2:
            return None

        out_dim, in_dim = delta.shape
        k = min(max_rank, min(out_dim, in_dim))
        if k <= 0:
            return None

        delta_f32 = delta.to(device="cpu", dtype=torch.float32)
        # SVD: delta_f32 ≈ U @ diag(S) @ V.T
        U, S, V = torch.svd_lowrank(delta_f32, q=k)
        sqrt_s = torch.sqrt(torch.clamp(S, min=0.0))
        # l1 is (out_dim, k), l2 is (k, in_dim)
        l1 = (U * sqrt_s.unsqueeze(0)).to(device=device, dtype=dtype)
        l2 = (sqrt_s.unsqueeze(1) * V.T * strength).to(device=device, dtype=dtype)
        return l1.contiguous(), l2.contiguous()
    except Exception as exc:
        logging.warning("[krea2-svdquant] SVD adapter decomposition skipped: %s", exc)
        return None


def _supports_bypass(adapter) -> bool:
    """True when the adapter can run without touching the base weight.

    ComfyUI's bypass contract is `bypass(f)(x) = g(f(x) + h(x))`, written for exactly our
    situation -- its own docstring says "designed for quantized models where weights may not
    be accessible". An adapter that overrides neither is one ComfyUI can only apply by
    rewriting the weight, which for a QuantizedTensor means dequantize -> add -> requantize.
    """
    base = comfy.weight_adapter.WeightAdapterBase
    return (type(adapter).h is not base.h) or (type(adapter).g is not base.g)


def _bypass_runner(adapter, strength: float):
    """`y -> g(y + h(x, y))` for one adapter, staged the way the low-rank branch is staged.

    ComfyUI's own bypass hook parks adapter weights on the GPU at inject time "to avoid
    per-forward transfers", and a LoKr `w2` (1536x1536) is far bigger than a rank-64 LoRA
    factor, so caching the staged copy looked like the obvious win. Measured on a 3090 at
    1024px it is worth nothing: 2.01 s/step cached against 2.02 s/step restaging every call.
    The cost is the Kronecker math itself -- ~38.7 GMAC per layer, ~0.5 s across 224 layers,
    which is the +0.66 s/step this path adds over a plain LoRA. So this keeps `cast_to` per
    call like `add_low_rank` does, and ComfyUI's VRAM accounting keeps seeing the truth
    instead of ~1 GB of adapter weights it does not know about.

    The shallow copy is because `h()` reads `self.weights`/`self.multiplier` and the adapter
    object is shared with every other patcher holding the same cached LoRA.
    """
    def run(y, x):
        staged = copy.copy(adapter)
        staged.weights = [comfy.model_management.cast_to(w, x.dtype, x.device)
                          if torch.is_tensor(w) else w for w in adapter.weights]
        staged.multiplier = strength
        return staged.g(y + staged.h(x, y))

    return run


def _make_lora_forward(module, factors, adapters):
    """A forward that is "quantized weight + svdq branch (if any) + this LoRA stack".

    Built on `module._krea2_forward` rather than `module.forward` so that it composes with
    the svdq branch but never with a previously installed LoRA patch -- each patcher owns
    the whole LoRA stack and rebuilds it from scratch.

    A no-low-rank checkpoint has quantized layers with no branch and therefore no
    `_krea2_forward`; there the module's own forward already *is* "quantized weight", so it
    is the right base. Object patches are applied in `patch_model`, after this runs, so what
    we capture is the unpatched forward either way.

    `factors` is the folded plain-LoRA branch (one pair of GEMMs for the whole stack);
    `adapters` are everything else -- LoKr, LoHa, OFT -- run through ComfyUI's bypass
    contract `g(f(x) + h(x))`, which never materialises a weight delta.
    """
    base = getattr(module, "_krea2_forward", None) or module.forward
    l1, l2 = factors if factors else (None, None)
    runners = [_bypass_runner(adapter, strength) for adapter, strength in adapters]

    def forward(x, *args, **kwargs):
        y = base(x, *args, **kwargs)
        if l1 is not None:
            y = add_low_rank(y, x, l1, l2)
        for run in runners:
            y = run(y, x)
        return y

    return forward


def _stack_factors(factors: list[tuple[torch.Tensor, torch.Tensor]]):
    """Collapse a LoRA stack for one layer into a single (l1, l2) pair.

    With the strength already folded into each l2, ``sum_i l1_i @ l2_i`` is exactly
    ``cat(l1, dim=1) @ cat(l2, dim=0)``, so an N-LoRA stack costs one pair of GEMMs per
    step instead of N.
    """
    if len(factors) == 1:
        return factors[0]
    l1 = torch.cat([f[0] for f in factors], dim=1)
    l2 = torch.cat([f[1] for f in factors], dim=0)
    return l1.contiguous(), l2.contiguous()


def collect_svdquant_lora(patcher, lora_sd, strength: float, quant_layers: set[str],
                          device, dtype, into_branch: dict, into_adapters: dict,
                          patch_leftover: bool = True, adapter_mode: str = ADAPTER_BYPASS):
    """Route one LoRA: quantized layers into `into_branch`/`into_adapters`, the rest normally.

    `patch_leftover=False` collects the quantized side only and leaves the non-quantized
    layers alone -- see `load_lora` for why the two sides of the stack are handled
    differently.

    Returns (quantized layers claimed, layers patched normally, how many of those normal
    patches landed on a *quantized* weight -- which is the case worth complaining about).
    """
    loaded = comfy.lora.load_lora(lora_sd, _key_map(patcher.model), log_missing=False)

    applied = 0
    leftover = {}
    dequantizing = 0
    for model_key, patch in loaded.items():
        layer = _layer_of(model_key, quant_layers)
        if layer is None:
            leftover[model_key] = patch
            continue
        factors = _plain_lora_factors(patch, strength, device, dtype)
        if factors is not None:
            # A plain LoRA folds into the branch for free in either mode -- one pair of
            # GEMMs for the whole stack -- so `bake` has nothing to win here.
            into_branch.setdefault(layer, []).append(factors)
        elif adapter_mode == ADAPTER_SVD:
            # Attempt SVD decomposition into low-rank factors
            svd_factors = _svd_adapter_factors(patch, strength, device, dtype)
            if svd_factors is not None:
                into_branch.setdefault(layer, []).append(svd_factors)
            elif isinstance(patch, comfy.weight_adapter.WeightAdapterBase) and _supports_bypass(patch):
                into_adapters.setdefault(layer, []).append((patch, strength))
            else:
                leftover[model_key] = patch
                dequantizing += 1
                continue
        elif (adapter_mode == ADAPTER_BYPASS
              and isinstance(patch, comfy.weight_adapter.WeightAdapterBase)
              and _supports_bypass(patch)):
            into_adapters.setdefault(layer, []).append((patch, strength))
        else:
            # `bake` mode, a `diff`/`set`/`bias` patch, or an adapter with no bypass form.
            # ComfyUI can still apply it, but only by rewriting the weight, so the LoRA
            # delta goes through 4-bit quantization along with it.
            leftover[model_key] = patch
            dequantizing += 1
            continue
        applied += 1

    if not patch_leftover:
        return applied, 0, 0

    # Everything the quantized layers did not claim (txtfusion, etc.) goes through
    # ComfyUI's normal LoRA path.
    patched_normally = 0
    if leftover:
        patcher.add_patches(leftover, strength)
        patched_normally = len(leftover)

    return applied, patched_normally, dequantizing


class Krea2SVDQuantLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Output of the Krea2 SVDQuant W4A4 Loader, or any Krea2 "
                               "checkpoint with convrot_w4a4 quantized blocks.",
                }),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "A Krea2 LoRA. Targets 'blocks.N.{attn,mlp}.*' under either "
                               "prefix ('diffusion_model.' or 'transformer.') for the "
                               "quantized blocks; anything else it carries (txtfusion "
                               "etc.) goes through ComfyUI's normal path.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "0 passes the model through untouched. Negative values invert "
                               "the LoRA. Chain more of these nodes to stack LoRAs - the "
                               "whole stack is rebuilt each time, so strengths stay exact.",
                }),
            },
            "optional": {
                "adapters": ([ADAPTER_BYPASS, ADAPTER_SVD, ADAPTER_BAKE], {
                    "default": ADAPTER_BYPASS,
                    "tooltip": "Only affects LoRAs that cannot fold directly into the low-rank "
                               "branch (LoKr, LoHa, OFT); a plain LoRA is free in all modes. "
                               "'bypass' computes the adapter every forward without touching "
                               "the 4-bit weight. 'svd delta' decomposes the delta into low-rank "
                               "factors (zero per-step overhead). 'bake' rewrites the weight once: "
                               "fast, but quantizes the delta to 4 bits.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("The model with the LoRA attached as a parallel branch.",)
    FUNCTION = "load_lora"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant LoRA Loader"
    DESCRIPTION = ("Applies a LoRA to a Krea2 W4A4 quantized model as a parallel low-rank "
                   "branch, leaving the 4-bit weight untouched. The stock loader has to "
                   "dequantize, add and requantize, which puts the LoRA delta through 4-bit "
                   "quantization along with the weight. LoKr/LoHa/OFT cannot fold into the "
                   "branch and run per-forward instead; the 'adapters' input trades that "
                   "exactness back for the stock loader's speed.")

    def load_lora(self, model, lora_name, strength, adapters=ADAPTER_BYPASS):
        if strength == 0:
            return (model,)
        patcher = model.clone()

        # Re-apply the whole stack from scratch rather than appending to whatever the
        # upstream node left behind. That keeps this node idempotent when a strength
        # changes, and still stacks correctly when nodes are chained. The mode rides along
        # per entry, so chaining a `bypass` LoRA and a `bake` one keeps each one's choice.
        stack = list(getattr(model, "krea2_lora_stack", [])) + [(lora_name, strength, adapters)]
        patcher.krea2_lora_stack = stack

        diffusion_model = patcher.model.diffusion_model
        # Every convrot_w4a4 layer needs the parallel-branch treatment, whether or not it
        # already carries an svdq branch. Keying this off `has_branch` instead would come up
        # empty on a no-low-rank checkpoint, silently route the whole LoRA to ComfyUI's
        # normal path, and there it cannot patch a QuantizedTensor weight at all -- the exact
        # failure this node exists to prevent, with no error to show for it.
        quant_modules = dict(quantized_linears(diffusion_model))
        quant_layers = set(quant_modules)
        unbranched = sum(1 for m in quant_modules.values() if not has_branch(m))

        # Attach on the offload device and let the forward stage them: matching whatever
        # device the module happens to be on right now would bake in a placement that
        # ComfyUI is free to change before the first step.
        device = patcher.offload_device
        dtype = patcher.model.get_dtype()

        # The two sides of the stack have to be replayed differently, because the two
        # mechanisms behind them compose differently:
        #
        # * `add_object_patch` is keyed, so writing the whole stack's folded factors for a
        #   layer *replaces* whatever an upstream node put there. Replaying every entry is
        #   what makes a strength change exact rather than cumulative.
        # * `add_patches` *appends*, and `clone()` copies the parent's `patches`. So
        #   replaying an earlier entry's non-quantized layers adds a second copy on top of
        #   the one just inherited: chaining two nodes used to apply the first LoRA's
        #   txtfusion layers twice (measured: 3 patch entries per key instead of 2) while
        #   its quantized layers stayed correct -- the two halves of one LoRA silently
        #   running at different strengths.
        #
        # So only the newest entry patches normally; earlier ones are already inherited.
        branch: dict[str, list] = {}
        bypassed: dict[str, list] = {}
        quantized = normal = dequantizing = 0
        newest = len(stack) - 1
        for i, (name, amount, mode) in enumerate(stack):
            path = folder_paths.get_full_path_or_raise("loras", name)
            q, n, d = collect_svdquant_lora(
                patcher, _load_lora_cached(path), amount, quant_layers, device, dtype,
                branch, bypassed, patch_leftover=(i == newest), adapter_mode=mode)
            if i == newest:
                quantized, normal, dequantizing = q, n, d

        # This guard is about the LoRA this node just added, not the stack total: an upstream
        # LoRA matching plenty of layers must not excuse this one matching none.
        if quantized == 0 and normal == 0:
            raise ValueError(
                "no layer of {} matched this model ({} quantized layers were available); "
                "is it a Krea2 LoRA?".format(lora_name, len(quant_layers))
            )
        # Matching only non-quantized layers is *fine* -- a txtfusion-only adapter is a real
        # thing (`diffusion_model.txtfusion.projector.diff`), and refusing it would be a bug.
        # What is worth saying out loud is a patch that does target a quantized weight in a
        # form with no bypass path, because ComfyUI then rewrites the weight and the delta is
        # requantized to 4 bits along with it -- exactly what this node exists to avoid.
        if dequantizing and adapters == ADAPTER_BYPASS:
            logging.warning(
                "[krea2-svdquant] %s: %d quantized layer(s) carry a patch with no bypass "
                "form (a diff/set patch, or an adapter ComfyUI cannot apply without the "
                "weight). Those go through dequantize -> add -> requantize, so their delta "
                "is quantized to 4 bits too.", lora_name, dequantizing)
        elif dequantizing:
            logging.info(
                "[krea2-svdquant] %s: %d quantized layer(s) baked into the weight on "
                "request -- no per-step cost, delta quantized to 4 bits with the weight.",
                lora_name, dequantizing)

        for layer in set(branch) | set(bypassed):
            module = diffusion_model.get_submodule(layer)
            factors = branch.get(layer)
            patcher.add_object_patch(
                "diffusion_model.{}.forward".format(layer),
                _make_lora_forward(module, _stack_factors(factors) if factors else None,
                                   bypassed.get(layer, ())),
            )

        # The dial is worthless if nobody finds it, and the shape that needs it is exactly
        # the one we can detect: an adapter running per forward on every block.
        if bypassed and adapters == ADAPTER_BYPASS:
            logging.info(
                "[krea2-svdquant] %d layer(s) run a per-forward adapter (LoKr/LoHa/OFT). "
                "That is exact but not free -- measured +1.8 s per model call at 1440x1920. "
                "Set this node's 'adapters' input to %r for the stock loader's speed, or see "
                "tools/bake_adapter.py to bake it in with no runtime cost at all.",
                len(bypassed), ADAPTER_BAKE)

        logging.info("[krea2-svdquant] LoRA stack %s -> %d quantized layers branched "
                     "(%d via bypass adapters); %s added %d quantized, %d normal layers%s",
                     [f"{n}@{a:.2f}" for n, a, _ in stack], len(set(branch) | set(bypassed)),
                     len(bypassed), lora_name, quantized, normal,
                     " (no-low-rank checkpoint: {} quantized layers carry no svdq branch)"
                     .format(unbranched) if unbranched else "")
        return (patcher,)


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantLoraLoader": Krea2SVDQuantLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantLoraLoader": "Krea2 SVDQuant LoRA Loader"}
