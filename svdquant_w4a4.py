"""Loader for SVDQuant-on-native-W4A4 checkpoints (built by ``quantize_krea2.py``).

This is SVDQuant's mechanism (low-rank bf16 branch + 4-bit residual) running on ComfyUI's
own ``convrot_w4a4`` kernel. Because the *activations* are 4-bit here too, the matmul runs
on hardware that is genuinely faster than bf16 -- weight-only 4-bit schemes keep 16-bit
activations and therefore still run at bf16 tensor-core speed.

The checkpoint is self-contained - it carries the quantized blocks *and* the untouched
high-precision layers - so no separate base file is needed. Everything except the 224
block linears loads through ComfyUI's normal path; only those 224 get a low-rank branch
attached on top of the native quantized Linear.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

from .quantize_krea2 import detect_prefix
from .sage_mask_guard import install_mask_guard
from .svdquant_diag import BUF_L1, BUF_L2, _CATEGORY, branch_factors, log_dispatch  # noqa: F401

# The checkpoint keys are the buffer names with a dot in front -- derived rather than
# retyped, because the two being identical is the property the round-trip depends on.
_L1 = "." + BUF_L1
_L2 = "." + BUF_L2


def has_branch(module: torch.nn.Module) -> bool:
    """True once this module carries a low-rank branch (i.e. it is a quantized linear)."""
    return BUF_L1 in getattr(module, "_buffers", {})


def add_low_rank(y: torch.Tensor, x: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor):
    """``y + (x @ l2.T) @ l1.T``, with l1/l2 moved to x's device and dtype as needed.

    `comfy.model_management.cast_to` returns the tensor untouched when it is already in
    the right place, so this is free on a fully-resident model. When ComfyUI has offloaded
    the layer, it performs the same kind of per-call staging copy that ComfyUI itself does
    for the quantized weight -- crucially *without* caching the result back onto the
    module, which would strand the factors on the GPU behind ComfyUI's back.
    """
    a1 = comfy.model_management.cast_to(l1, x.dtype, x.device)
    a2 = comfy.model_management.cast_to(l2, x.dtype, x.device)
    return y + F.linear(F.linear(x, a2), a1)


def _publish_in_state_dict(module: torch.nn.Module) -> None:
    """Make the branch buffers visible to ``state_dict()`` on a quantized Linear.

    Registering them as persistent buffers is not enough. ComfyUI's quantized Linear
    (``comfy.ops.mixed_precision_ops``) replaces ``state_dict`` wholesale with a body that
    emits weight/scale/marker/bias and nothing else -- buffers never reach it. That
    matters because ``model_management.module_size()`` sums ``state_dict()``, and every
    per-module VRAM decision ComfyUI makes (the lowvram split in particular) is derived
    from that number. Left alone, ~2.9 MB per layer and ~645 MB across the 224 blocks
    would sit on the GPU while the budget believed it was free -- which is exactly how an
    8 GB card OOMs on this checkpoint but not on the branch-free int8 one.

    Emitting them under their own names also means a model saved out of ComfyUI carries
    the same ``<layer>.svdq_l1`` keys quantize_krea2.py writes, so it round-trips.
    """
    if getattr(module, "_krea2_state_dict_patched", False):
        return
    inner = module.state_dict

    def state_dict(*args, destination=None, prefix="", **kwargs):
        sd = inner(*args, destination=destination, prefix=prefix, **kwargs)
        for name in (BUF_L1, BUF_L2):
            buf = module._buffers.get(name)
            if buf is not None:
                sd["{}{}".format(prefix, name)] = buf
        return sd

    module.state_dict = state_dict
    module._krea2_state_dict_patched = True


def attach_branch(module: torch.nn.Module, l1: torch.Tensor, l2: torch.Tensor,
                  scale: float = 1.0) -> None:
    """Add ``+ (x @ l2.T) @ l1.T * scale`` to a module's output, in place.

    The module is *not* replaced: swapping it for a wrapper would push its weight down a
    level in the state dict (``blocks.0.attn.wq.base.weight``), and every LoRA key map in
    ComfyUI expects ``blocks.0.attn.wq.weight``. Keeping the module identity keeps those
    paths - and therefore the rest of the ecosystem - intact.

    The factors are registered as persistent buffers *and* published into ``state_dict()``
    by `_publish_in_state_dict` -- on these modules persistence alone is not enough, see
    that function for why. Both steps are needed to get them counted by
    ``model_management.module_size()``, which is what every VRAM decision (full load, the
    lowvram split, how much to free) is derived from. Uncounted, they are roughly 2.9 MB
    per layer and 645 MB across the 224 blocks: the difference between fitting and OOMing
    on an 8 GB card.

    `scale` is folded into l2 at attach time rather than applied to the output every step:
    l2 is [rank, in_features] while the output is [tokens, out_features], so this is about
    three orders of magnitude less work and one fewer full-size allocation per call.
    """
    if has_branch(module):
        raise RuntimeError("module already carries a low-rank branch")

    if scale != 1.0:
        l2 = l2 * scale
    module.register_buffer(BUF_L1, l1.contiguous(), persistent=True)
    module.register_buffer(BUF_L2, l2.contiguous(), persistent=True)
    _publish_in_state_dict(module)

    original = module.forward

    def forward(x, *args, **kwargs):
        y = original(x, *args, **kwargs)
        factors = branch_factors(module)
        if factors is None:
            return y
        return add_low_rank(y, x, *factors)

    module.forward = forward
    # Kept as a stable handle so a LoRA can build on top of "quantized weight + svdq
    # branch" without having to trust whatever is currently in `module.forward` -- which
    # ComfyUI's object-patch machinery swaps in and out around sampling.
    module._krea2_forward = forward


def _get_submodule(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    """Walk a dotted path, reporting the layer name rather than the raw attribute error.

    Left to `getattr`/`__getitem__` this surfaces as a bare ``AttributeError: 'ModuleList'
    object has no attribute 'nope'`` or ``IndexError: index 999 is out of range``, which says
    nothing about which checkpoint key failed to map.
    """
    module = root
    for i, p in enumerate(dotted.split(".")):
        try:
            module = module[int(p)] if p.isdigit() else getattr(module, p)
        except (AttributeError, IndexError, KeyError) as exc:
            raise RuntimeError(
                "checkpoint refers to layer {!r}, which this model does not have "
                "(failed at {!r}): {}".format(dotted, ".".join(dotted.split(".")[:i + 1]), exc)
            ) from exc
    return module


def _shield_from_dynamo(module: torch.nn.Module) -> None:
    """Fallback path: let torch.compile skip the quantized kernel instead of failing on it.

    Dynamo cannot trace ``F.linear(x, QuantizedTensor)`` -- comfy_kitchen dispatches that
    through ``__torch_dispatch__`` into a C extension that wants real pointers, so
    fake-tensor tracing raises. Marking the call as a graph break lets inductor still fuse
    everything *around* it (norms, modulation, RoPE), which is a third of the step time.

    Measured cost of doing it this way: **two graph breaks per quantized layer**, 448 across
    the 224 blocks (`diagnose.py --mode compile`). Inductor never sees two consecutive
    layers in one graph and cudagraphs is off entirely. `_install_custom_op` is the path
    that avoids this; this one remains for when the kitchen layout it depends on has moved.

    Call order matters and is easy to get backwards: this must run *before*
    `attach_branch`, so that what gets wrapped is the real `nn.Linear.forward` and the
    branch closure installed on top of it stays traceable.
    """
    try:
        module.forward = torch._dynamo.disable(module.forward)
    except Exception as exc:
        # Not fatal on its own -- the model runs fine uncompiled. But if this silently
        # no-ops, a later TorchCompileModel dies inside the comfy_kitchen kernel with a
        # fake-tensor error that points nowhere near here, so leave a trail.
        logging.debug("[krea2-svdquant] could not shield %s from dynamo: %s",
                      type(module).__name__, exc)


# The opaque op. Registering the kernel call under `torch.library` is the whole trick:
# Dynamo does not try to trace *into* a custom op, it emits a single node for it, so the
# 224 linears stop being graph breaks and a compiled step becomes one graph. The kitchen
# call inside is byte-for-byte the one `_convrot_w4a4_forward` makes, so numerics are
# unchanged -- this moves where the call is visible from, not what it computes.
#
# Built lazily and guarded: everything it depends on (the backend registry,
# `TensorCoreConvRotW4A4Layout.get_plain_tensors`, the `_params` field names) is
# comfy_kitchen's internal API, not a published one. If a kitchen update moves any of it,
# `_w4a4_op()` returns None and the loader falls back to `_shield_from_dynamo` rather than
# failing to load a checkpoint.
_W4A4_OP = None
_W4A4_OP_ERROR = None

# Backend resolution, memoized. `convrot_w4a4_linear` re-runs it on *every call*: it builds a
# seven-key dict, walks `["cuda", "triton", "eager"]` and revalidates seven ParamConstraints,
# with no caching anywhere in `BackendRegistry`. At 224 layers times 8 steps that is 1792
# resolutions per image, all of them answering the same question.
#
# The key is exactly what those constraints read -- dtypes, device and rank, since the only
# shape rule on `x` is `MinDims(2)` and the ones on `qweight`/`wscales` are satisfied
# identically by every layer in the model. Token count is deliberately *not* in the key: it
# never reaches a constraint, only kernel selection inside the implementation.
#
# Not invalidated, because the thing that would change the answer -- ComfyUI's
# `ck.registry.disable("cuda")` under a pre-cu130 torch -- happens once, at
# `comfy.quant_ops` import, long before any of this runs.
_IMPL_CACHE: dict = {}


def _resolve_impl(x, qweight, wscales, bias, convrot_groupsize, quant_group_size,
                  linear_dtype):
    from comfy_kitchen.registry import registry as ck_registry

    key = (x.dtype, x.device, x.ndim, qweight.dtype, wscales.dtype,
           None if bias is None else bias.dtype,
           convrot_groupsize, quant_group_size, linear_dtype)
    impl = _IMPL_CACHE.get(key)
    if impl is None:
        impl = ck_registry.get_implementation("convrot_w4a4_linear", kwargs={
            "x": x, "qweight": qweight, "wscales": wscales, "bias": bias,
            "convrot_groupsize": convrot_groupsize,
            "quant_group_size": quant_group_size, "linear_dtype": linear_dtype,
        })
        _IMPL_CACHE[key] = impl
    return impl


def _w4a4_op():
    """The registered `krea2::w4a4_linear` op, or None if this kitchen build cannot host it."""
    global _W4A4_OP, _W4A4_OP_ERROR
    if _W4A4_OP is not None or _W4A4_OP_ERROR is not None:
        return _W4A4_OP

    try:
        # Imported for its side effect on this try block: no kitchen registry means no
        # implementation to dispatch to, and finding that out here is what makes the
        # fallback a load-time decision instead of a crash on the first forward.
        from comfy_kitchen.registry import registry as ck_registry  # noqa: F401

        @torch.library.custom_op("krea2::w4a4_linear", mutates_args=())
        def w4a4_linear(x: torch.Tensor, qweight: torch.Tensor, wscales: torch.Tensor,
                        bias: torch.Tensor | None, convrot_groupsize: int,
                        quant_group_size: int, linear_dtype: str) -> torch.Tensor:
            impl = _resolve_impl(x, qweight, wscales, bias, convrot_groupsize,
                                 quant_group_size, linear_dtype)
            return impl(x, qweight, wscales, bias=bias,
                        convrot_groupsize=convrot_groupsize,
                        quant_group_size=quant_group_size, linear_dtype=linear_dtype)

        @w4a4_linear.register_fake
        def _(x, qweight, wscales, bias, convrot_groupsize, quant_group_size, linear_dtype):
            # qweight is [out_features, in_features // 2] -- int4 packed two to a byte, so
            # the output width is its *row* count and cannot be read off the last dim.
            return x.new_empty(x.shape[:-1] + (qweight.shape[0],))

        _W4A4_OP = w4a4_linear
    except Exception as exc:
        _W4A4_OP_ERROR = "{}: {}".format(type(exc).__name__, exc)
        logging.info("[krea2-svdquant] no compile-friendly kernel op (%s); falling back to "
                     "graph breaks around the quantized linears", _W4A4_OP_ERROR)
    return _W4A4_OP


def _install_custom_op(module: torch.nn.Module) -> bool:
    """Route this Linear's matmul through `krea2::w4a4_linear`. True if it took.

    The fast path deliberately handles only the case ComfyUI's own quantized forward calls
    "quantized": no LoRA weight/bias function, weight resident on the input's device, not
    forced to full precision. Those are the same conditions `comfy/ops.py` gates
    `_use_quantized` on, and the reason is the same -- anything else means the weight is
    being rewritten or staged per call, which an op holding plain tensors cannot see. All
    of it is re-checked *per forward* rather than at load, because ComfyUI attaches lowvram
    patches and offloads weights long after this runs; when a check fails the call goes to
    the stock forward and is simply a graph break, i.e. no worse than the old behaviour.

    `params.transposed` is checked once here rather than per call: a transposed weight makes
    kitchen dequantize and run a bf16 linear (`_handle_convrot_w4a4_linear`), which is a
    different computation, not a slower one. A checkpoint whose weights arrive transposed
    should keep whatever kitchen does with it.
    """
    # An escape hatch, and the only honest way to A/B this: the two paths cannot coexist in
    # one process, so the comparison is one server run against another, and a flag is what
    # makes those two runs differ by exactly this decision.
    if os.environ.get("KREA2_W4A4_OP") == "0":
        return False
    # Idempotent: the loader installs this at load time, the compile-prep node can be asked
    # to do it again on a model that already has it, and wrapping a wrapper would put the
    # guard block on the hot path twice for no benefit.
    if getattr(module, "_krea2_op_installed", False):
        return True

    op = _w4a4_op()
    if op is None:
        return False

    try:
        from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout
        params = module.weight._params
        if params.transposed:
            return False
        groupsize = int(params.convrot_groupsize)
        quant_group_size = int(params.quant_group_size)
        linear_dtype = str(params.linear_dtype)
    except Exception as exc:
        logging.debug("[krea2-svdquant] cannot read layout params off %s: %s",
                      type(module).__name__, exc)
        return False

    original = module.forward
    get_plain = TensorCoreConvRotW4A4Layout.get_plain_tensors

    def forward(x, *args, **kwargs):
        weight = module.weight
        if (args or kwargs or x.ndim < 2 or x.requires_grad
                or module.weight_function or module.bias_function
                or getattr(module, "comfy_force_cast_weights", False)
                or getattr(module, "_full_precision_mm", False)
                or weight._qdata.device != x.device):
            return original(x, *args, **kwargs)
        qweight, wscales = get_plain(weight)
        bias = module.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        # The op exists for Dynamo's benefit; an eager call gains nothing from routing
        # through `torch.library` and can skip that dispatch. `is_compiling()` is
        # constant-folded during tracing, so the compiled graph still gets the op node --
        # the same mechanism ComfyUI uses to keep `run_every_op` out of compiled graphs
        # (`comfy/ops.py`). Measured either way it is within noise in eager; what the
        # memoized `_resolve_impl` below is worth is not (0.846 -> 0.833 s/step on a 3090 at
        # 1024px, rank 256, against `KREA2_W4A4_OP=0`).
        if torch.compiler.is_compiling():
            return op(x, qweight, wscales, bias, groupsize, quant_group_size, linear_dtype)
        impl = _resolve_impl(x, qweight, wscales, bias, groupsize, quant_group_size,
                             linear_dtype)
        return impl(x, qweight, wscales, bias=bias, convrot_groupsize=groupsize,
                    quant_group_size=quant_group_size, linear_dtype=linear_dtype)

    module.forward = forward
    module._krea2_op_installed = True
    return True


def load_svdquant_w4a4(path: str, model_options: dict | None = None,
                       compile_safe: bool = True):
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    # `comfy.sd.load_diffusion_model_state_dict` strips this prefix internally when it builds
    # the module tree, but we walk that tree ourselves to attach branches, so we strip it too.
    # `default=""` rather than raising: a loader must not blow up on a checkpoint with no
    # blocks, and the "found no branches" error below says far more about what went wrong.
    layer_prefix = detect_prefix(sd.keys(), default="")

    branches: dict[str, dict[str, torch.Tensor]] = {}
    for key in list(sd.keys()):
        for suffix, slot in ((_L1, "l1"), (_L2, "l2")):
            if key.endswith(suffix):
                branches.setdefault(key[: -len(suffix)], {})[slot] = sd.pop(key)
                break
    if not branches:
        raise ValueError(
            "{} carries no svdq_l1/svdq_l2 tensors - it is a plain quantized checkpoint, "
            "load it with UNETLoader instead".format(path)
        )

    # `disable_dynamic=True` pins this to the classic ModelPatcher. On current ComfyUI
    # that is a no-op (`CoreModelPatcher is ModelPatcher`), but when upstream flips it to
    # ModelPatcherDynamic the streaming patcher takes ownership of the weights via
    # `load_model_weights(..., assign=patcher.is_dynamic())`, which we have not validated
    # against the branch buffers. Keep it pinned until that path is tested; the
    # diagnostics node reports which patcher is actually in use.
    patcher = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options=model_options or {}, metadata=metadata, disable_dynamic=True
    )
    if patcher is None:
        raise RuntimeError("could not detect a model in {}".format(path))

    diffusion_model = patcher.model.diffusion_model
    attached = 0
    via_op = 0
    ranks: set[int] = set()
    incomplete = []
    for layer, parts in branches.items():
        if "l1" not in parts or "l2" not in parts:
            incomplete.append(layer)
            continue
        submodule_path = layer[len(layer_prefix):] if layer_prefix else layer
        base = _get_submodule(diffusion_model, submodule_path)
        if compile_safe:
            # The op keeps the layer inside the graph; the shield takes it out of one. Only
            # one of the two can be installed, and the op is tried first.
            if _install_custom_op(base):
                via_op += 1
            else:
                _shield_from_dynamo(base)
        attach_branch(base, parts["l1"], parts["l2"])
        # Collected per layer rather than read once off the first branch, so a checkpoint
        # with a non-uniform rank budget reports honestly instead of quoting layer zero.
        ranks.add(int(parts["l1"].shape[1]))
        attached += 1

    # Silently returning a patcher with no branches would hand back a plain quantized model
    # dressed as an SVDQuant one -- same class of failure quantize_krea2.py hard-fails on.
    if attached == 0:
        raise RuntimeError(
            "{}: found {} svdq_l1/svdq_l2 key pairs but attached none of them"
            "{}. The checkpoint's layer names do not line up with this model "
            "(detected prefix {!r}).".format(
                path, len(branches),
                "; {} were missing their other half".format(len(incomplete))
                if incomplete else "",
                layer_prefix)
        )

    # The branch buffers were registered after the patcher computed (and cached) its size,
    # so drop the cache and let `model_size()` re-derive it from the state dict.
    patcher.size = 0

    install_mask_guard(patcher)

    # Metadata is a newer addition; checkpoints published before it still load, with the
    # rank recovered from the factor shape exactly as before. The shapes are the ground
    # truth, so they win over a metadata value that disagrees with them.
    meta = metadata or {}
    rank_desc = (str(next(iter(ranks))) if len(ranks) == 1
                 else "{}-{} mixed".format(min(ranks), max(ranks)))
    meta_rank = meta.get("krea2_svdquant_rank")
    if meta_rank and len(ranks) == 1 and int(meta_rank) != next(iter(ranks)):
        logging.warning("[krea2-svdquant] metadata says rank %s but the factors are rank %s; "
                        "trusting the factors", meta_rank, rank_desc)
    variant = meta.get("krea2_svdquant_variant", "unknown")
    # Which of the two compile strategies each layer got is worth stating on every load, not
    # just when it fails: "torch.compile barely helped" and "the op path silently fell back
    # to graph breaks" look identical from the outside otherwise.
    if not compile_safe:
        compile_desc = "compile shielding off"
    elif via_op == attached:
        compile_desc = "compile: {} layers in-graph via krea2::w4a4_linear".format(via_op)
    elif via_op:
        compile_desc = "compile: {} layers in-graph, {} as graph breaks".format(
            via_op, attached - via_op)
    else:
        compile_desc = "compile: all {} layers are graph breaks{}".format(
            attached, " ({})".format(_W4A4_OP_ERROR) if _W4A4_OP_ERROR else "")
    summary = ("w4a4 + low-rank: attached {} branches (rank {}, variant {}), "
               "model_size {:.2f} GiB, {}".format(
                   attached, rank_desc, variant, patcher.model_size() / 1024 ** 3,
                   compile_desc))
    logging.info("[krea2-svdquant] %s", summary)

    dispatch = log_dispatch(diffusion_model)

    # Stashed rather than returned so callers that just want the model (diagnose.py, the
    # head-to-head scripts) keep working unchanged. The node surfaces it in the UI, which
    # matters most for the dispatch warning: buried in the console, the people who most need
    # to read it are exactly the ones who never see it.
    patcher.krea2_load_summary = "\n".join(x for x in (summary, dispatch) if x)
    return patcher


class Krea2SVDQuantW4A4Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "A checkpoint from quantize_krea2.py --format svdq (it carries "
                               "*.svdq_l1/*.svdq_l2 tensors). The --format w4a4 / int8 / fp8 "
                               "checkpoints have no branch and load with the stock UNETLoader "
                               "instead.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    OUTPUT_TOOLTIPS = ("Wire this to a KSampler.",
                       "Rank, variant, size and which kernel the quantized layers will "
                       "actually dispatch to. Read this if generation is slow.")
    OUTPUT_NODE = True
    FUNCTION = "load"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant W4A4 Loader"
    DESCRIPTION = ("Loads a W4A4 + low-rank (SVDQuant) Krea2 checkpoint. Self-contained: "
                   "no separate base model needed. The status output tells you whether the "
                   "fast int4 kernel is in play.")

    def load(self, model_name):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        patcher = load_svdquant_w4a4(path)
        status = getattr(patcher, "krea2_load_summary", "")
        return {"ui": {"text": [status]}, "result": (patcher, status)}


NODE_CLASS_MAPPINGS = {"Krea2SVDQuantW4A4Loader": Krea2SVDQuantW4A4Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2SVDQuantW4A4Loader": "Krea2 SVDQuant W4A4 Loader"}
