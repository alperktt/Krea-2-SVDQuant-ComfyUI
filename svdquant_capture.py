"""Capture per-input-channel activation statistics, for an activation-aware low-rank split.

`quantize_krea2.py`'s refinement loop minimises **weight** reconstruction error,
``||W - (Q + L1 L2)||_F``. That is the true output error only if the input covariance is the
identity -- the assumption its own docstring admits to. DeepCompressor instead minimises the
error *through real activations*, which is most of why a calibrated W4A4 build lands closer to
BF16 than a calibration-free one, and it is the last untried quality lever that costs nothing
at runtime: same rank, same format, same file size, same kernel.

Weighting the split by per-input-channel activation RMS approximates that objective. A channel
the model drives hard deserves more of the branch's capacity than one that is usually near
zero, and plain Frobenius spends them equally.

Why this is a pair of ComfyUI nodes rather than a standalone script: the statistics have to
come from real sampling, and real sampling needs the Qwen3-VL conditioning, the scheduler and
the whole ComfyUI graph. Reimplementing that outside ComfyUI to run a calibration pass would
be a second, subtly different sampler. Instead, put `Start` before the KSampler and `Save`
after it, run whatever prompts you like, and the hooks ride along on the real thing.

Accumulators persist across queued prompts on purpose -- calibration wants several prompts,
and each ComfyUI run is a separate execution. `Start` with `reset=True` on the first one.
"""

from __future__ import annotations

import logging
import os

import torch

import folder_paths

from .quantize_krea2 import ALL_QUANT_SUFFIXES as _QUANT_SUFFIXES
from .svdquant_diag import _CATEGORY

# layer name -> {"sumsq": float64 [in_features] on cpu, "count": int}
_STATS: dict[str, dict] = {}
_HANDLES: list = []
# Checked first inside every hook, so a hook that outlives its handle is inert rather than
# quietly accumulating. A leaked hook is the worst failure this module has: it would keep
# adding activations from later, unrelated generations into _STATS, and the next calibration
# file would be contaminated with no sign of it.
_ACTIVE = False

FILE_VERSION = 1


def _hook(name: str):
    def pre_hook(module, args):
        if not _ACTIVE or not args:
            return
        x = args[0]
        if not isinstance(x, torch.Tensor) or x.ndim < 2:
            return
        # [..., in_features] -> per-channel sum of squares over every token seen. float64 on
        # the accumulator because this sums millions of terms across a whole calibration run
        # and float32 stops being able to represent the increments long before that.
        flat = x.detach().reshape(-1, x.shape[-1])
        sq = flat.float().pow(2).sum(dim=0).double().cpu()
        entry = _STATS.get(name)
        if entry is None:
            _STATS[name] = {"sumsq": sq, "count": flat.shape[0]}
        else:
            entry["sumsq"] += sq
            entry["count"] += flat.shape[0]
    return pre_hook


def is_target(name: str) -> bool:
    """The same layer set `quantize_krea2.is_target` picks, matched on a live module tree.

    It cannot be reused directly: that one takes checkpoint keys (`blocks.0.attn.wq.weight`,
    optionally prefixed) while `named_modules()` yields module paths. The rule is identical
    though -- inside `blocks.`, ending in one of the eight projection leaves -- and the
    suffix tuple is imported rather than retyped so the two cannot drift apart.

    Deliberately *not* keyed on the weight being quantized: the statistics are supposed to
    come from the BF16 model, where no weight is a QuantizedTensor yet.
    """
    # The union across every architecture in `quantize_krea2.ARCHITECTURES`, not one
    # model's set: this matches against a live module tree, so a name that no model has
    # simply never appears. Matching the union is what lets the same capture nodes serve
    # Krea 2 and MiniMax H3 without knowing which one is loaded.
    return name.startswith("blocks.") and name.endswith(_QUANT_SUFFIXES)


def attach(diffusion_model) -> int:
    global _ACTIVE
    detach()
    count = 0
    for name, module in diffusion_model.named_modules():
        if is_target(name) and getattr(module, "weight", None) is not None:
            _HANDLES.append(module.register_forward_pre_hook(_hook(name)))
            count += 1
    _ACTIVE = True
    return count


def detach() -> None:
    """Stop capturing, then remove the hooks.

    `_ACTIVE` goes down first so capture stops even for a hook whose handle refuses to
    detach. A failed handle is kept in `_HANDLES` so the next `detach()` retries it, and the
    failure is logged at warning -- at DEBUG (ComfyUI's default is INFO) nobody would ever
    see the one condition that can silently corrupt a calibration file.
    """
    global _ACTIVE
    _ACTIVE = False
    kept = []
    for handle in _HANDLES:
        try:
            handle.remove()
        except Exception:
            logging.warning("[krea2-svdquant] could not remove a capture hook; capture is "
                            "off but the hook is still installed", exc_info=True)
            kept.append(handle)
    _HANDLES[:] = kept


def write(path: str) -> dict:
    """RMS per input channel, as a safetensors file keyed by layer name."""
    from safetensors.torch import save_file

    if not _STATS:
        raise RuntimeError(
            "no activation statistics were collected. The Start node must run before the "
            "KSampler and the Save node after it -- if Save is not downstream of the "
            "sampler's output, ComfyUI is free to execute it first.")

    tensors = {}
    for name, entry in _STATS.items():
        rms = (entry["sumsq"] / max(1, entry["count"])).sqrt().float()
        tensors[name] = rms.contiguous()

    counts = [e["count"] for e in _STATS.values()]
    # Every target layer is called exactly once per forward and sees the same token count, so
    # a clean capture ends with all 224 counts identical -- our own reference run was
    # 264200/264200. A spread means some layers accumulated activations the others did not,
    # and there is one way that happens: `detach()` only runs from the Save node, so a graph
    # that errors between Start and Save leaves the hooks live, and whatever is generated
    # before the next Start lands in here. `attach()` detaches first, which bounds the window
    # but does not clear what was already added. Loud, not fatal: the statistics are still
    # usable, just no longer only about what you meant to calibrate on.
    if min(counts) != max(counts):
        logging.warning(
            "[krea2-svdquant] uneven token counts across layers (%d to %d). Capture stayed "
            "live through something other than the calibration run -- most likely a graph "
            "that errored between the Start and Save nodes. Re-run the capture with "
            "reset=True on the first prompt if these statistics are meant to describe one "
            "model.", min(counts), max(counts))
    metadata = {
        "krea2_actstats_version": str(FILE_VERSION),
        "krea2_actstats_layers": str(len(tensors)),
        "krea2_actstats_min_tokens": str(min(counts)),
        "krea2_actstats_max_tokens": str(max(counts)),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tensors, path, metadata=metadata)
    return {"layers": len(tensors), "min_tokens": min(counts), "max_tokens": max(counts),
            "path": path}


def summary() -> str:
    if not _STATS:
        return "no activation statistics collected yet"
    counts = [e["count"] for e in _STATS.values()]
    return "{} layers, {}-{} tokens each".format(len(_STATS), min(counts), max(counts))


class Krea2SVDQuantCaptureStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The BF16 model, ideally. Statistics taken "
                                               "from a quantized model describe the "
                                               "quantized model's activations, which is the "
                                               "thing being corrected."}),
                "reset": ("BOOLEAN", {"default": True,
                                      "tooltip": "Clear anything gathered so far. Leave it "
                                                 "off to accumulate across several prompts, "
                                                 "which is the point of calibrating."}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    OUTPUT_TOOLTIPS = ("Wire this to the KSampler.",
                       "How many layers are being watched.")
    OUTPUT_NODE = True
    FUNCTION = "start"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant Capture Start"
    DESCRIPTION = ("Watches every quantized linear and records the RMS of its inputs during "
                   "sampling. Feeds --act-stats in quantize_krea2.py.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Attaching hooks is a side effect, so this must never be served from cache.
        return float("nan")

    def start(self, model, reset):
        if reset:
            _STATS.clear()
        attached = attach(model.model.diffusion_model)
        if attached == 0:
            raise RuntimeError(
                "found no layers to watch: nothing under 'blocks.' ends in one of {}. "
                "Capture targets exactly the layers quantize_krea2.py quantizes, so a model "
                "matching none of them is not one these statistics would apply to."
                .format(", ".join(_QUANT_SUFFIXES)))
        status = "capturing {} layers ({})".format(attached, summary())
        logging.info("[krea2-svdquant] %s", status)
        return {"ui": {"text": [status]}, "result": (model, status)}


class Krea2SVDQuantCaptureSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Wire the KSampler's output here. This is "
                                                 "what forces the node to run *after* "
                                                 "sampling rather than before it."}),
                "filename": ("STRING", {"default": "krea2_act_stats.safetensors",
                                        "tooltip": "Written under ComfyUI/output/."}),
                "keep_capturing": ("BOOLEAN", {"default": False,
                                               "tooltip": "Leave the hooks attached so the "
                                                          "next queued prompt keeps adding "
                                                          "to the same statistics."}),
            }
        }

    # `act_stats_path` is appended rather than inserted so workflows saved against the
    # two-output version keep their links: slot indices 0 and 1 still mean what they did.
    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "status", "act_stats_path")
    OUTPUT_TOOLTIPS = ("Passed through unchanged.", "Where the file went and what is in it.",
                       "Absolute path of the file just written. Wire it into the Quantize "
                       "node's act_stats: that is also what forces the quantizer to run "
                       "after the calibration pass rather than before it.")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant Capture Save"
    DESCRIPTION = ("Writes the captured activation RMS to ComfyUI/output/. Must be "
                   "downstream of the KSampler.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def save(self, latent, filename, keep_capturing):
        # Relative to ComfyUI/output/, which is what the tooltip promises. `os.path.join`
        # alone does not keep that promise: hand it an absolute path and it discards the
        # first argument, and `write()` then happily creates whatever directory it names.
        # The sibling Quantize node already resolves `act_stats` this way, so make the two
        # agree about what a bare filename means.
        filename = filename.strip()
        if not filename:
            raise ValueError("filename is empty")
        if os.path.isabs(filename) or os.path.splitdrive(filename)[0]:
            raise ValueError(
                "filename must be relative to ComfyUI/output/, got an absolute path: {}"
                .format(filename))
        out_dir = folder_paths.get_output_directory()
        path = os.path.normpath(os.path.join(out_dir, filename))
        if os.path.commonpath([os.path.abspath(out_dir), os.path.abspath(path)]) != \
                os.path.abspath(out_dir):
            raise ValueError(
                "filename must stay under ComfyUI/output/, got: {}".format(filename))
        info = write(path)
        if not keep_capturing:
            detach()
        status = ("wrote {layers} layers to {path}\n"
                  "tokens per layer: {min_tokens}-{max_tokens}{more}").format(
            more="" if keep_capturing else "\nhooks detached", **info)
        logging.info("[krea2-svdquant] %s", status.replace("\n", "  "))
        return {"ui": {"text": [status]}, "result": (latent, status, info["path"])}


NODE_CLASS_MAPPINGS = {
    "Krea2SVDQuantCaptureStart": Krea2SVDQuantCaptureStart,
    "Krea2SVDQuantCaptureSave": Krea2SVDQuantCaptureSave,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2SVDQuantCaptureStart": "Krea2 SVDQuant Capture Start",
    "Krea2SVDQuantCaptureSave": "Krea2 SVDQuant Capture Save",
}
