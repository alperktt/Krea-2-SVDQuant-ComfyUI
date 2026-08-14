"""Run the quantizer from inside ComfyUI instead of a terminal.

Same code path as ``quantize_krea2.py`` on the command line -- this imports `convert` rather
than reimplementing it, so there is one quantizer, not two that drift.

The honest caveats, which the node's DESCRIPTION repeats because people do not read module
docstrings:

* It blocks the queue. 54 s for a single-shot split, ~5.7 min with the refinement loop, on a
  3090. Nothing else runs meanwhile.
* It needs the GPU to itself, so it unloads whatever ComfyUI is holding first. Your next
  generation will pay a reload.
* It writes ~8 GB.
"""

from __future__ import annotations

import logging
import os
import shutil

import comfy.model_management
import comfy.utils
import folder_paths

from .quantize_krea2 import (
    SAMPLER_HINTS,
    convert,
    derive_out_path,
    resolve_format,
)
from .svdquant_diag import _CATEGORY

# A source has to be dequantized to bf16 before it can be requantized, so leave a margin over
# the ~8 GB output rather than exactly it.
_MIN_FREE_BYTES = 12 * 1024 ** 3


def _free_bytes(path: str) -> int:
    return shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free


class Krea2SVDQuantQuantize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_model": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "The BF16 Krea 2 checkpoint to quantize (~24 GB). An "
                               "already-quantized file cannot be used as a source, except "
                               "FP8, which is unpacked back to BF16 first.",
                }),
                "format": (["svdq", "w4a4", "int8", "fp8"], {
                    "default": "svdq",
                    "tooltip": "svdq: 4-bit weights and activations plus a low-rank bf16 "
                               "correction branch. w4a4: the same without the branch - "
                               "smaller and ~9%% faster per step. int8: the most faithful "
                               "option and still ~2x fp8 on Ampere. fp8: storage only.",
                }),
                "rank": ("INT", {
                    "default": 64, "min": 8, "max": 1024, "step": 8,
                    "tooltip": "svdq only: size of the low-rank branch. Only pays off with "
                               "refine_iters > 0. Without a LoRA, 64 / 128 / 256 measure the "
                               "same, so 64 is enough. With a LoRA loaded, 256 wins clearly "
                               "and 64 loses most of its advantage - so 256 if you use LoRAs.",
                }),
                "rank_alloc": (["uniform", "gqa"], {
                    "default": "uniform",
                    "tooltip": "svdq only: how the rank budget is spread across the eight "
                               "projection types. Same file size either way. uniform gives "
                               "every layer the same rank. gqa moves the budget to attn.wk / "
                               "attn.wv, which absorb ~2x the quantization error at a third of "
                               "the branch cost because Krea 2 has only 12 kv heads. Measured "
                               "and it does not pay: LPIPS 0.3523 vs 0.3403 for uniform, 5 of "
                               "10 prompts better, no mean effect. It does halve the spread "
                               "across prompts and improve the worst one. Leave on uniform "
                               "unless you are re-testing that.",
                }),
                "refine_iters": ("INT", {
                    "default": 100, "min": 0, "max": 200,
                    "tooltip": "svdq only. 0 is a single-shot SVD split (~54s); 100 refines "
                               "the branch against the quantization error and early-stops "
                               "(~5.7min). Keep this on if rank > 16: refinement is what "
                               "makes rank behave. Without it, raising rank costs file size "
                               "and buys nothing measurable.",
                }),
                "groupsize": ("INT", {
                    "default": 256, "min": 32, "max": 1024, "step": 32,
                    "tooltip": "convrot rotation group size. Unused for fp8.",
                }),
                "variant": (["turbo", "base", "unknown"], {
                    "default": "unknown",
                    "tooltip": "Which Krea 2 release this is. Affects only the output "
                               "filename and the recorded metadata - quantization is "
                               "identical; what differs is the sampler settings afterwards.",
                }),
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Filename inside models/diffusion_models/. Leave empty to "
                               "derive it from the variant and format.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off means an existing file of the same name is an error "
                               "rather than 8 GB written over your last run.",
                }),
            },
            # Optional so workflows saved before this input existed keep validating.
            "optional": {
                "act_stats": ("STRING", {
                    "default": "",
                    "tooltip": "svdq only: an activation-statistics file from the Capture "
                               "nodes (a bare filename is looked up in ComfyUI/output/). "
                               "Fits the low-rank branch against measured per-channel "
                               "activation energy instead of assuming it is uniform. Free at "
                               "inference and the best-measured setting here - LPIPS to BF16 "
                               "0.3378 to 0.2825 with no LoRA. Empty means the plain objective.",
                }),
                # Appended after act_stats rather than inserted among the required inputs:
                # ComfyUI matches widgets_values positionally, so anywhere else would shift
                # every value in a workflow saved before this input existed.
                "seed": ("INT", {
                    "default": 0, "min": -1, "max": 0xFFFFFFFF,
                    "tooltip": "svdq only: seed for the randomized low-rank SVD. Quantizing "
                               "twice with the same seed on the same GPU now gives identical "
                               "files; -1 restores the old unseeded behaviour, where it did "
                               "not. The same seed on a different device still differs (~1e-4 "
                               "per weight) - CPU and CUDA do not draw the same numbers.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, source_model, format, variant, rank, rank_alloc, output_name,
                        overwrite):
        """Refuse the obviously-doomed run before the graph starts, not after.

        Everything checked here is a widget, so ComfyUI has it at validation time. That
        matters most in the calibrated graph, where this node sits behind a full sampling
        pass: without this, "that file already exists" arrives several minutes after you
        pressed Queue, having burned the calibration run to tell you.

        Deliberately not checked here: `act_stats`, which can be wired from the Capture Save
        node and therefore has no value yet at validation time. Its checks stay in `run`.
        """
        try:
            src = folder_paths.get_full_path_or_raise("diffusion_models", source_model)
        except Exception as exc:
            return str(exc)
        try:
            _, rank = resolve_format(format, rank, rank_was_set=False)
        except Exception as exc:
            return str(exc)

        if output_name.strip():
            name = output_name.strip()
            if not name.endswith(".safetensors"):
                name += ".safetensors"
            dst = os.path.join(os.path.dirname(src), name)
        else:
            # An act-aware build lands on a different filename, and act_stats is unknowable
            # here. Pass None and the check is against the un-tagged name: it can miss a
            # collision, never invent one, which is the right way for a pre-flight check to
            # be wrong.
            dst, _note = derive_out_path(src, format, rank, variant, rank_alloc, None)

        if os.path.exists(dst) and not overwrite:
            return ("{} already exists. Enable 'overwrite', or set a different output_name."
                    .format(dst))
        free = _free_bytes(dst)
        if free < _MIN_FREE_BYTES:
            return ("only {:.1f} GB free on the drive holding {}; quantizing needs roughly "
                    "{:.0f} GB of headroom.".format(free / 1024 ** 3, os.path.dirname(dst),
                                                    _MIN_FREE_BYTES / 1024 ** 3))
        return True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Writes an ~8 GB file as its side effect, so a cached summary would claim a file
        # exists that the user may have since deleted. Re-queueing is cheap to refuse
        # (overwrite=False still fails fast) and expensive to get wrong.
        return float("nan")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    OUTPUT_TOOLTIPS = ("Where the checkpoint was written, and what went into it.",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant Quantize"
    DESCRIPTION = ("Builds a quantized Krea 2 checkpoint from a BF16 one, without leaving "
                   "ComfyUI. BLOCKS THE QUEUE while it runs (54s to ~6min), unloads any "
                   "loaded model to free the GPU, and writes ~8 GB. Load the result with the "
                   "Krea2 SVDQuant W4A4 Loader (svdq) or the stock UNETLoader (w4a4/int8/fp8).")

    def run(self, source_model, format, rank, rank_alloc, refine_iters, groupsize, variant,
            output_name, overwrite, act_stats="", seed=0):
        src = folder_paths.get_full_path_or_raise("diffusion_models", source_model)

        # `rank` always carries a value from the widget, so "was it set?" cannot be inferred
        # the way argparse does it. Non-svdq formats simply ignore it here rather than
        # erroring, which is the friendlier reading of a dropdown the user cannot un-set.
        fmt, rank = resolve_format(format, rank, rank_was_set=False)

        # Same validation the CLI does for --act-stats: a typed path that silently did nothing
        # would produce a checkpoint indistinguishable from a plain one.
        stats_path = act_stats.strip() or None
        if stats_path is not None:
            if format != "svdq":
                raise RuntimeError("act_stats only applies to format 'svdq': it weights the "
                                   "low-rank branch, and the other formats have no branch.")
            if not os.path.isabs(stats_path):
                stats_path = os.path.join(folder_paths.get_output_directory(), stats_path)
            if not os.path.isfile(stats_path):
                raise RuntimeError("act_stats file not found: {}".format(stats_path))

        if output_name.strip():
            name = output_name.strip()
            if not name.endswith(".safetensors"):
                name += ".safetensors"
            dst = os.path.join(os.path.dirname(src), name)
        else:
            dst, note = derive_out_path(src, format, rank, variant, rank_alloc, stats_path)
            if note:
                logging.info("[krea2-svdquant] %s", note)

        if os.path.exists(dst) and not overwrite:
            raise RuntimeError(
                "{} already exists. Enable 'overwrite', or set a different output_name."
                .format(dst))

        free = _free_bytes(dst)
        if free < _MIN_FREE_BYTES:
            raise RuntimeError(
                "only {:.1f} GB free on the drive holding {}; quantizing needs roughly {:.0f} "
                "GB of headroom.".format(free / 1024 ** 3, os.path.dirname(dst),
                                         _MIN_FREE_BYTES / 1024 ** 3))

        # Quantization wants the card to itself. Without this it competes with whatever the
        # last generation left resident and OOMs on the dequantize-to-bf16 step.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        pbar = comfy.utils.ProgressBar(1)
        state = {"total": 0}

        def progress(done, total, message):
            if total != state["total"]:
                state["total"] = total
                pbar.total = total
            pbar.update_absolute(done, total)

        logging.info("[krea2-svdquant] quantizing %s -> %s (format %s, rank %s/%s, "
                     "refine_iters %s, act_stats %s, seed %s)", src, dst, fmt, rank,
                     rank_alloc, refine_iters if rank else 0, stats_path or "none",
                     "unseeded" if seed < 0 else seed)
        summary = convert(src, dst, fmt, groupsize, "cuda", rank, refine_iters,
                          variant=variant, progress_cb=progress,
                          rank_alloc=rank_alloc if rank else "uniform",
                          act_stats=stats_path, seed=None if seed < 0 else int(seed))

        hint = SAMPLER_HINTS.get(variant)
        loader = ("Krea2 SVDQuant W4A4 Loader" if rank else "the stock UNETLoader")
        text = "\n".join(x for x in (
            summary, "Load it with {}.".format(loader), hint) if x)
        logging.info("[krea2-svdquant] %s", text)
        return {"ui": {"text": [text]}, "result": (text,)}


class Krea2SVDQuantQuantizeAllInOne:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_dit": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "The Krea 2 DiT model: either a BF16 source (~24 GB, will be "
                               "quantized first) or an already-quantized SVDQuant/W4A4/INT8 file.",
                }),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "The Qwen3-VL 4B BF16 text encoder safetensors (~8.88 GB). "
                               "Its 252 linear projections will be quantized to te_format, "
                               "reducing it to ~3.2 GB without losing the vision tower.",
                }),
                "vae": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "The Krea 2 VAE (~0.51 GB, kept unquantized).",
                }),
                "format": (["svdq", "w4a4", "int8", "fp8"], {
                    "default": "svdq",
                    "tooltip": "Diffusion model format. svdq: 4-bit weights and activations "
                               "plus a low-rank bf16 correction branch.",
                }),
                "te_format": (["w4a4", "int8"], {
                    "default": "w4a4",
                    "tooltip": "Text encoder format: w4a4 (convrot_w4a4, ~3.2 GB) or int8 "
                               "(int8_tensorwise, ~5.2 GB).",
                }),
                "rank": ("INT", {
                    "default": 64, "min": 8, "max": 1024, "step": 8,
                    "tooltip": "svdq only: size of the low-rank branch on the diffusion model.",
                }),
                "refine_iters": ("INT", {
                    "default": 100, "min": 0, "max": 200,
                    "tooltip": "svdq only (if source_dit is BF16). Refines the branch against "
                               "quantization error. Ignored if source_dit is already quantized.",
                }),
                "groupsize": ("INT", {
                    "default": 256, "min": 32, "max": 1024, "step": 32,
                    "tooltip": "convrot rotation group size.",
                }),
                "variant": (["turbo", "base", "unknown"], {
                    "default": "turbo",
                    "tooltip": "Which Krea 2 release this is.",
                }),
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Filename in models/checkpoints/. Leave empty to derive automatically.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow overwriting existing output checkpoint.",
                }),
            },
            "optional": {
                "act_stats": ("STRING", {
                    "default": "",
                    "tooltip": "svdq only (if source_dit is BF16): activation statistics file.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": -1, "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for reproducible SVD.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    OUTPUT_TOOLTIPS = ("Size, tensor counts and loader instructions for the baked checkpoint.",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    TITLE = "Krea2 SVDQuant Quantize All-in-One"
    DESCRIPTION = ("Bakes DiT, 4-bit quantized Qwen3-VL 4B text encoder, and VAE into a single "
                   "combined checkpoint in ComfyUI/models/checkpoints/ (~12 GB instead of 15.5 GB "
                   "across three files).")

    def run(self, source_dit, text_encoder, vae, format, te_format, rank, refine_iters,
            groupsize, variant, output_name, overwrite, act_stats="", seed=0):
        from safetensors import safe_open
        from .tools.build_all_in_one import build_all_in_one_checkpoint

        dit_src = folder_paths.get_full_path_or_raise("diffusion_models", source_dit)
        te_src = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        vae_src = folder_paths.get_full_path_or_raise("vae", vae)

        fmt, rank = resolve_format(format, rank, rank_was_set=False)

        # Determine output path in models/checkpoints/
        ckpt_dir = folder_paths.get_folder_paths("checkpoints")[0]
        os.makedirs(ckpt_dir, exist_ok=True)

        if output_name.strip():
            name = output_name.strip()
            if not name.endswith(".safetensors"):
                name += ".safetensors"
            dst = os.path.join(ckpt_dir, name)
        else:
            stem = "Krea2-{}".format(variant.capitalize()) if variant != "unknown" else "Krea2"
            tag = "SVDQuant-W4A4-rank{}".format(rank) if rank else format.upper()
            dst = os.path.join(ckpt_dir, "{}-AllInOne-{}-TE{}.safetensors".format(
                stem, tag, te_format.upper()))

        if os.path.exists(dst) and not overwrite:
            raise RuntimeError(
                "{} already exists. Enable 'overwrite', or set a different output_name."
                .format(dst))

        free = _free_bytes(dst)
        if free < 15 * 1024 ** 3:
            raise RuntimeError(
                "only {:.1f} GB free on the drive holding {}; baking all-in-one needs roughly 15 GB headroom."
                .format(free / 1024 ** 3, dst))

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # Check if DiT is already quantized or needs quantization
        with safe_open(dit_src, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        already_quantized = any(k.endswith(".comfy_quant") for k in keys)
        temp_dit = None

        if not already_quantized:
            stats_path = act_stats.strip() or None
            if stats_path is not None and not os.path.isabs(stats_path):
                stats_path = os.path.join(folder_paths.get_output_directory(), stats_path)
            temp_dit = dst + ".tmp_dit.safetensors"
            logging.info("[krea2-svdquant] quantizing DiT first: %s -> %s", dit_src, temp_dit)
            convert(dit_src, temp_dit, fmt, groupsize, "cuda", rank, refine_iters,
                    variant=variant, rank_alloc="uniform", act_stats=stats_path,
                    seed=None if seed < 0 else int(seed))
            active_dit = temp_dit
        else:
            active_dit = dit_src

        pbar = comfy.utils.ProgressBar(1)
        state = {"total": 0}

        def progress(done, total, message):
            if total != state["total"]:
                state["total"] = total
                pbar.total = total
            pbar.update_absolute(done, total)

        try:
            summary = build_all_in_one_checkpoint(
                dit_path=active_dit,
                text_encoder_path=te_src,
                vae_path=vae_src,
                out_path=dst,
                fmt_name=format,
                te_format=te_format,
                rank=rank,
                variant=variant,
                groupsize=groupsize,
                device="cuda",
                progress_cb=progress,
            )
        finally:
            if temp_dit and os.path.exists(temp_dit):
                try:
                    os.remove(temp_dit)
                except Exception:
                    pass

        hint = SAMPLER_HINTS.get(variant)
        loader = ("Krea2 SVDQuant Checkpoint Loader" if rank else "CheckpointLoaderSimple")
        text = "\n".join(x for x in (
            summary, "Load it with {}.".format(loader), hint) if x)
        logging.info("[krea2-svdquant] %s", text)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "Krea2SVDQuantQuantize": Krea2SVDQuantQuantize,
    "Krea2SVDQuantQuantizeAllInOne": Krea2SVDQuantQuantizeAllInOne,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2SVDQuantQuantize": "Krea2 SVDQuant Quantize",
    "Krea2SVDQuantQuantizeAllInOne": "Krea2 SVDQuant Quantize All-in-One",
}
