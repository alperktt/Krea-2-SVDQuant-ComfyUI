"""Bake a LoRA into a Krea2 checkpoint *before* it is quantized.

    python tools/bake_adapter.py raw.safetensors --lora Krea2/realism_engine.safetensors \
        --format svdq --rank 256 --variant base --act-stats krea2_act_stats_base.safetensors

Why this exists. A LoKr, LoHa or OFT cannot fold into the low-rank branch, so the node has
two choices at runtime and neither is free: compute the adapter every forward (exact, and on
a 3090 at 1440x1920 that is +1.8 s per model call), or hand it to ComfyUI, which rewrites the
4-bit weight and requantizes the LoRA delta along with it (free, and lossy).

Baking before quantization avoids the choice. The delta is added to the **bf16** weight, and
then the SVDQuant split runs on the merged weight -- so the low-rank branch is fitted against
what you will actually sample with, and absorbs the error that the naive
dequantize-add-requantize path throws away. At runtime the LoRA is simply gone: no adapter,
no branch of its own, no per-step cost at all.

The delta is computed by ComfyUI's own weight adapters (`adapter.calculate_weight`), so every
format and naming convention ComfyUI supports works here, alpha and dora_scale included.

The merged weights are never written out whole -- the source is 24 GB -- they are streamed
into `convert()` through its `weight_patch` hook.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # the node package

import torch  # noqa: E402

# `quantize_krea2` before `comfy.*`, and that order is the point. It runs
# `_find_comfyui_root()` at import -- $COMFYUI_PATH, then two levels up from itself, then a
# couple of fallbacks -- so importing it is what puts ComfyUI on the path. This file used to
# do that itself with a bare `dirname(dirname(dirname(HERE)))`, which is correct only when
# the pack sits in `ComfyUI/custom_nodes/`: from a working copy anywhere else, even `--help`
# died on `No module named 'comfy'` with nothing to say about why.
from quantize_krea2 import (  # noqa: E402
    LAYER_PREFIXES,
    convert,
    derive_out_path,
    detect_prefix,
    resolve_format,
)

import comfy.lora  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.weight_adapter  # noqa: E402

_PREFIX = "diffusion_model."
_ALT_PREFIX = "transformer."


def build_key_map(keys, prefix: str) -> dict:
    """`{lora key: checkpoint key}` for a bare diffusion-model state dict.

    ComfyUI's `model_lora_keys_unet` needs a live model; here there is only a file. The
    mapping it would produce for the native names is mechanical, and the diffusers names come
    from the same table ComfyUI uses (`krea2_to_diffusers`), so both stay in step with it.
    """
    key_map = {}
    for key in keys:
        if not key.endswith(".weight"):
            continue
        name = key[: -len(".weight")]
        bare = name[len(prefix):] if prefix and name.startswith(prefix) else name
        for alias in (_PREFIX + bare, _ALT_PREFIX + bare, bare):
            key_map.setdefault(alias, key)

    to_diffusers = getattr(comfy.utils, "krea2_to_diffusers", None)
    if to_diffusers is not None:
        # `unet_config` only has to carry the layer counts the map is built from; the real
        # config is not on disk next to the weights.
        cfg = {"layers": sum(1 for k in keys if k.endswith("blocks.0.attn.wq.weight")) and 28}
        try:
            for lora_key, model_key in to_diffusers(cfg, output_prefix=prefix).items():
                if model_key in keys and lora_key.endswith(".weight"):
                    stem = lora_key[: -len(".weight")]
                    key_map.setdefault(_PREFIX + stem, model_key)
                    key_map.setdefault(_ALT_PREFIX + stem, model_key)
                    key_map.setdefault(stem, model_key)
        except Exception as exc:                      # a config mismatch is not fatal here
            print("note: diffusers key map unavailable ({})".format(exc), flush=True)
    return key_map


def load_patches(source_keys, prefix, loras):
    """`{checkpoint key: [(adapter, strength), ...]}` for every LoRA on the command line."""
    key_map = build_key_map(source_keys, prefix)
    patches: dict[str, list] = {}
    for path, strength in loras:
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        loaded = comfy.lora.load_lora(sd, key_map, log_missing=False)
        if not loaded:
            raise SystemExit("no layer of {} matched this checkpoint".format(path))
        for key, patch in loaded.items():
            patches.setdefault(key, []).append((patch, strength))
        print("  {}: {} layers @ {:.2f}".format(os.path.basename(path), len(loaded), strength),
              flush=True)
    return patches


def make_weight_patch(patches: dict, applied: dict):
    """The `convert()` hook: weight -> weight + sum of the LoRA deltas for that key."""
    def weight_patch(key: str, tensor: torch.Tensor) -> torch.Tensor:
        entries = patches.get(key)
        if not entries:
            return tensor
        # fp32 for the accumulation: the deltas are small next to the weight, and this is a
        # one-off cost paid at build time rather than per step. `copy=True` because
        # `calculate_weight` adds in place and `.to()` is a no-op on a tensor that is already
        # fp32 -- without it this would mutate the caller's tensor.
        merged = tensor.to(torch.float32, copy=True)
        for adapter, strength in entries:
            # `offset` is None when the patch covers the whole weight; passing 0 makes
            # ComfyUI subscript it (`offset[0]`) and die -- and it dies on the *last* key,
            # after the quantizer has already burned a quarter of an hour.
            if isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
                merged = adapter.calculate_weight(
                    merged, key, strength, strength, None, lambda x: x,
                    intermediate_dtype=torch.float32)
            else:
                merged = comfy.lora.calculate_weight(
                    [(strength, adapter, strength, None, None)], merged, key,
                    intermediate_dtype=torch.float32)
        applied[key] = applied.get(key, 0) + len(entries)
        return merged.to(tensor.dtype)

    return weight_patch


def preflight(src: str, patches: dict) -> None:
    """Apply one patch of each kind for real, before the quantizer runs for a quarter hour.

    The quantizer processes the block weights first and the plain ones last, so a mistake in
    how a `diff` patch is applied surfaces only after every layer has been quantized. One
    tensor per distinct patch type costs a couple of seconds and moves that to the start.
    """
    from safetensors import safe_open

    seen = set()
    probes = []
    for key, entries in patches.items():
        kind = tuple(type(a).__name__ for a, _ in entries)
        if kind not in seen:
            seen.add(kind)
            probes.append(key)

    with safe_open(src, framework="pt", device="cpu") as handle:
        for key in probes:
            hook = make_weight_patch(patches, {})
            hook(key, handle.get_tensor(key))
    print("  preflight ok ({} patch shape(s))".format(len(probes)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="the high-precision checkpoint (raw.safetensors, turbo...)")
    ap.add_argument("--lora", action="append", required=True, metavar="PATH[:STRENGTH]",
                    help="repeat to bake several, in order; strength defaults to 1.0")
    ap.add_argument("--format", choices=["int8", "w4a4", "svdq", "fp8"], default="svdq")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--rank-alloc", default="uniform")
    ap.add_argument("--refine-iters", type=int, default=100)
    ap.add_argument("--variant", choices=["turbo", "base", "unknown"], default="unknown")
    ap.add_argument("--act-stats", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loras = []
    for spec in args.lora:
        path, _, strength = spec.rpartition(":")
        # A bare Windows path has a colon in it ("D:/loras/x.safetensors"), so only treat the
        # tail as a strength when it actually parses as one.
        try:
            loras.append((path, float(strength)) if path else (spec, 1.0))
        except ValueError:
            loras.append((spec, 1.0))

    fmt, rank = resolve_format(args.format, args.rank, rank_was_set=True)

    from safetensors import safe_open
    with safe_open(args.src, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    prefix = detect_prefix(keys, default=LAYER_PREFIXES[0])

    print("baking {} adapter(s) into {}".format(len(loras), os.path.basename(args.src)),
          flush=True)
    patches = load_patches(keys, prefix, loras)
    applied: dict[str, int] = {}
    preflight(args.src, patches)

    out = args.out
    if out is None:
        out, _ = derive_out_path(args.src, args.format, rank, args.variant, args.rank_alloc,
                                 args.act_stats)
        stem, ext = os.path.splitext(out)
        out = "{}-baked{}".format(stem, ext)

    convert(args.src, out, fmt, args.groupsize, args.device, rank, args.refine_iters,
            variant=args.variant, rank_alloc=args.rank_alloc, act_stats=args.act_stats,
            weight_patch=make_weight_patch(patches, applied))

    missing = set(patches) - set(applied)
    if missing:
        # Silence here would mean shipping a checkpoint that is quietly missing part of the
        # LoRA, which is exactly the failure this tool exists to avoid.
        raise SystemExit("{} matched layer(s) never reached the writer: {}".format(
            len(missing), ", ".join(sorted(missing)[:5])))
    print("baked {} layer(s) into {}".format(len(applied), out))


if __name__ == "__main__":
    main()
