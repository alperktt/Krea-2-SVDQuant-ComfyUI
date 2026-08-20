"""Command-line front end for `build_all_in_one.py`.

The mechanism lives one directory up, beside `quantize_krea2.py`, because `.comfyignore`
keeps `tools/` out of the published package and the in-graph quantize node imports it.
This file is the part a terminal needs: the sys.path bootstrap that makes the package
importable when it is run as a plain script, and the argument parsing.

    python tools/build_all_in_one.py --dit ... --text-encoder ... --vae ... --format svdq

Run `--help` for the full set. `--dry-run` does the whole key mapping and prints the size
arithmetic without quantizing anything or touching the GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

# The package directory, so `import build_all_in_one` resolves to the module beside
# `quantize_krea2.py`. It in turn finds ComfyUI itself (`quantize_krea2._find_comfyui_root`),
# so there is nothing else to bootstrap here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_all_in_one import TE_FORMATS, build_all_in_one_checkpoint  # noqa: E402
from quantize_krea2 import DEFAULT_SEED, REFINE_TOL, resolve_format  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", required=True,
                    help="the diffusion model: a BF16 source, or one this tool already "
                         "quantized (detected, and passed through untouched)")
    ap.add_argument("--text-encoder", required=True, help="qwen3vl_4b BF16 safetensors")
    ap.add_argument("--vae", required=True)
    ap.add_argument("--format", choices=["int8", "w4a4", "svdq", "fp8"], default="svdq",
                    help="diffusion side. svdq needs this repo's checkpoint loader node; "
                         "w4a4/int8 load with the stock CheckpointLoaderSimple")
    ap.add_argument("--te-format", choices=sorted(TE_FORMATS), default="w4a4",
                    help="text encoder side. No low-rank branch either way")
    ap.add_argument("--rank", type=int, default=64, help="diffusion side, svdq only")
    ap.add_argument("--rank-alloc", default="uniform")
    ap.add_argument("--refine-iters", type=int, default=100)
    ap.add_argument("--refine-tol", type=float, default=REFINE_TOL)
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--act-stats", default=None, help="diffusion side only")
    ap.add_argument("--variant", choices=["turbo", "base", "unknown"], default="unknown")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-dit-temp", action="store_true",
                    help="do not delete the intermediate quantized diffusion file")
    ap.add_argument("--dry-run", action="store_true",
                    help="map every key and print the size arithmetic. Quantizes nothing, "
                         "allocates nothing, never touches the GPU")
    args = ap.parse_args()

    fmt, rank = resolve_format(args.format, args.rank, rank_was_set=True)

    out = args.out
    if out is None:
        stem = "Krea2-{}".format(args.variant.capitalize()) if args.variant != "unknown" \
            else "Krea2"
        tag = "SVDQuant-W4A4-rank{}".format(rank) if rank else args.format.upper()
        dit_dir = os.path.dirname(os.path.abspath(args.dit))
        ckpt_dir = os.path.join(os.path.dirname(dit_dir), "checkpoints")
        target_dir = ckpt_dir if os.path.isdir(ckpt_dir) else dit_dir
        out = os.path.join(target_dir,
                           "{}-AllInOne-{}-TE{}.safetensors".format(
                               stem, tag, args.te_format.upper()))

    print("diffusion : {}".format(os.path.basename(args.dit)))
    print("encoder   : {} -> {}".format(os.path.basename(args.text_encoder), args.te_format))
    print("vae       : {} (unquantized)".format(os.path.basename(args.vae)))
    print("out       : {}".format(out))

    summary = build_all_in_one_checkpoint(
        dit_path=args.dit,
        text_encoder_path=args.text_encoder,
        vae_path=args.vae,
        out_path=out,
        fmt_name=args.format,
        te_format=args.te_format,
        rank=rank,
        variant=args.variant,
        groupsize=args.groupsize,
        device=args.device,
        dry_run=args.dry_run,
    )
    print("\n" + summary)
    if not args.dry_run:
        if rank:
            print("svdq checkpoint: load it with the Krea2 SVDQuant Checkpoint Loader node, "
                  "not CheckpointLoaderSimple -- the low-rank branch needs attaching.")
        else:
            print("branchless: loads with the stock CheckpointLoaderSimple.")


if __name__ == "__main__":
    main()
