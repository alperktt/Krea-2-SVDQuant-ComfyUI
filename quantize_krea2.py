"""Quantize a BF16 Krea2 checkpoint into a ComfyUI-native quantized checkpoint.

ComfyUI already ships native kernels for these formats via comfy_kitchen, so the output
loads with a plain ``UNETLoader`` -- no custom node, and ordinary LoRA loaders work.

Four targets:

* ``int8``  -> ``int8_tensorwise`` with per-channel scales + convrot (W8A8).
               Natively accelerated on Ampere INT8 tensor cores.
* ``w4a4``  -> ``convrot_w4a4`` (W4A4), smaller and potentially faster, 4-bit quality.
* ``svdq``  -> the same W4A4 residual plus an SVDQuant low-rank bf16 branch, which
               absorbs the outlier directions 4 bits handle worst. Needs this repo's
               loader node; a plain ``UNETLoader`` cannot see the branch.
* ``fp8``   -> ``float8_e4m3fn``, a single per-tensor scale, no convrot, no calibration.
               Lightest touch of the four -- best fidelity, smallest speedup, and the
               only one that needs no rotation or outlier handling because 8-bit float
               already has enough dynamic range for these weights.

None of them need a calibration dataset: int8/w4a4/svdq spread outliers analytically via
the convrot (group-wise Hadamard) rotation, and activations are quantized by the kernel at
run time; fp8 just needs one abs-max scale per tensor.

Only the 224 transformer-block linears are quantized, matching the layer set used by the
reference ``krea2_raw_int8_convrot`` checkpoint. Norms, modulation, the text-fusion stack
and the final layer stay in their original precision -- they are small and quantization
noise there is disproportionately visible.

The layer selection is variant-agnostic: Krea 2 Turbo and the base (non-distilled)
release share the same block naming, so the same command converts either. What differs is
how you sample afterwards, not how you quantize -- pass ``--variant`` so the output is
named accordingly and the file records which one it came from.

    python quantize_krea2.py <bf16-model.safetensors> [--format int8|w4a4|svdq|fp8]
                             [--rank 64] [--variant turbo|base] [--out PATH]
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
import zlib

__version__ = "0.2.0"

import torch
from safetensors import safe_open
from safetensors.torch import save_file

def _find_comfyui_root() -> str | None:
    """Locate the ComfyUI root so `comfy.*` imports work regardless of cwd.

    Preference order: $COMFYUI_PATH, then walking up from this file (this script
    lives in ComfyUI/custom_nodes/<pkg>/, so comfy/ is two levels up). None if
    neither resolves -- deliberately *not* cwd. This module is imported by the
    Quantize node inside ComfyUI's own process, and putting cwd at sys.path[0]
    there lets any file in the working directory shadow a real import.
    """
    env = os.environ.get("COMFYUI_PATH")
    if env and os.path.isdir(os.path.join(env, "comfy")):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(here, "..", ".."))
    if os.path.isdir(os.path.join(candidate, "comfy")):
        return candidate
    for extra in ("D:/ComfyUI", "C:/ComfyUI", os.path.abspath(os.path.join(here, "..", "ComfyUI"))):
        if os.path.isdir(os.path.join(extra, "comfy")):
            return extra
    return None


# Running inside ComfyUI, `comfy` already imports and the path needs no help.
if importlib.util.find_spec("comfy") is None:
    _root = _find_comfyui_root()
    if _root is not None:
        sys.path.insert(0, _root)

# Guarded because __init__.py imports this module transitively: an unguarded failure
# here would deregister every node in the pack, including Env Check, whose whole job is
# diagnosing a broken quantization backend. Degrade to "no formats available" instead.
QUANT_UNAVAILABLE = ""
try:
    from comfy.quant_ops import QUANT_ALGOS, get_layout_class  # noqa: E402
except Exception as exc:  # pragma: no cover - depends on the ComfyUI build
    QUANT_ALGOS = {}
    get_layout_class = None
    QUANT_UNAVAILABLE = "{}: {}".format(type(exc).__name__, exc)

# Relative improvement the refinement loop must keep making to earn another iteration. See
# `svdquant_split` for the traced numbers this default comes from.
REFINE_TOL = 0.001

# Deterministic by default. The low-rank split is a *randomized* SVD, so an unseeded build is
# irreproducible even on the same machine -- two files that differ by ~1e-4 per weight and by
# a visibly different image after 8 steps of sampling. `None` restores the old ambient-RNG
# behaviour for anyone who wants to sample the spread rather than pin it.
DEFAULT_SEED = 0

_QUANT_SUFFIXES = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo",
                   "mlp.gate", "mlp.up", "mlp.down")

# MiniMax H3: 50 blocks, four linears each, and Q/K/V arrive fused as one `qkv_proj` rather
# than split the way Krea 2 has them. `adaln_proj.linear` is deliberately absent -- on the
# pruned release it is [96768, 8], i.e. eight input channels against a group size of 256,
# and it is a lookup table rather than a projection.
_H3_QUANT_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")

# What differs between models that this quantizer can handle: the leaf names under
# `blocks.N.`, how many blocks there are (progress reporting only), and which rank
# allocations make sense. `gqa` moves budget onto `attn.wk`/`attn.wv`, which only exist as
# separate tensors on Krea 2 -- there is nothing to move it onto when Q/K/V are fused.
ARCHITECTURES: dict[str, dict] = {
    "krea2": {
        "suffixes": _QUANT_SUFFIXES,
        "blocks": 28,
        "allocs": ("uniform", "gqa"),
    },
    "minimax_h3": {
        "suffixes": _H3_QUANT_SUFFIXES,
        "blocks": 50,
        "allocs": ("uniform",),
    },
}

# The union, for callers that walk a live module tree rather than a checkpoint and can
# safely match a superset -- only layers that exist get matched. `svdquant_capture` does.
ALL_QUANT_SUFFIXES = tuple(sorted(
    {suf for arch in ARCHITECTURES.values() for suf in arch["suffixes"]}))


def detect_architecture(keys, prefix: str) -> str:
    """Which entry in `ARCHITECTURES` this checkpoint's block leaves look like.

    Scored by how many distinct leaves each candidate matches, so a model that happens to
    share one name with another does not win on that alone. Ties and zero-matches raise,
    with the observed leaves in the message -- the alternative is quantizing nothing and
    writing a full-size file that does nothing, which is the failure this whole family of
    checks exists to prevent.
    """
    observed = set()
    head = "{}blocks.".format(prefix)
    for key in keys:
        if not key.endswith(".weight") or not key.startswith(head):
            continue
        leaf = leaf_name(key[: -len(".weight")], prefix)
        if leaf is not None:
            observed.add(leaf)

    scores = {name: len(observed & set(spec["suffixes"]))
              for name, spec in ARCHITECTURES.items()}
    best = max(scores.values()) if scores else 0
    if best == 0:
        raise RuntimeError(
            "no known architecture matches this checkpoint. Nothing under '{}blocks.' ends "
            "in a leaf name any of {} expects.\n  observed leaves: {}".format(
                prefix, ", ".join(sorted(ARCHITECTURES)),
                ", ".join(sorted(observed)) or "none"))
    winners = [n for n, sc in scores.items() if sc == best]
    if len(winners) > 1:
        raise RuntimeError(
            "checkpoint matches {} equally well ({} leaves each); cannot tell them "
            "apart.\n  observed leaves: {}".format(
                " and ".join(sorted(winners)), best, ", ".join(sorted(observed))))
    return winners[0]


# Checkpoints ship either bare ("blocks.0...") or prefixed ("model.diffusion_model.blocks.0...").
LAYER_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "")

# Formats we know how to reconstruct back to BF16 from disk alone. FP8 storage is one
# byte per element with no packing or pre-rotation, so `qdata * scale` recovers the
# original tensor exactly as it was cast -- no architecture knowledge needed beyond
# what's already in the file. INT8/W4A4 sources pack multiple values per byte and are
# rotated (convrot) before quantization; unpacking those correctly needs the exact
# in/out feature counts from the live nn.Linear, which isn't recoverable from the
# checkpoint alone, so those are rejected instead.
_DEQUANTIZABLE_FORMATS = ("float8_e4m3fn", "float8_e5m2")


def detect_prefix(keys, default: str | None = None) -> str:
    """Return the prefix the transformer blocks live under.

    Raises `RuntimeError` if no prefix matches and no `default` was given. Not
    `SystemExit`: that is a `BaseException`, and `convert()` runs inside ComfyUI's
    executor via the Quantize node, where a BaseException escapes error handling
    entirely. `main()` catches RuntimeError and prints it as a clean CLI line.
    """
    for prefix in LAYER_PREFIXES:
        if any(k.startswith("{}blocks.".format(prefix)) for k in keys):
            return prefix
    if default is not None:
        return default
    raise RuntimeError(
        "Could not find transformer blocks in this checkpoint. Expected keys like\n"
        "  blocks.0.attn.wq.weight  or  model.diffusion_model.blocks.0.attn.wq.weight\n"
        "This does not look like a Krea 2 diffusion model."
    )


def check_requantizable(handle, keys, prefix: str, suffixes=None) -> None:
    """Make sure every target layer that's already quantized is something we can
    dequantize back to BF16 (see `_DEQUANTIZABLE_FORMATS`). Anything else -- most
    importantly INT8/W4A4 -- needs the original BF16 (or FP16) release instead.
    """
    for key in keys:
        if not key.endswith(".comfy_quant"):
            continue
        layer = key[: -len(".comfy_quant")]
        if not is_target(layer, prefix, suffixes):
            continue
        conf = json.loads(bytes(handle.get_tensor(key).tolist()))
        fmt = conf.get("format")
        if fmt not in _DEQUANTIZABLE_FORMATS:
            raise RuntimeError(
                "Layer {} is already quantized as '{}'. Only FP8-quantized layers can be "
                "automatically reconstructed and re-quantized; for INT8/W4A4 sources, use "
                "the original BF16 (or FP16) release of the model instead.".format(layer, fmt)
            )


def dequantize_target_weight(handle, layer: str, device: str) -> torch.Tensor:
    """Load a target layer's weight as BF16, dequantizing it first if it's FP8.

    Scaled FP8 (has a `comfy_quant` marker + `weight_scale`): ``qdata.float() * scale``.
    Unscaled FP8 (a bare ``.to(float8_e4m3fn)`` cast, no marker): ``qdata`` as-is.
    Anything else is already ruled out by `check_requantizable`.
    """
    weight = handle.get_tensor("{}.weight".format(layer)).to(device=device)
    conf_key = "{}.comfy_quant".format(layer)
    if conf_key in handle.keys():
        scale = handle.get_tensor("{}.weight_scale".format(layer)).to(device=device).float()
        return (weight.to(torch.float32) * scale).to(torch.bfloat16)
    return weight.to(torch.bfloat16)


def is_target(layer: str, prefix: str, suffixes=None) -> bool:
    """Only the transformer blocks; txtfusion and friends stay high precision.

    `suffixes` defaults to Krea 2's set so existing callers keep their behaviour; `convert`
    passes the detected architecture's.
    """
    if not layer.startswith("{}blocks.".format(prefix)):
        return False
    return any(layer.endswith(s) for s in (suffixes or _QUANT_SUFFIXES))


def leaf_name(layer: str, prefix: str) -> str | None:
    """``blocks.7.attn.wq`` -> ``attn.wq``. None if the layer is not inside a block.

    Only used to make a "nothing matched" failure self-diagnosing: a Krea 2 variant that
    keeps the ``blocks.`` container but renames its leaves would otherwise quantize zero
    layers and still write a full-size file.
    """
    head = "{}blocks.".format(prefix)
    if not layer.startswith(head):
        return None
    rest = layer[len(head):]
    return rest.split(".", 1)[1] if "." in rest else None


# Rank is bought in multiples of this, so a reallocated budget lands on the same grid the
# `rank` widget and the CLI already use.
_RANK_STEP = 8

# Per-leaf rank multipliers, relative to a uniform budget.
#
# Uniform rank is the obvious default and the wrong one. Measured on Krea2 Turbo at rank 64,
# error removed per million branch parameters spans 6.9x across the eight leaves:
#
#     attn.wk  0.0992   (1536x6144, branch 491K params, absorbs 30.1% of the error)
#     attn.wv  0.0844   (1536x6144, 491K, 25.6%)
#     attn.wq  0.0445   (6144x6144, 786K, 21.6%)
#     attn.wo  0.0411   (6144x6144, 786K, 17.6%)
#     attn.gate 0.0384  (6144x6144, 786K, 18.7%)
#     mlp.down 0.0169   (6144x16384, 1.44M, 14.3%)
#     mlp.gate 0.0155   (16384x6144, 1.44M, 13.8%)
#     mlp.up   0.0143   (16384x6144, 1.44M, 12.7%)
#
# The cause is GQA: Krea 2 has 12 kv heads against 48 query heads, so wk/wv are 1536-wide and
# their branch costs a third of an MLP branch -- while absorbing twice as much error. Uniform
# rank therefore spends most of the budget on the layers that reward it least.
#
# These multipliers come from a greedy allocation under a fixed byte budget (branch cost is
# exactly linear in rank, so scaling all of them preserves byte-neutrality at any base rank).
# At base rank 64 they reproduce the solved table: wk 360, wv 256, wq 72, wo 64, gate 56,
# mlp 8, and the checkpoint comes out 0.02% smaller than uniform rank 64.
#
# MEASURED, and it does not deliver. The greedy solve predicted 6% less mean layer weight
# error; the images say otherwise. 10 prompts, seed 987654321, against the same BF16 reference:
#
#     uniform rank 64  LPIPS 0.3403 +-0.1019
#     gqa     rank 64  LPIPS 0.3523 +-0.0473
#
# gqa is closer on 5 of 10 prompts, paired t = +0.55 on 9 df. No mean effect in either
# direction. Two things did not survive scrutiny and are recorded so nobody re-derives them:
# the -6% rests on an absorption ~ sqrt(rank) model fitted to nothing stronger than "returns
# are sublinear", and the tempting pattern that gqa wins on the hard prompts is an artifact --
# re-pairing the gqa scores at random reproduces the same -0.92 correlation, because the
# baseline appears on both sides of the difference.
#
# What is left is real but narrow: the spread across prompts shrinks (variance ratio 4.64,
# F-test p = 0.032) and the worst prompt improves, 0.4975 -> 0.4470. That is one nominally
# significant result among several things tried on the same 10 images, so it is a lead for a
# larger measurement, not a reason to change the default. `uniform` stays the default.
#
# The 6.9x spread itself is solid -- it is measured weight error over real shapes. The lesson
# is that weight error is a poor proxy for image outcome, which is now the third time this
# has held on this model (the refinement objective and the depth hypothesis were the others).
_GQA_ALLOC = {
    "attn.wk": 5.625, "attn.wv": 4.0, "attn.wq": 1.125, "attn.wo": 1.0, "attn.gate": 0.875,
    "mlp.gate": 0.125, "mlp.up": 0.125, "mlp.down": 0.125,
}

RANK_ALLOCATIONS: dict[str, dict[str, float] | None] = {
    "uniform": None,
    "gqa": _GQA_ALLOC,
}


def leaf_ranks(rank: int, rank_alloc: str = "uniform", suffixes=None) -> dict[str, int]:
    """Expand a rank budget into a per-leaf rank, one entry per leaf name.

    `uniform` gives every leaf the same rank -- the historical behaviour. Other allocations
    redistribute the same total branch bytes according to `RANK_ALLOCATIONS`.
    """
    suffixes = suffixes or _QUANT_SUFFIXES
    if rank_alloc not in RANK_ALLOCATIONS:
        raise RuntimeError("unknown rank allocation {!r}; expected one of {}".format(
            rank_alloc, ", ".join(sorted(RANK_ALLOCATIONS))))
    mults = RANK_ALLOCATIONS[rank_alloc]
    if mults is None:
        return {leaf: rank for leaf in suffixes}
    missing = [leaf for leaf in suffixes if leaf not in mults]
    if missing:
        raise RuntimeError(
            "rank allocation {!r} has no entry for {}; it was written for a different "
            "architecture.".format(rank_alloc, ", ".join(sorted(missing))))

    # Byte-neutrality is the whole point of a non-uniform allocation, and it only survives
    # while every leaf can be expressed on the step-8 grid. Below that the smallest leaves get
    # rounded *up* to the floor and the file grows: at budget 16 the MLP leaves want rank 2,
    # take 8, and the checkpoint comes out 20% larger than uniform -- which would make any
    # comparison against uniform meaningless. Refuse rather than quietly change the size.
    smallest = min(mults.values())
    minimum = int(math.ceil(_RANK_STEP / smallest / _RANK_STEP)) * _RANK_STEP
    if rank < minimum:
        raise RuntimeError(
            "rank allocation {!r} needs a budget of at least {} (you asked for {}). Its "
            "smallest leaf gets {}x the budget, which below that rounds up to the step-{} "
            "floor and makes the checkpoint *larger* than uniform instead of the same size."
            .format(rank_alloc, minimum, rank, smallest, _RANK_STEP))

    out = {}
    for leaf in suffixes:
        out[leaf] = int(round(rank * mults[leaf] / _RANK_STEP)) * _RANK_STEP
    return out


def rank_for_leaf(ranks: dict[str, int], leaf: str | None) -> int:
    """The branch rank for a target layer, given its leaf name. 0 means "no branch".

    Matches by suffix because `is_target` does. An exact-key lookup would disagree with it
    on any variant that nested the leaves one level deeper (`blocks.0.wrapper.attn.wq`):
    such a layer is a quantization target, so it would be quantized, but would find no rank
    entry and be written without a low-rank branch -- an svdq checkpoint silently missing
    branches on part of the network, which is the class of quiet wrongness the zero-layer
    check below exists to prevent.
    """
    if not ranks or leaf is None:
        return 0
    if leaf in ranks:
        return ranks[leaf]
    for suffix, value in ranks.items():
        if leaf.endswith(suffix):
            return value
    return 0


def svd_lowrank(weight: torch.Tensor, rank: int, oversample: int = 16, niter: int = 2):
    """Randomized truncated SVD: return (L1 [out, rank], L2 [rank, in]) with W ~ L1 @ L2.

    Randomized rather than exact because these are 6144x16384 matrices and only the top
    few dozen singular directions matter here; `oversample` extra probe columns plus a
    couple of power iterations recover them to well past the precision 4-bit quantization
    cares about.

    "Randomized" is load-bearing for reproducibility: `torch.svd_lowrank` draws its probe
    matrix from the *ambient* RNG, so seeding is the caller's job (`svdquant_split` does it
    per layer). Left unseeded, two runs of this script on the same GPU with identical
    arguments produce different checkpoints -- close, but not equal, and visibly so once a
    sampler amplifies them.
    """
    w = weight.float()
    min_dim = min(w.shape)
    rank = max(1, min(int(rank), min_dim))
    q = min(rank + max(0, int(oversample)), min_dim)

    if min_dim <= max(q, 32):
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u_r, s_r, vh_r = u[:, :rank], s[:rank], vh[:rank, :]
    else:
        u, s, v = torch.svd_lowrank(w, q=q, niter=max(0, int(niter)))
        u_r, s_r, vh_r = u[:, :rank], s[:rank], v[:, :rank].transpose(-2, -1)

    return (u_r * s_r.unsqueeze(0)).contiguous(), vh_r.contiguous()


def _act_weighting(act_rms: torch.Tensor | None, in_features: int, device, floor: float = 0.05):
    """Turn a captured activation RMS vector into the column weighting `d`, or None.

    Normalised to mean 1 so the weighting only redistributes emphasis and never rescales the
    problem -- otherwise the error metric would no longer be comparable across layers and the
    refinement loop's stopping threshold would mean something different in each one.

    The floor matters more than it looks. `l2` is recovered by dividing by `d`, so a channel
    whose activation is ~0 would divide a finite number by ~0 and put an enormous row into the
    branch -- a branch that then dominates the output for a channel the model never actually
    drives. Clamping to 5% of the mean bounds that amplification at 20x.
    """
    if act_rms is None:
        return None
    d = act_rms.to(device=device, dtype=torch.float32).flatten()
    if d.numel() != in_features:
        raise RuntimeError("activation stats have {} channels but the weight has {}"
                           .format(d.numel(), in_features))
    mean = d.mean()
    if not torch.isfinite(mean) or mean <= 0:
        return None
    d = d / mean
    return d.clamp(min=floor)


def svdquant_split(weight: torch.Tensor, rank: int, fmt: str, groupsize: int,
                   refine_iters: int = 100, act_rms: torch.Tensor | None = None,
                   refine_tol: float = REFINE_TOL, seed: int | None = None):
    """SVDQuant ordering: pull a low-rank bf16 branch out of W, quantize the residual.

    The low-rank branch absorbs the outlier-heavy directions, so the part that has to
    survive 4 bits is better conditioned.

    A single SVD of W is only the first guess: it picks the directions that are largest
    in W, which is not the same as the directions the quantizer handles worst. So this
    iterates the way deepcompressor does -- re-fit the branch to whatever the quantizer
    is *currently* getting wrong, requantize, repeat -- and keeps the best result. Pass
    ``refine_iters=0`` for the plain single-shot split.

    Iteration one is exactly that single-shot split and the best is kept, so refining
    can only match or beat it. Measured on Krea2 Turbo, rank 64: ~10% less
    reconstruction error, every layer improving.

    `refine_tol` is what stops it, and it is *relative* for a reason. It used to be an
    absolute `1e-6` against an error of order 1e-1, so the loop only stopped when an
    iteration failed to improve by one part in a hundred thousand -- which measured at a mean
    of **79.8 of the 100 iterations** across a sample of twelve layers at rank 256. What those
    iterations buy, traced per iteration on the same layers:

        stop at              iterations   reconstruction error
        1e-6 absolute (old)     ~80        baseline
        0.5% relative            ~9        +4.3%
        0.1% relative (now)     ~22        +2.0%

    Conversion goes from ~14 minutes to ~4. The 2% is reconstruction error, which this repo
    has three times measured to be a poor predictor of image outcome (see `--rank-alloc` in
    the README), so read it as a conversion-time change rather than a quality one.

    The objective is weight reconstruction error, which needs no calibration data.
    That is the true output error under the assumption that the input covariance is
    identity -- and spreading outliers with the convrot rotation is what makes that
    assumption a reasonable one. Closing the remaining gap to deepcompressor means
    measuring the real covariance, which is what makes their conversion take hours.

    `seed` makes the run reproducible. The refinement draws a fresh random probe matrix on
    every iteration through `svd_lowrank`, from the ambient RNG, so without this two builds
    of the same checkpoint with the same arguments are merely *similar*: measured at ~1e-4
    relative on the dequantized weights, which is small until a sampler spends 8-50 steps
    compounding it and you get a visibly different image. Seeding once here rather than per
    iteration is enough -- the whole sequence of draws then follows from one number. Pass
    `None` for the old ambient-RNG behaviour.

    Reproducibility is per device, not across devices: CPU and CUDA generate different
    numbers from the same seed, and their GEMM reduction orders differ too, so a CPU build
    and a GPU build of the same checkpoint will never be equal. `convert` records the device
    in the metadata for that reason.

    Returns None for a degenerate weight (all-zero, or one that makes the error metric
    non-finite); the caller quantizes those without a branch rather than aborting.
    """
    if seed is not None:
        # fork_rng so a caller's RNG state survives: this runs inside somebody else's
        # process (ComfyUI), where silently reseeding the global generator would make every
        # later sampler node in the same session reproduce a different image.
        with torch.random.fork_rng(devices=_rng_devices(weight.device)):
            torch.manual_seed(seed)
            return _svdquant_split(weight, rank, fmt, groupsize, refine_iters, act_rms,
                                   refine_tol)
    return _svdquant_split(weight, rank, fmt, groupsize, refine_iters, act_rms, refine_tol)


def _layer_seed(seed: int | None, layer: str) -> int | None:
    """A per-layer seed derived from the run's seed and the layer name.

    Per layer rather than one seed for the whole run so that the result does not depend on
    the order layers happen to be visited in, nor on how many refinement iterations the
    previous layer took before its tolerance stopped it. Either would make a rebuild with a
    different `--refine-tol` diverge everywhere instead of only where it should.

    `zlib.crc32` rather than `hash()`: Python randomises string hashing per process, so
    `hash()` here would be reproducible within a run and useless across runs -- the exact
    failure this is meant to fix, hidden one level deeper.
    """
    if seed is None:
        return None
    return (seed + zlib.crc32(layer.encode("utf-8"))) % (2 ** 63)


def _rng_devices(device: torch.device) -> list:
    """Which generators `fork_rng` has to save and restore for work on `device`."""
    return [device] if device.type == "cuda" else []


def _svdquant_split(weight: torch.Tensor, rank: int, fmt: str, groupsize: int,
                    refine_iters: int, act_rms, refine_tol: float):
    w = weight.float()
    d = _act_weighting(act_rms, w.shape[1], w.device)
    # With a weighting, both the SVD and the error metric work in the scaled space: minimising
    # ||(W - W_hat) diag(d)|| approximates the error the layer's real inputs would produce,
    # which is what actually reaches the image. Without one this is the identity and the whole
    # function reduces to exactly the old behaviour.
    w_norm = torch.linalg.matrix_norm(w if d is None else w * d).item()
    if not math.isfinite(w_norm) or w_norm == 0.0:
        return None
    qw = torch.zeros((), device=w.device, dtype=torch.float32)

    best = None
    best_err = float("inf")
    for _ in range(max(1, refine_iters)):
        target = w - qw
        if d is None:
            l1, l2 = svd_lowrank(target, rank, oversample=16, niter=2)
        else:
            # Fit in the scaled space so the retained directions are the ones that matter
            # once real activations hit them, then undo the scale on `l2` so `l1 @ l2` is
            # still an approximation of the *unscaled* residual.
            l1, l2s = svd_lowrank(target * d, rank, oversample=16, niter=2)
            l2 = l2s / d
            del l2s
        l1 = l1.to(torch.bfloat16)
        l2 = l2.to(torch.bfloat16)
        lw = l1.float() @ l2.float()
        # The quantizer must see the unscaled residual: the scale is an analysis device, and
        # baking it into what gets written would change what the kernel reconstructs.
        residual = (w - lw).to(torch.bfloat16)

        qdata, params, layout, _ = _quantize_raw(residual, fmt, groupsize)
        qw = layout.dequantize(qdata, params).float()
        del qdata, params

        left = w - (lw + qw)
        err = (torch.linalg.matrix_norm(left if d is None else left * d) / w_norm).item()
        del lw, left
        # NaN fails every comparison, so without this the loop would neither break nor
        # ever update `best` -- it would burn all `refine_iters` and return the last
        # split rather than the best one.
        if not math.isfinite(err):
            break
        if best is not None and err >= best_err * (1.0 - refine_tol):
            break  # refinement has stopped paying off
        best_err, best = err, (residual, l1, l2)

    return best


def _quantize_raw(weight: torch.Tensor, fmt: str, groupsize: int):
    """Quantize to `fmt`, handing back the layout and params too.

    Split out from `quantize_weight` so the refinement loop can round-trip a weight
    (quantize then dequantize) to see what the quantizer is actually getting wrong.
    """
    layout = get_layout_class(QUANT_ALGOS[fmt]["comfy_tensor_layout"])
    if fmt == "int8_tensorwise":
        qdata, params = layout.quantize(
            weight, is_weight=True, per_channel=True,
            convrot=True, convrot_groupsize=groupsize,
        )
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}
    elif fmt == "convrot_w4a4":
        qdata, params = layout.quantize(weight, convrot_groupsize=groupsize)
        conf = {"format": "convrot_w4a4", "convrot_groupsize": groupsize,
                "linear_dtype": getattr(params, "linear_dtype", "int4")}
    elif fmt == "float8_e4m3fn":
        qdata, params = layout.quantize(weight, scale="recalculate")
        conf = {"format": "float8_e4m3fn"}
    else:
        raise ValueError(fmt)
    return qdata, params, layout, conf


def quantize_weight(weight: torch.Tensor, fmt: str, groupsize: int):
    qdata, params, _, conf = _quantize_raw(weight, fmt, groupsize)
    return qdata, {"weight_scale": params.scale}, conf


def conf_tensor(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def load_act_stats(path: str) -> dict[str, torch.Tensor]:
    """Read the file the capture nodes write: {layer name: per-input-channel RMS}."""
    stats = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            stats[key] = handle.get_tensor(key)
    if not stats:
        raise RuntimeError("{} contains no activation statistics".format(path))
    return stats


def check_act_stats_coverage(stats: dict, keys, prefix: str, ranks: dict, suffixes=None) -> None:
    """Fail before any GPU work if the statistics don't cover every layer that will branch.

    Partial statistics are the quiet-wrongness case: the covered layers would get an
    activation-aware split and the rest a plain one, producing a checkpoint that is neither,
    and no measurement of it would mean anything.

    Deliberately key-only. Which layers branch depends on `is_target` plus a positive
    `rank_for_leaf`, both of which read the layer *name*; the one weight-dependent step in
    the loop (`min(wanted, min(w.shape))`) can only lower a rank, never raise a zero above
    it, so this pre-pass sees exactly the set the loop will branch. Doing it here instead of
    after the loop turns a ~6 minute mistake into a two-second one.
    """
    covered, missing = 0, []
    for key in keys:
        if not key.endswith(".weight"):
            continue
        layer = key[: -len(".weight")]
        if not is_target(layer, prefix, suffixes):
            continue
        if rank_for_leaf(ranks, leaf_name(layer, prefix)) <= 0:
            continue
        if stats.get(layer[len(prefix):] if prefix else layer) is None:
            missing.append(layer)
        else:
            covered += 1
    if missing:
        raise RuntimeError(
            "activation stats cover {} of {} branched layers; {} are missing, e.g. {}.\n"
            "The capture run did not see every layer -- usually because sampling was "
            "interrupted, or the statistics came from a different model. Re-capture rather "
            "than building a checkpoint that is activation-aware in some layers and not "
            "others.".format(covered, covered + len(missing), len(missing),
                             ", ".join(missing[:3])))


def convert(src: str, dst: str, fmt: str, groupsize: int, device: str = "cuda", rank: int = 0,
            refine_iters: int = 100, variant: str = "unknown", progress_cb=None,
            rank_alloc: str = "uniform", act_stats: str | None = None,
            weight_patch=None, refine_tol: float = REFINE_TOL,
            seed: int | None = DEFAULT_SEED):
    """Quantize `src` into `dst`. Returns the summary line it printed.

    `rank` is a budget, `rank_alloc` decides how it is spread across the eight projection
    types -- see `RANK_ALLOCATIONS`. Non-uniform allocations keep the same total branch bytes.

    `weight_patch(key, tensor) -> tensor` runs on every tensor as it is read, before anything
    is quantized. `tools/bake_adapter.py` uses it to add a LoRA delta into the high-precision
    weight, so the low-rank branch is fitted against the *merged* weight rather than the LoRA
    being requantized on top of a finished checkpoint. Streaming it here rather than writing a
    merged source out first matters in practice: the bf16 source is 24 GB.

    `progress_cb(done, total, message)` is called as layers complete, for callers with a
    progress bar to drive (the ComfyUI node). Errors are `RuntimeError`, never `SystemExit`:
    `SystemExit` derives from `BaseException`, so raising it from inside a node would slip
    past ComfyUI's executor instead of being reported as a failed node. `main()` translates
    them back for the CLI.
    """
    out: dict[str, torch.Tensor] = {}
    quantized = 0
    kept = 0
    branched = 0
    observed_leaves: set[str] = set()
    used_ranks: set[int] = set()
    clamped: dict[str, int] = {}
    stats = load_act_stats(act_stats) if act_stats else {}
    t0 = time.time()

    with safe_open(src, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefix = detect_prefix(keys)
        # Which model this is decides the leaf names, and therefore everything downstream:
        # what counts as a target, how a rank budget is spread, and how many layers the
        # progress bar is counting towards. Detected rather than passed, because the
        # checkpoint already knows and an argument would be one more thing to get wrong.
        arch = detect_architecture(keys, prefix)
        suffixes = ARCHITECTURES[arch]["suffixes"]
        expected_layers = ARCHITECTURES[arch]["blocks"] * len(suffixes)
        allowed = ARCHITECTURES[arch]["allocs"]
        if rank_alloc not in allowed:
            raise RuntimeError(
                "rank allocation {!r} does not apply to {}: it is defined over {}'s leaf "
                "names. Available here: {}.".format(
                    rank_alloc, arch,
                    "krea2" if rank_alloc in ARCHITECTURES["krea2"]["allocs"] else "another model",
                    ", ".join(allowed)))
        ranks = leaf_ranks(rank, rank_alloc, suffixes) if rank > 0 else {}
        print("architecture: {} ({} leaf names, {} blocks expected)".format(
            arch, len(suffixes), ARCHITECTURES[arch]["blocks"]))
        check_requantizable(handle, keys, prefix, suffixes)
        if stats:
            check_act_stats_coverage(stats, keys, prefix, ranks, suffixes)

        # Companion keys (old scales/markers) for target layers get regenerated fresh
        # when we process the ".weight" key below -- skip the stale copies on disk,
        # otherwise they'd overwrite our new ones later in iteration order.
        stale_companions = set()
        for key in keys:
            if key.endswith(".weight"):
                layer = key[: -len(".weight")]
                if is_target(layer, prefix, suffixes):
                    for suffix in ("weight_scale", "weight_scale_2", "input_scale", "comfy_quant"):
                        stale_companions.add("{}.{}".format(layer, suffix))

        for i, key in enumerate(keys):
            if key.endswith(".weight"):
                layer = key[: -len(".weight")]
                leaf = leaf_name(layer, prefix)
                if leaf is not None:
                    observed_leaves.add(leaf)
                if is_target(layer, prefix, suffixes):
                    w = dequantize_target_weight(handle, layer, device)
                    if weight_patch is not None:
                        w = weight_patch(key, w)
                    # A reallocated budget can ask for more rank than the weight has singular
                    # directions (attn.wk is only 1536 wide). `svd_lowrank` clamps internally,
                    # but clamping here too keeps the reported rank honest.
                    wanted = rank_for_leaf(ranks, leaf)
                    leaf_rank = min(wanted, min(w.shape))
                    if leaf and leaf_rank < wanted:
                        clamped[leaf] = leaf_rank
                    if leaf_rank > 0:
                        # Stats are keyed by module path (`blocks.0.attn.wq`); checkpoint
                        # keys may carry a prefix, so strip it to look them up.
                        # Coverage was already proven complete before the loop started, so
                        # a None here cannot happen for a branched layer.
                        act = stats.get(layer[len(prefix):] if prefix else layer)
                        split = svdquant_split(w, leaf_rank, fmt, groupsize, refine_iters,
                                               act_rms=act, refine_tol=refine_tol,
                                               seed=_layer_seed(seed, layer))
                        if split is None:
                            print("  warning: {} is degenerate (zero or non-finite); "
                                  "quantizing it without a low-rank branch".format(layer),
                                  flush=True)
                        else:
                            w, l1, l2 = split
                            out["{}.svdq_l1".format(layer)] = l1.cpu()
                            out["{}.svdq_l2".format(layer)] = l2.cpu()
                            branched += 1
                            used_ranks.add(int(l1.shape[1]))
                            del l1, l2
                    qdata, scales, conf = quantize_weight(w, fmt, groupsize)
                    out[key] = qdata.cpu()
                    for name, value in scales.items():
                        out["{}.{}".format(layer, name)] = value.cpu()
                    out["{}.comfy_quant".format(layer)] = conf_tensor(conf)
                    del w, qdata, scales
                    quantized += 1
                    if progress_cb is not None:
                        progress_cb(quantized, expected_layers,
                                    "quantized {} layers ({:.0f}s)".format(
                                        quantized, time.time() - t0))
                    if quantized % 32 == 0:
                        torch.cuda.empty_cache()
                        print(f"  [{i + 1}/{len(keys)}] quantized {quantized} layers "
                              f"({time.time() - t0:.0f}s)", flush=True)
                    continue
            if key in stale_companions:
                continue
            value = handle.get_tensor(key)
            # The non-quantized layers (txtfusion, embeddings) can carry a LoRA too, and they
            # are the half a bake would otherwise silently drop.
            out[key] = value if weight_patch is None else weight_patch(key, value)
            kept += 1

    gc.collect()
    torch.cuda.empty_cache()

    if quantized == 0:
        raise RuntimeError(
            "Found the transformer blocks but quantized nothing: no layer under "
            "'{0}blocks.' ends in one of the expected leaf names.\n"
            "  expected: {1}\n"
            "  observed: {2}\n"
            "This is a {3} variant with different layer naming; quantize_krea2.py "
            "would otherwise have written a full-size file that does nothing."
            .format(prefix, ", ".join(sorted(suffixes)),
                    ", ".join(sorted(observed_leaves)) or "none", arch)
        )

    # A leaf can only be clamped when the allocation asked for more rank than the weight is
    # wide, which means part of the budget went unspent and the file is *smaller* than the
    # uniform one -- so a comparison against uniform at this budget is no longer like-for-like.
    if clamped:
        print("warning: rank allocation {!r} asked for more rank than these layers are wide, "
              "so they were capped and part of the budget went unspent: {}. The checkpoint is "
              "smaller than uniform rank {}, not the same size -- lower the budget to compare "
              "them fairly.".format(
                  rank_alloc, ", ".join("{}={}".format(k, v) for k, v in sorted(clamped.items())),
                  rank), flush=True)

    created = len(out) - kept - quantized
    factors = "" if not rank else " + {} low-rank factors".format(branched * 2)
    print(f"quantized {quantized} layers; {kept} tensors passed through; "
          f"{created} tensors created (qdata/scale/marker{factors}); "
          f"{len(out)} tensors total; writing {dst} ...", flush=True)

    # Recorded so the loader (and anyone inspecting the file) can tell what produced it.
    # safetensors metadata is str -> str only. The loader treats all of this as optional:
    # checkpoints published before this existed still load, with the rank recovered from
    # the svdq_l1 shape as before.
    # The loader trusts the factor shapes over `krea2_svdquant_rank`, and under a non-uniform
    # allocation no single number is the rank -- so record the budget, the allocation name and
    # the full per-leaf map, and leave `rank` meaning "budget" for older loaders reading it.
    metadata = {
        "krea2_svdquant_tool_version": __version__,
        "krea2_svdquant_format": fmt,
        "krea2_svdquant_rank": str(rank),
        "krea2_svdquant_rank_alloc": rank_alloc,
        "krea2_svdquant_rank_map": json.dumps(ranks, sort_keys=True) if ranks else "",
        "krea2_svdquant_groupsize": str(groupsize),
        "krea2_svdquant_refine_iters": str(refine_iters if rank else 0),
        "krea2_svdquant_variant": variant,
        "krea2_svdquant_quantized_layers": str(quantized),
        "krea2_svdquant_branched_layers": str(branched),
        "krea2_svdquant_source_name": os.path.basename(src),
        "krea2_svdquant_source_bytes": str(os.path.getsize(src)),
        # Two checkpoints with identical rank/refine settings are still different models if
        # one was activation-weighted, so this has to be recoverable from the file.
        "krea2_svdquant_act_stats": os.path.basename(act_stats) if act_stats else "",
        # A build is reproducible only against the same seed AND the same device: the low-rank
        # split draws a random probe matrix, and CPU and CUDA neither generate the same numbers
        # from a seed nor reduce their GEMMs in the same order. Both go in the file so "why is
        # my checkpoint not byte-identical to yours" is answerable from the file.
        "krea2_svdquant_seed": "" if seed is None else str(seed),
        "krea2_svdquant_device": str(device),
    }
    if progress_cb is not None:
        progress_cb(quantized, expected_layers, "writing {:.2f} GB ...".format(
            sum(t.numel() * t.element_size() for t in out.values()) / 1024 ** 3))
    save_file(out, dst, metadata=metadata)
    size = os.path.getsize(dst) / 1024 ** 3
    rank_desc = ""
    if used_ranks:
        rank_desc = (" rank {}".format(next(iter(used_ranks))) if len(used_ranks) == 1
                     else " rank {}-{} ({})".format(min(used_ranks), max(used_ranks),
                                                    rank_alloc))
    summary = ("quantized {} layers ({} branched{}) in {:.0f}s -> {}  ({:.2f} GB, source "
               "{:.2f} GB)".format(quantized, branched, rank_desc, time.time() - t0, dst, size,
                                   os.path.getsize(src) / 1024 ** 3))
    print("done in {:.0f}s -> {}  ({:.2f} GB, source {:.2f} GB)".format(
        time.time() - t0, dst, size, os.path.getsize(src) / 1024 ** 3))
    return summary


# The Krea 2 block count, used only to scale a progress bar. A variant with a different
# depth still quantizes correctly; the bar is just less accurate.
# Layer counts are per-architecture now (`ARCHITECTURES[...]["blocks"]`); `convert` derives
# `expected_layers` from the detected one. The old module-level constant hardcoded 28 blocks
# of Krea 2 and would have quietly mis-scaled the progress bar on any other model.

_FORMAT_ALIASES = {
    "int8": "int8_tensorwise",
    "w4a4": "convrot_w4a4",
    "svdq": "convrot_w4a4",
    "fp8": "float8_e4m3fn",
}

# Which format names carry a low-rank branch. A tuple rather than a comparison against
# "svdq" scattered through `resolve_format`, so that adding a second branched base is one
# edit here instead of four that have to agree.
_BRANCHED_FORMATS = ("svdq",)

_GENERIC_STEMS = ("raw", "model", "diffusion_pytorch_model", "turbo")

SAMPLER_HINTS = {
    "base": ("Base (non-distilled) model: start from ~50 steps, cfg 3.5, euler/simple, and a "
             "real negative prompt. See workflows/krea2_base_svdquant_w4a4_t2i.json."),
    "turbo": ("Turbo (distilled) model: 8 steps, cfg 1.0, euler/simple. "
              "See workflows/krea2_turbo_svdquant_w4a4_t2i.json."),
}


def resolve_format(fmt_name: str, rank: int, rank_was_set: bool = True) -> tuple[str, int]:
    """Map a CLI/node format name to a comfy_kitchen format, validating the rank with it.

    Shared by `main()` and the ComfyUI node so the two cannot drift apart on which
    combinations are legal.
    """
    fmt = _FORMAT_ALIASES[fmt_name]
    if fmt not in QUANT_ALGOS:
        raise RuntimeError("{} is not available in this ComfyUI build".format(fmt))
    # Silently zeroing the rank here used to make `--format w4a4 --rank 128` look like it had
    # done something it had not.
    if fmt_name not in _BRANCHED_FORMATS and rank_was_set:
        raise RuntimeError(
            "rank only applies to {} (you asked for '{}'). The low-rank branch is what "
            "distinguishes svdq from plain w4a4.".format(
                " and ".join(repr(f) for f in _BRANCHED_FORMATS), fmt_name))
    if fmt_name in _BRANCHED_FORMATS and rank <= 0:
        plain = {"svdq": "w4a4"}[fmt_name]
        raise RuntimeError("format {!r} needs rank > 0; use format {!r} for no branch"
                           .format(fmt_name, plain))
    return fmt, (rank if fmt_name in _BRANCHED_FORMATS else 0)


def derive_out_path(src: str, fmt_name: str, rank: int, variant: str,
                    rank_alloc: str = "uniform", act_stats: str | None = None
                    ) -> tuple[str, str | None]:
    """The default output path, plus a note to show when the source name is uninformative."""
    stem = os.path.splitext(os.path.basename(src))[0]
    note = None
    if variant != "unknown":
        stem = "Krea2-{}".format(variant.capitalize())
    elif stem.lower() in _GENERIC_STEMS:
        note = ("note: '{}' is a generic filename. Set the variant (or an explicit output "
                "name) to get a checkpoint name you will still recognise next month."
                .format(stem))
    # The allocation goes in the filename because two checkpoints with the same budget and
    # different allocations are the same size -- nothing else would tell them apart.
    alloc_tag = "" if rank_alloc == "uniform" else "-{}".format(rank_alloc)
    # Same reasoning as alloc_tag: an activation-weighted build is byte-identical in shape to
    # a plain one, so without a tag the two are indistinguishable on disk.
    act_tag = "-actaware" if act_stats else ""
    suffix = ("SVDQuant-W4A4-rank{}{}{}".format(rank, alloc_tag, act_tag) if rank else
              ("{}-convrot".format(fmt_name.upper()) if fmt_name != "fp8" else "FP8"))
    return os.path.join(os.path.dirname(src), "{}-{}.safetensors".format(stem, suffix)), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--format", choices=["int8", "w4a4", "svdq", "fp8"], default="int8",
                    help="svdq = w4a4 residual + SVDQuant low-rank bf16 branch; "
                         "fp8 = float8_e4m3fn, no convrot, no low-rank branch")
    ap.add_argument("--groupsize", type=int, default=256, help="unused for fp8")
    ap.add_argument("--rank", type=int, default=64, help="low-rank branch budget, svdq only")
    ap.add_argument("--rank-alloc", choices=sorted(RANK_ALLOCATIONS), default="uniform",
                    help="how to spread the rank budget across the eight projection types. "
                         "uniform = same rank everywhere. gqa = byte-neutral reallocation "
                         "towards the GQA kv projections, which absorb ~2x the error at a "
                         "third of the branch cost (see RANK_ALLOCATIONS)")
    ap.add_argument("--refine-iters", type=int, default=100,
                    help="svdq only: refine the low-rank branch against the quantization "
                         "error, keeping the best (0 = plain single-shot SVD, much faster "
                         "but ~10%% more reconstruction error). This is a cap; --refine-tol "
                         "is what usually stops the loop first")
    ap.add_argument("--refine-tol", type=float, default=REFINE_TOL, metavar="FRACTION",
                    help="svdq only: stop refining a layer once an iteration improves its "
                         "reconstruction error by less than this fraction (default %(default)s "
                         "= 0.1%%). Lower means more iterations for less return: 0.001 takes "
                         "~22 iterations per layer, 0.005 takes ~9 for 2%% more error, and 0 "
                         "restores the old behaviour of running nearly all --refine-iters")
    ap.add_argument("--variant", choices=["turbo", "base", "unknown"], default="unknown",
                    help="which Krea 2 release this is. Only affects the output filename "
                         "and the recorded metadata -- quantization is identical for both, "
                         "the difference is the sampler settings you run afterwards")
    ap.add_argument("--act-stats", default=None, metavar="PATH",
                    help="svdq only: activation statistics from the Krea2 SVDQuant Capture "
                         "nodes. Weights the low-rank split by per-input-channel activation "
                         "RMS, so the branch spends its capacity on the directions the model "
                         "actually drives instead of on the largest weights. Costs nothing "
                         "at runtime -- same rank, format, size and kernel")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="seed for the randomized low-rank SVD, so a build is reproducible "
                         "(default %(default)s). Reproducible on the same device only -- a "
                         "CPU build and a CUDA build of the same checkpoint differ by ~1e-4 "
                         "per weight whatever the seed. Pass -1 for the old unseeded "
                         "behaviour, where even two runs on one GPU differ")
    args = ap.parse_args()

    if args.act_stats:
        if args.format != "svdq":
            raise SystemExit("--act-stats only applies to format 'svdq': it weights the "
                             "low-rank split, and the other formats have no branch")
        if not os.path.exists(args.act_stats):
            raise SystemExit("--act-stats file not found: {}".format(args.act_stats))

    # RuntimeError is the shared failure type (see `convert`); the CLI wants SystemExit so it
    # prints one clean line instead of a traceback.
    try:
        fmt, rank = resolve_format(args.format, args.rank,
                                   rank_was_set=args.rank != ap.get_default("rank"))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    if args.rank_alloc != "uniform" and not rank:
        raise SystemExit("--rank-alloc only applies to format 'svdq'")

    out = args.out
    if out is None:
        out, note = derive_out_path(args.src, args.format, rank, args.variant, args.rank_alloc,
                                    args.act_stats)
        if note:
            print(note, flush=True)
    try:
        convert(args.src, out, fmt, args.groupsize, args.device, rank, args.refine_iters,
                variant=args.variant, rank_alloc=args.rank_alloc, act_stats=args.act_stats,
                refine_tol=args.refine_tol,
                seed=None if args.seed < 0 else args.seed)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    hint = SAMPLER_HINTS.get(args.variant)
    if hint:
        print("\n" + hint)


if __name__ == "__main__":
    main()
