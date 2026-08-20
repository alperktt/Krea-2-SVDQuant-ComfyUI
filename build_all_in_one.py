"""Bake the diffusion model, the text encoder and the VAE into one checkpoint.

The mechanism. `tools/build_all_in_one.py` is the command-line front end for it and
`Krea2SVDQuantQuantizeAllInOne` (svdquant_quantize.py) is the in-graph one; this module
lives beside `quantize_krea2.py` rather than under `tools/` because `.comfyignore`
keeps `tools/` out of the published package and the node has to be able to import it.

    python tools/build_all_in_one.py \\
        --dit  D:/ComfyUI/models/diffusion_models/turbo.safetensors \\
        --text-encoder D:/ComfyUI/models/text_encoders/qwen3vl_4b_bf16.safetensors \\
        --vae  D:/ComfyUI/models/vae/Krea2-HD-vae.safetensors \\
        --format svdq --rank 64 --te-format w4a4 --variant turbo \\
        --act-stats calibration/krea2_act_stats_turbo.safetensors

Three downloads that have to match becomes one file. `--dry-run` does the whole key mapping
and prints the size arithmetic without quantizing anything or touching the GPU, which is the
right way to check the layout before spending ten minutes on a build.

WHAT COMFYUI EXPECTS, AND WHY THIS IS MOSTLY A RENAMING PROBLEM

`comfy/supported_models.py`'s `Krea2` already declares `vae_key_prefix = ["vae."]` and
`text_encoder_key_prefix = ["text_encoders."]`, and its `clip_target()` runs
`detect_layer_quantization` over the encoder prefix -- a format-agnostic scan for
`.comfy_quant` markers. So a quantized text encoder inside a combined checkpoint is a case
ComfyUI already handles; nothing upstream needs changing. The three maps:

    diffusion  blocks.N...                ->  model.diffusion_model.blocks.N...
    vae        conv1... / encoder...      ->  vae.conv1... / vae.encoder...
    encoder    model.language_model.X     ->  text_encoders.qwen3vl_4b.transformer.model.X
               model.visual.X             ->  text_encoders.qwen3vl_4b.transformer.visual.X
               lm_head.X                  ->  text_encoders.qwen3vl_4b.transformer.model.lm_head.X

The encoder map is not invented here: it is the same `state_dict_prefix_replace` call
`comfy/sd.py` makes in `load_text_encoder_state_dicts` when you load that file through
`CLIPLoader`. Standalone loading applies it at load time; a combined checkpoint has to carry
the keys already converted, because `load_state_dict_guess_config` does not repeat it.

WHICH FORMATS

The diffusion side takes any `--format` `quantize_krea2.py` supports. `svdq` gives the best
fidelity but the `svdq_l1`/`svdq_l2` buffers need this repo's loader to attach them, so an
svdq all-in-one needs the Krea2 SVDQuant Checkpoint Loader node rather than the stock
`CheckpointLoaderSimple`. Build with `--format w4a4` or `int8` for a file that loads with
stock ComfyUI and no custom node at all.

The text encoder takes `--te-format w4a4` or `int8` only, and never a low-rank branch. Not a
limitation of the quantizer -- `load_state_dict_guess_config` builds the CLIP through
ComfyUI's own path, which has nowhere to hang a branch. 4-bit is the only setting that
changes the number worth changing: INT8 is one byte per weight, the same as the FP8 encoder
already published.

WHAT THIS DOES NOT BUY

The text encoder and the diffusion model are never resident at the same time -- ComfyUI
encodes the prompt, unloads, then samples. This is a disk, download and don't-mismatch-your-
files win, plus a VRAM win during the encode pass. Sampling speed and sampling VRAM do not
move at all.
"""
from __future__ import annotations

import json
import os
import struct
import time

import torch
from safetensors import safe_open

# Relative when imported as part of the pack, absolute when `tools/build_all_in_one.py` runs
# this as a plain script. The relative arm is the one that matters: importing
# `quantize_krea2` absolutely from inside the package creates a *second* top-level copy of
# it, with its own module-level state, alongside the pack's own `.quantize_krea2`.
#
# Branching on `__package__` rather than try/except ImportError so that a real failure
# inside `quantize_krea2` -- a missing comfy, a broken quant backend -- surfaces as itself
# instead of being retried down the other path and reported as the wrong problem.
if __package__:
    from .quantize_krea2 import (
        DEFAULT_SEED,
        REFINE_TOL,
        __version__,
        conf_tensor,
        detect_prefix,
        quantize_weight,
        resolve_format,
    )
else:  # run outside the package by tools/build_all_in_one.py, which puts it on sys.path
    from quantize_krea2 import (
        DEFAULT_SEED,
        REFINE_TOL,
        __version__,
        conf_tensor,
        detect_prefix,
        quantize_weight,
        resolve_format,
    )

# `model.` here is the encoder's own `model.`, i.e. what `model.language_model.` becomes.
TE_PREFIX = "text_encoders.qwen3vl_4b.transformer."
DIT_PREFIX = "model.diffusion_model."
VAE_PREFIX = "vae."

# Exactly `comfy/sd.py`'s `load_text_encoder_state_dicts` mapping for a Qwen3-VL file. Order
# matters: `model.language_model.` has to be tried before any shorter `model.` rule would be.
TE_KEY_MAP = (
    ("model.language_model.", "model."),
    ("model.visual.", "visual."),
    ("lm_head.", "model.lm_head."),
)

# The 7 projections in a Qwen3 decoder layer. Deliberately not the DiT's `_QUANT_SUFFIXES`:
# different architecture, different names, and sharing one tuple between them would make a
# typo in either silently quantize nothing in the other.
TE_QUANT_SUFFIXES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
TE_LAYERS = 36
TE_EXPECTED = TE_LAYERS * len(TE_QUANT_SUFFIXES)

# Formats with no low-rank branch. The text encoder loads through ComfyUI's own CLIP path,
# which has nowhere to attach one.
TE_FORMATS = {"w4a4": "convrot_w4a4", "int8": "int8_tensorwise"}


def te_is_target(key: str) -> bool:
    """True for a decoder-layer projection weight, in *converted* key space.

    Converted, not raw: `model.visual.` becomes `visual.`, and the vision tower has layers
    and projections of its own. Matching after the rename is what keeps them out -- Krea 2's
    text encoder instantiates the vision tower (315 tensors, 0.83 GB) and krea2edit's image
    conditioning may go through it, so it is carried unquantized rather than dropped.
    """
    if not key.startswith("model.layers.") or not key.endswith(".weight"):
        return False
    return key[: -len(".weight")].endswith(TE_QUANT_SUFFIXES)


def remap_te_key(key: str) -> str:
    for old, new in TE_KEY_MAP:
        if key.startswith(old):
            return new + key[len(old):]
    return key


_DTYPE_STR = {
    torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.float64: "F64", torch.int64: "I64", torch.int32: "I32", torch.int16: "I16",
    torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2",
}


class StreamingWriter:
    """A safetensors writer that never holds the whole checkpoint in memory.

    `safetensors.torch.save_file` takes a dict, so writing a 12 GB combined checkpoint that
    way needs 12 GB of RAM on top of whatever the quantizer is using. The format does not
    require that: it is an 8-byte header length, a JSON header of name -> {dtype, shape,
    offsets}, then the raw buffers back to back.

    So this is two passes. `plan()` collects names, dtypes and shapes -- which are known
    without materialising anything -- and `write()` streams the tensors past in the same
    order. The cost is that every tensor is produced twice or held once; here the shapes come
    from safetensors headers and quantizer outputs we already have, so it is neither.
    """

    def __init__(self, path: str, metadata: dict | None = None):
        self.path = path
        self.metadata = metadata or {}
        self.entries = []          # (name, dtype_str, shape, nbytes)
        self._names = set()

    def plan(self, name: str, dtype, shape) -> None:
        if name in self._names:
            raise RuntimeError("duplicate key in the combined checkpoint: {}".format(name))
        self._names.add(name)
        dtype_str = _DTYPE_STR.get(dtype)
        if dtype_str is None:
            raise RuntimeError("no safetensors name for dtype {}".format(dtype))
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= torch.empty((), dtype=dtype).element_size()
        self.entries.append((name, dtype_str, list(shape), nbytes))

    def header(self) -> bytes:
        header, offset = {}, 0
        for name, dtype_str, shape, nbytes in self.entries:
            header[name] = {"dtype": dtype_str, "shape": shape,
                            "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        if self.metadata:
            header["__metadata__"] = {k: str(v) for k, v in self.metadata.items()}
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # The spec wants the buffer 8-byte aligned; pad the header with spaces, which JSON
        # ignores, rather than the buffer, whose offsets are already committed.
        pad = (-len(blob)) % 8
        return blob + b" " * pad

    def __enter__(self):
        self._blob = self.header()
        self._fh = open(self.path, "wb")
        self._fh.write(struct.pack("<Q", len(self._blob)))
        self._fh.write(self._blob)
        self._i = 0
        return self

    def write(self, name: str, tensor: torch.Tensor) -> None:
        expected = self.entries[self._i]
        if name != expected[0]:
            raise RuntimeError("write order does not match the plan: expected {!r}, got {!r}"
                               .format(expected[0], name))
        t = tensor.contiguous().cpu()
        buf = t.view(torch.uint8).reshape(-1) if t.dtype not in (torch.uint8,) else t.reshape(-1)
        data = buf.numpy().tobytes()
        if len(data) != expected[3]:
            raise RuntimeError("{}: planned {} bytes, got {}".format(name, expected[3], len(data)))
        self._fh.write(data)
        self._i += 1

    def __exit__(self, *exc):
        self._fh.close()
        if exc[0] is None and self._i != len(self.entries):
            raise RuntimeError("planned {} tensors but wrote {}".format(len(self.entries), self._i))


def plan_dit(handle, keys, prefix):
    """(combined key, source key) for every diffusion tensor, already prefixed."""
    for key in keys:
        bare = key[len(prefix):] if prefix and key.startswith(prefix) else key
        yield DIT_PREFIX + bare, key


def plan_vae(keys):
    for key in keys:
        yield VAE_PREFIX + key, key


def quantize_text_encoder(path: str, fmt: str, groupsize: int, device: str, writer,
                          dry_run: bool):
    """Plan (and on the second pass, write) the text encoder side.

    Returns (planned entries, quantized layer count). The two passes run the same key walk so
    the plan and the write cannot disagree about order -- which is the one way a streaming
    writer produces a file that is subtly, unreadably wrong.
    """
    algo = TE_FORMATS[fmt]
    quantized = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            converted = remap_te_key(key)
            out_key = TE_PREFIX + converted
            if not te_is_target(converted):
                if dry_run:
                    slice_ = handle.get_slice(key)
                    writer.plan(out_key, _slice_dtype(slice_), slice_.get_shape())
                else:
                    writer.write(out_key, handle.get_tensor(key))
                continue

            layer = out_key[: -len(".weight")]
            if dry_run:
                slice_ = handle.get_slice(key)
                shape = slice_.get_shape()
                # int4 packs two values per byte along the input dimension; int8 is 1:1.
                packed = [shape[0], shape[1] // 2] if fmt == "w4a4" else list(shape)
                writer.plan(out_key, torch.int8, packed)
                writer.plan(layer + ".weight_scale", torch.float32, [shape[0], 1])
                writer.plan(layer + ".comfy_quant", torch.uint8, [_conf_len(algo, groupsize)])
            else:
                w = handle.get_tensor(key).to(device=device, dtype=torch.bfloat16)
                qdata, scales, conf = quantize_weight(w, algo, groupsize)
                writer.write(out_key, qdata)
                for name, value in scales.items():
                    writer.write("{}.{}".format(layer, name), value)
                writer.write(layer + ".comfy_quant", conf_tensor(conf))
                del w, qdata, scales
                if quantized % 32 == 0:
                    torch.cuda.empty_cache()
            quantized += 1
    return quantized


def _slice_dtype(slice_):
    return {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
            "I8": torch.int8, "U8": torch.uint8, "I64": torch.int64,
            "F8_E4M3": torch.float8_e4m3fn}[slice_.get_dtype()]


def _conf_len(algo: str, groupsize: int) -> int:
    """Byte length of the comfy_quant marker, without quantizing a weight to find out."""
    if algo == "convrot_w4a4":
        conf = {"format": "convrot_w4a4", "convrot_groupsize": groupsize,
                "linear_dtype": "int4"}
    else:
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}
    return len(conf_tensor(conf))


def build_all_in_one_checkpoint(dit_path: str, text_encoder_path: str, vae_path: str,
                                out_path: str, fmt_name: str = "svdq", te_format: str = "w4a4",
                                rank: int = 64, variant: str = "unknown", groupsize: int = 256,
                                device: str = "cuda", dry_run: bool = False,
                                progress_cb=None) -> str:
    """Combine DiT, quantized text encoder, and VAE into a single checkpoint."""
    fmt, rank = resolve_format(fmt_name, rank, rank_was_set=False)

    with safe_open(dit_path, framework="pt", device="cpu") as handle:
        dit_keys = list(handle.keys())
    dit_prefix = detect_prefix(dit_keys, default="")
    already = any(k.endswith(".comfy_quant") for k in dit_keys)

    if not already and not dry_run:
        raise RuntimeError(
            "{} is a BF16 source. Quantize it first with quantize_krea2.py (or the "
            "Quantize node) and pass the result as dit_path.".format(dit_path))

    metadata = {
        "krea2_svdquant_tool_version": __version__,
        "krea2_allinone": "1",
        "krea2_allinone_dit": os.path.basename(dit_path),
        "krea2_allinone_te": os.path.basename(text_encoder_path),
        "krea2_allinone_te_format": te_format,
        "krea2_allinone_vae": os.path.basename(vae_path),
        "krea2_svdquant_variant": variant,
    }

    writer = StreamingWriter(out_path, metadata)
    with safe_open(dit_path, framework="pt", device="cpu") as dh:
        for out_key, src in plan_dit(dh, dit_keys, dit_prefix):
            s = dh.get_slice(src)
            writer.plan(out_key, _slice_dtype(s), s.get_shape())
    dit_planned = len(writer.entries)

    with safe_open(vae_path, framework="pt", device="cpu") as vh:
        for out_key, src in plan_vae(list(vh.keys())):
            s = vh.get_slice(src)
            writer.plan(out_key, _slice_dtype(s), s.get_shape())
    vae_planned = len(writer.entries) - dit_planned

    te_quantized = quantize_text_encoder(text_encoder_path, te_format, groupsize,
                                         device, writer, dry_run=True)
    te_planned = len(writer.entries) - dit_planned - vae_planned

    if te_quantized != TE_EXPECTED:
        raise RuntimeError(
            "expected {} quantizable text-encoder linears ({} layers x {}), matched {}. "
            "Layer naming in {} is not recognized.".format(
                TE_EXPECTED, TE_LAYERS, len(TE_QUANT_SUFFIXES), te_quantized, text_encoder_path))

    total = sum(e[3] for e in writer.entries)
    by = lambda pred: sum(e[3] for e in writer.entries if pred(e[0])) / 1024 ** 3

    stats = (
        "planned {} tensors, {:.2f} GiB (diffusion: {:.2f} GiB, encoder: {:.2f} GiB [{} linears -> {}], vae: {:.2f} GiB)"
        .format(len(writer.entries), total / 1024 ** 3, by(lambda k: k.startswith(DIT_PREFIX)),
                by(lambda k: k.startswith(TE_PREFIX)), te_quantized, te_format,
                by(lambda k: k.startswith(VAE_PREFIX)))
    )

    if dry_run:
        return stats + "\n[dry-run: nothing written, GPU untouched]"

    t0 = time.time()
    total_steps = len(writer.entries)
    step = 0

    with writer as w:
        with safe_open(dit_path, framework="pt", device="cpu") as dh:
            for out_key, src in plan_dit(dh, dit_keys, dit_prefix):
                w.write(out_key, dh.get_tensor(src))
                step += 1
                if progress_cb:
                    progress_cb(step, total_steps, "Copying DiT")
        with safe_open(vae_path, framework="pt", device="cpu") as vh:
            for out_key, src in plan_vae(list(vh.keys())):
                w.write(out_key, vh.get_tensor(src))
                step += 1
                if progress_cb:
                    progress_cb(step, total_steps, "Copying VAE")
        quantize_text_encoder(text_encoder_path, te_format, groupsize, device,
                              w, dry_run=False)
        if progress_cb:
            progress_cb(total_steps, total_steps, "Complete")

    elapsed = time.time() - t0
    final_size = os.path.getsize(out_path) / 1024 ** 3
    summary = (
        "baked all-in-one checkpoint: {} ({:.2f} GiB, {:.0f}s)\n"
        "  diffusion : {} (format {})\n"
        "  encoder   : {} (quantized to {})\n"
        "  vae       : {}\n"
        "{}".format(
            os.path.basename(out_path), final_size, elapsed,
            os.path.basename(dit_path), fmt_name,
            os.path.basename(text_encoder_path), te_format,
            os.path.basename(vae_path), stats
        )
    )
    return summary
