# Roadmap

What is worth doing next, why, and what it is expected to buy — with the measurement behind
each estimate so none of them is a guess. Ordered by value against effort.

Everything here is *open*. Things that were tried and did not work live in
[BENCHMARKS.md](BENCHMARKS.md) as negative results, because a failed experiment nobody wrote
down gets run twice.

---

## 1. Let the svdq loader use ComfyUI's dynamic patcher

**Status: done.** `disable_dynamic=True` is gone from both loaders; they default to whatever
ComfyUI would do for any other model, which on an ordinary launch is `ModelPatcherDynamic`.
The `vram_management` input (and `KREA2_DISABLE_DYNAMIC=1`) restores the pin.

The pin was waiting on two conditions. Both hold on current ComfyUI, and both were
measured, not argued from the source:

1. **The branch buffers survive the streaming patcher.** `ModelPatcherDynamic.load()` ends
   by walking `self.model.named_buffers(recurse=True)`, moving every buffer to the load
   device with `set_attr_buffer` and stashing the original in `self.backup_buffers`. Unload
   is symmetric: `partially_unload()` falls through to `restore_loaded_backups()`.

   Measured on a 3090 with `--reserve-vram 15` (~7 GB usable against an 8.51 GiB model),
   read from the diagnostics node *after* a render — which is the only time it means
   anything, since both arms show `cpu` at load time:

   | `vram_management` | factor devices | weight devices | lowvram |
   |---|---|---|---|
   | `auto` | **448 × `cuda:0`** | 224 × `cpu` | False |
   | `classic` | 368 × `cpu`, 80 × `cuda:0` | 184 × `cpu`, 40 × `cuda:0` | True |

   So `add_low_rank`'s `cast_to` really is a no-op under `auto` — the opposite of the
   feared per-step multi-MB staging copy — and the division of labour is the one you would
   pick by hand: the small branch every step needs stays resident, the 4-bit weights
   stream.
2. **`module_size()` still counts them.** `_load_list()` derives its per-module budget from
   `comfy.model_management.module_size`, which sums `state_dict()` — exactly what
   `_publish_in_state_dict` exists to feed. The 1.6 GiB at rank 256 is visible to the
   budget.

Two things that also had to be true and are:

* The compile fast path self-disables rather than reading a stale weight. Under the dynamic
  patcher the resident copy lives in `_v_weight`/`_prefetch` while `module.weight` stays on
  the host, so `_install_custom_op`'s per-call `weight._qdata.device != x.device` check
  sends the call to the stock forward. Correct, at the cost of the in-graph op — which only
  matters under `TorchCompileModel`.
* The LoRA node is unaffected: it works through `add_object_patch`, and
  `ModelPatcherDynamic.patch_model` delegates object patches to `super()`.

**What is still open:** the buffers are moved with a plain device copy, outside the dynamic
allocator's `allocated_size` accounting. Real VRAM, invisible bookkeeping. That is why the
escape hatch shipped with the fix rather than the fix shipping alone.

## 1b. Decide which patcher renders a LoRA *correctly*

**Status: open, and the only loose end left by the unpin.** Without a LoRA the two patchers
are bit-identical. With one they are not, and `classic` is additionally not reproducible --
its output moves with how much of the model was offloaded:

| pair | PSNR | SSIM |
|---|---|---|
| `auto` vs `auto`, three runs across two VRAM budgets | ∞ | 1.000000 |
| `classic` full-load vs `classic` under `--reserve-vram 15` | 17.86 dB | 0.712 |
| `classic` vs `auto` | 15.4-15.9 dB | 0.61-0.65 |

The branch bookkeeping is identical in both arms -- 224 quantized layers branched, 32
patched normally, logged the same either way -- so this is not the node failing to attach
something. The suspects are the 32 non-quantized layers, which go through ComfyUI's own
`add_patches`, and the fact that the classic path has two ways of applying a patch (in place
when resident, `LowVramPatch` at cast time when not) while the dynamic path has one.

`auto` being invariant is a good sign and not a proof. **Nothing here establishes which
image is the faithful one.** What would: apply the same LoRA to a BF16 Krea 2 and compare
both arms against it with `tools/fidelity_bench.py` -- the harness already does paired
LPIPS against a BF16 reference, so this is a run, not new machinery. Until that happens the
honest statement is the one in TROUBLESHOOTING.md: the default changed, the output changed,
and the new one is at least reproducible.

## 1c. MiniMax H3 (issue #5): quantizes, loads and renders — quality unmeasured

**Status: the mechanism works end to end; the reason to want it is still unproven.**

`quantize_krea2.py` only ever knew one Krea 2 thing: which leaf names under `blocks.N.` are
worth quantizing, as a module-level tuple five functions closed over. That is now
`ARCHITECTURES`, detected from the checkpoint by `detect_architecture()`. Krea 2 is eight
leaves over 28 blocks; MiniMax H3 is four over 50 (`attn.qkv_proj`, `attn.out_proj`,
`mlp.fc1`, `mlp.fc2` — Q/K/V arrive fused, which is also why `--rank-alloc gqa` is refused
there: it is defined over `attn.wk`/`attn.wv`, which H3 does not have as separate tensors).
This is the same generalization item 7 needs for the text encoder.

Measured, from `Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16`
(40.2 GB, already in ComfyUI's own layout, so no key conversion is needed at all):

| | |
|---|---|
| build | 200/200 layers, rank 64, refine 100, **11.11 GB from 37.46 GB, 185 s** |
| load | `MiniMaxH3Model`, 200/200 branches attached, `ModelPatcherDynamic` |
| kernel | `cuda (comfy_kitchen.backends.cuda.convrot_w4a4_linear)` |
| render | 640×352, 5 frames, 20 steps, no turbo LoRA — coherent frames |

Branch cost on H3, computed from the real shapes: rank 64 is 0.56 GiB over a 8.97 GiB
4-bit body (6%), rank 256 is 2.22 GiB (25%) — the same proportion Krea 2 pays.

The kernel was never the risk: a plain `convrot_w4a4` H3 already exists publicly and carries
exactly the config `--format w4a4` writes here (`{"format": "convrot_w4a4",
"convrot_groupsize": 256}`, 1-D scales). What it does not carry is a low-rank branch.

### The branch does not pay for itself here — measured, one cell

The reason to want SVDQuant on H3 was that plain 4-bit reportedly costs quality, and the
low-rank branch is the mechanism that absorbs quantization residual. On the picture, at
rank 64, that did not reproduce.

Against the 40 GB bf16 source, 4-step turbo LoRA, 640×352, 5 frames, one prompt, one seed,
mean over frames:

| | LPIPS ↓ | PSNR | SSIM | size |
|---|---|---|---|---|
| **noise floor** — two bf16 runs, different seeds | 0.5844 | 12.27 dB | 0.3669 | — |
| svdq rank 64 (branch) | **0.5645** | 13.57 dB | 0.4573 | 11.11 GB |
| w4a4 noLowRank (no branch) | 0.5963 | 13.57 dB | 0.4729 | 10.56 GB |

**On LPIPS the branch is worth 0.032**, and it is the difference between landing just inside
the noise floor and just outside it. PSNR and SSIM cannot see this at all — they score the
two arms identically to two decimal places, and SSIM ranks the branchless one *ahead*.

That disagreement is the finding, and it is a correction: an earlier revision of this entry
read "the branch buys 0.00 dB and costs 0.55 GB", concluded from PSNR/SSIM alone. This repo
already argues against exactly that mistake -- `fidelity_bench.py` scores on LPIPS for a
reason -- and the claim was made anyway. It is withdrawn.

What survives is smaller and less comfortable. All three arms sit around LPIPS 0.58 against
the reference, so **4-bit H3 is a long way from bf16 whatever the branch does**; 0.032 is a
small margin next to a 0.58 floor. The branch helps and does not rescue. And this is still
one cell -- one prompt, one seed pair, five frames -- which is the methodology
`tools/fidelity_bench.py` exists to avoid.

An earlier comparison against the published plain-int4 H3 was discarded rather than
reported: that file is broken, and a number against a broken baseline is worse than none.

### What would actually settle it

* **Audio.** Still the real complaint and still unmeasured: H3 generates audio in the same
  forward pass, and `pixel_metrics` is stills only. This needs an audio metric — new
  machinery, not another run — and it is the one measurement that could still justify the
  branch after the result above.
* **Act-aware calibration.** On Krea 2 this was the large fidelity win (LPIPS 0.3378 →
  0.2825), much larger than rank. No statistics have been captured for H3.
  `svdquant_capture` now matches the union of every architecture's leaves, so it will hook
  H3's layers; nobody has run it.
* **A real bench.** Multiple prompts and seeds, paired, the way Test 3 was run.
* **`ref2va`.** Untouched. It and `fl2va` fail closed, so it needs its own build.

So: "H3 can be SVDQuantized" is measured and true. "H3 *should* be" currently has one
measurement pointing at no, on the half of the model that was never the complaint. Until
that is resolved, this stays a roadmap entry and **no H3 checkpoint is published or listed
in the README** — the branchless `--format w4a4` build is the one that looks worth having,
and it needs no code from this repo to load.

## 2. Act-aware at ranks other than 256

**Status:** never tried. `--act-stats` shipped after the rank sweep was run, and only a
rank-256 build was ever made. So "is act-aware rank 64 as good as act-aware rank 256" is
simply unasked.

**Why it matters:** act-aware's large win (LPIPS 0.3378 → 0.2825, t=4.68) was measured with
no LoRA, and the earlier rank sweep found rank 64 ≈ rank 256 *without* a LoRA (rank only
clearly mattered with one). If those two hold together, rank-64 act-aware is 1.3 GB smaller
and 6.6% faster for the same fidelity, and the download table's recommendation changes.

**What it is not:** a speed lever. Measured per-step on a 3090 at 1024², one session:

| checkpoint | s/step | vs noLowRank |
|---|---|---|
| noLowRank | 0.670 | — |
| rank 16 | 0.766 | +14.3% |
| rank 64 | 0.774 | +15.5% |
| rank 128 | 0.787 | +17.5% |
| rank 256 | 0.829 | +23.7% |

Most of the branch's cost is *fixed*: two extra bf16 GEMM launches, a full-size add and two
allocations per layer, all paid at rank 16 as much as at rank 256. Dropping 256 → 64 is worth
6.6%; dropping the branch entirely is worth 19%. (This also corrects the older README claim
that rank 256 is "~4% slower than rank 16" — it is 8.2%.)

**Cost:** two builds (~14 min each) plus a `base`-arm fidelity run (~25 min).

## 3. Make `TorchCompileModel` work on branchless checkpoints

**Status:** open bug, one failed attempt.

`--format w4a4` and `--format int8` checkpoints load through the stock `UNETLoader`, which
never registers the kitchen kernel as an opaque custom op, so compiling one **raises**:

```
Cannot access data pointer of Tensor (e.g. FakeTensor ...) ... please wrap the custom
kernel into an opaque custom op.
```

**Why it matters:** `noLowRank` is the fastest checkpoint here at 0.670 s/step, and it is the
one that cannot be compiled. Compiled it would be roughly 0.58 s/step — about 1.5x over
compiled INT8, and the fastest Krea 2 configuration on Ampere by a clear margin. Right now
that combination is unreachable.

**What was tried:** a node that walks the loaded model and installs `krea2::w4a4_linear` on
every quantized Linear. It patched all 224 layers (confirmed in its own log) and the crash was
unchanged — Dynamo kept tracing ComfyUI's stock `ops.py` forward. The same module compiles
cleanly in isolation (`graphs 1, breaks 0`), and a fresh server with no prior eager pass fails
identically, so it is not a stale patch being restored by `unpatch_model`. Cause is somewhere
in ComfyUI's patcher/compile lifecycle and was not found; the node was reverted rather than
shipped broken.

**Next probe:** compile the whole `diffusion_model` outside ComfyUI with the op installed and
see whether Dynamo honours an instance-level `forward` on a *child* module in that setting.
That separates "Dynamo ignores instance patches on children" from "something in ComfyUI
replaces the module".

## 4. Fold LoKr/LoHa/OFT deltas into the low-rank branch by SVD

**Status:** Completed in `experimental/all-in-one`. Added as the `svd delta` mode for `Krea2SVDQuantLoraLoader`, decomposing weight deltas at load time via `torch.svd_lowrank` and merging low-rank factors into the low-rank branch with zero per-step runtime overhead.

## 5. Mixed precision: keep the first and last blocks at W8A8

**Status:** standard SVDQuant lever, never tried here. Costs file size, buys fidelity. No
estimate — nothing has been measured, which is exactly why it is on this list rather than
above the items that have been.

## 6. Benchmark the base checkpoint

`Krea2-Base-SVDQuant-W4A4-rank256-actaware` is published and has **never been through the
paired benchmark** (README says so). `tools/fidelity_bench.py --variant base` already
supports it. Not a quality lever — a claim with no evidence behind it.

## 7. All-in-one checkpoint: diffusion + text encoder + VAE in one file

**Status:** Completed in `experimental/all-in-one`.
- Standalone builder: `tools/build_all_in_one.py` (streaming two-pass safetensors assembler).
- In-Graph node: `Krea2SVDQuantQuantizeAllInOne` in `svdquant_quantize.py`.
- Loader node: `Krea2SVDQuantCheckpointLoader` in `svdquant_w4a4.py`.
- Workflows: `workflows/krea2_turbo_all_in_one_t2i.json` and `workflows/krea2_quantize_all_in_one.json`.
- Quantizes the 252 language projections of Qwen3-VL 4B to 4-bit (`convrot_w4a4`, ~3.2 GB) while keeping the vision tower (315 tensors) and embeddings in native precision for 100% prompt fidelity. Total size: ~11.7–12.3 GB (down from 33.5 GB BF16 / 15.5 GB split).

**ComfyUI already supports this, which is the surprise.** `class Krea2` in
`comfy/supported_models.py` defines `vae_key_prefix = ["vae."]` and
`text_encoder_key_prefix = ["text_encoders."]`, and its `clip_target()` runs `llama_detect`
over `text_encoders.qwen3vl_4b.transformer.`, which resolves to
`comfy.utils.detect_layer_quantization` — a check for `.comfy_quant` markers that is entirely
format-agnostic. Any layer we quantize with the machinery already in `quantize_krea2.py`
switches the encoder onto `mixed_ops` and loads. So a **branchless** all-in-one file
(`--format w4a4` or `int8`) needs no ComfyUI change and no node from this repo: it loads with
the stock `CheckpointLoaderSimple`.

An **svdq** all-in-one does need a node, because the `svdq_l1`/`svdq_l2` buffers have to be
popped and attached the way `load_svdquant_w4a4` does. That is a wrapper around
`comfy.sd.load_state_dict_guess_config`, not new mechanism.

### The size arithmetic

Measured file sizes, not estimates, except where marked:

| component | today | all-in-one target |
|---|---|---|
| diffusion | 9.78 GB (svdq rank 256) | 8.48 GB (rank 64) or 8.05 GB (noLowRank) |
| text encoder | 5.24 GB (`qwen3vl_4b_fp8_scaled`) | ~3.3 GB at 4-bit *(estimated)* |
| VAE | 0.51 GB | 0.51 GB, unquantized |
| **total** | **15.5 GB across 3 files** | **~11.9–12.3 GB in 1 file** |

The 4-bit encoder estimate scales the BF16 encoder (8.88 GB) by the ratio the diffusion model
gets (24 GB → 8.05 GB, ~34%), and is the number most likely to be wrong: a 4B LLM spends a much
larger share of itself on embeddings than a DiT does, and embeddings do not quantize here.

Note what this does **not** buy: the text encoder and the diffusion model are never resident at
the same time — ComfyUI encodes, unloads, then samples. So this is a disk, download and
correctness win, and a VRAM win during the encode pass only. Sampling speed and sampling VRAM
do not move. Worth saying out loud before anyone reads "12 GB instead of 15.5 GB" as a VRAM
figure.

### What has to be built

1. **A second layer set.** `_QUANT_SUFFIXES` targets `blocks.N.{attn,mlp}.*`, which is the DiT.
   The encoder is a Qwen3 LLM: `layers.N.self_attn.{q,k,v,o}_proj`,
   `layers.N.mlp.{gate,up,down}_proj`. `is_target` has to take the set rather than close over
   one.
2. **Leave the vision tower alone**, at least at first. Qwen3-VL carries one, krea2edit's image
   conditioning path may use it, and quantizing something to find out whether it is dead weight
   is the wrong order of operations.
3. **A combiner**, streaming three state dicts into one with the prefixes above. The BF16 source
   is 24 GB and the encoder another 8.88 GB, so it has to stream the way `convert()` already
   does rather than build the dict in memory.
4. **An svdq checkpoint loader node**, if the svdq variant is wanted.

### The risk, and how to measure it

4-bit is a much bigger ask of an LLM than of this DiT. The DiT tolerates it because convrot
spreads the outliers and the low-rank branch absorbs what is left; LLM activations have
outlier channels severe enough that a whole literature (SmoothQuant, AWQ) exists about them.
INT8 is not an alternative — it is the same byte count as the FP8 encoder we already ship, so
4-bit is the only setting that changes the number.

Failure will not look like noise. It will look like **prompt adherence quietly getting worse**:
a dropped clause, a colour that drifts, a count that stops being respected. That is exactly the
kind of regression the existing harness catches, because it is measured against a BF16
reference at fixed seeds — hold the diffusion model constant, swap only the encoder, and run
`tools/fidelity_bench.py`. The dense-text and counting prompts in the benchmark set are the
ones to read first.

Act-aware calibration on the encoder is the natural follow-up if plain 4-bit is close but not
close enough. It needs its own capture: the hooks would ride on `CLIPTextEncode`, not on the
sampler, so `svdquant_capture.py` needs a second pair of nodes rather than a flag. Do not build
that before the plain 4-bit measurement says whether it is needed.

---

---

## Measured and rejected — do not redo these

* **Fusing qkv+gate into one GEMM.** Direct kernel timing at 4096 tokens: 4 separate linears
  2.811 ms against 1 fused 2.074 ms, i.e. 0.736 ms per block and 20.6 ms per step across 28
  blocks — **2.4%**. Not worth a checkpoint format change plus re-uploading every published
  file. The older "~4%" estimate came from a nunchaku head-to-head, not a direct measurement.
* **Recalibrating `--act-stats` with a LoRA loaded.** See BENCHMARKS.md — the LoRA-calibrated
  build is marginally *worse* (t = -0.51), so calibration mismatch is not why act-aware goes
  null under a LoRA.
* **`--rank-alloc gqa`.** Measured, t = +0.55, no effect in either direction.
* **Folding the plain-LoRA branch into the svdq branch.** The algebra works, but
  concatenating factors duplicates them, which at rank 256 is ~2.6 GB — it spends the VRAM
  advantage that is this project's main win over INT8. Folding in place cannot be undone on
  unpatch. `torch.compile` already fuses the intermediate allocations this would save.
