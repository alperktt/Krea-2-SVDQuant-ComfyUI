# Rank sweep & krea2edit LoRA benchmarks

Two independent tests on **Krea 2 Turbo**, all checkpoints produced by `quantize_krea2.py`
in this repo. Same seed across every checkpoint in a given test so images are directly
comparable. RTX 3090, cu130 torch build.

> **The quality scores in Tests 1 and 2 have been withdrawn (2026-07-27).** They came from
> an LLM judge (Gemini 3.5 Flash Lite) that turned out to be saturated: 9 of the 12 rows in
> the LoRA table scored a flat 10.00/10, and the whole t2i table spanned 0.46 points. An
> instrument with no discrimination left still produces a confident-looking ranking, so the
> old conclusion — "every SVDQuant rank is statistically indistinguishable" — was an artifact
> of the judge, not a property of the checkpoints. The speed columns are unaffected and kept.
>
> [Test 3](#test-3--paired-lpips-fidelity-with-and-without-a-lora) replaces it with LPIPS
> against a BF16 reference, multi-seed and paired, and reaches the opposite conclusion on the
> question that matters most here: **with a LoRA loaded, branch rank does change fidelity.**

## Test 1 — text-to-image rank sweep

10 prompts picked to stress different failure modes (dense text, hands, crowds, symmetry,
reflections, counting, complex composition), same seed, 1024x1024, 8 steps, cfg 1.0.
Checkpoints: BF16 reference, W4A4 with no low-rank branch, and SVDQuant rank 16/32/64/128/256
in both the fast single-shot split (`--refine-iters 0`, "no refine") and the alternating-refit
version (default, "refined") — see the main README's [Build your own checkpoint](README.md#build-your-own-checkpoint)
section for what that flag changes.

[`examples/rank_sweep_t2i_comparison/`](examples/rank_sweep_t2i_comparison/) — one grid PNG
per prompt, all 12 checkpoints side by side. The tiles have the judge's score printed under
them; those numbers are the withdrawn ones, baked into the images. Ignore them and look at
the images.

| id | prompt | stresses |
|---|---|---|
| 01_dense_text | "A rain-soaked neon diner sign at night, below it a handwritten chalkboard menu with three lines of text reading 'SOUP $4 / PIE $6 / COFFEE $2', reflections on wet asphalt, cinematic" | dense multi-line small text |
| 02_curved_text | "Close-up of a person holding a paper coffee cup with large bold curved text 'STAY WARM' printed around the cup, soft morning light, shallow depth of field" | large text on a curved surface |
| 03_hands_detail | "A violinist's hands mid-performance, fingers pressed on the strings, bow in motion with visible blur, studio lighting, extreme close-up, photorealistic" | fine anatomy, hands |
| 04_crowd_faces | "A busy Tokyo street crossing at dusk, dozens of pedestrians with distinct faces and expressions, neon signage in the background, wide angle, high detail" | many small faces |
| 05_symmetry_pattern | "A perfectly symmetrical Islamic geometric tile mosaic, intricate repeating star and polygon pattern, deep blue and gold, overhead flat lighting, ultra sharp" | geometric symmetry / repeating pattern |
| 06_multi_subject | "Two chefs in white uniforms plating a dish together in a busy kitchen, one holding tweezers placing a garnish, the other pouring sauce, steam rising, low angle shot" | multi-subject interaction |
| 07_reflections_glass | "A glass of iced whiskey on a dark wood bar, condensation droplets, warm bokeh lights reflected in the glass and the liquid, macro photography" | specular reflections / transparency |
| 08_logo_typography | "A vintage motorcycle fuel tank with a hand-painted logo reading 'IRON WOLF GARAGE' in bold serif letters, chrome and scratched paint texture, studio product shot" | stylized typography |
| 09_counting_objects | "A wooden table from directly above with exactly seven red apples arranged in a neat row next to three green pears, soft natural light, flat lay photography" | object counting |
| 10_complex_scene | "A fantasy marketplace street at golden hour, merchant stalls with hanging fabrics and baskets of spices, a dragon perched on a rooftop in the background, dense crowd, painterly digital art" | complex composition |

All ten grids: [GALLERY.md](GALLERY.md#rank-sweep-refined-vs-not).

## Test 2 — krea2edit LoRA (identity-preserving editing)

Same rank sweep, this time with the [Krea 2 Identity Edit LoRA](https://github.com/lbouaraba/comfyui-krea2edit)
on top, using its `Krea2EditModelPatch` / `Krea2EditGroundedEncode` nodes wired exactly per
the LoRA repo's example workflow (`ref_boost=4`, `fit_mode=fit`, `grounding_px=768`,
10 steps, cfg 1.0). Quantized checkpoints use this repo's **Krea2 SVDQuant LoRA Loader**
node instead of the stock one, which would dequantize the weight to apply the LoRA and then
requantize both together (see the main README's [LoRA](README.md#lora) section).

Source photographs are in [GALLERY.md](GALLERY.md#identity-edit-lora).

Three real stock photos of different women, resized to 1024x1536 before editing (feeding
multi-thousand-pixel originals straight into VAEEncode wastes VRAM/time for no quality
gain at this model's ~1MP working resolution).

| id | source | instruction |
|---|---|---|
| e1_paris_w1 | woman 1 | "Place her in Paris with the Eiffel Tower visible in the background, golden afternoon light, keep her exact face, hair, and outfit unchanged." |
| e2_sunset_sky_w1 | woman 1 | "Change the sky and background to a dramatic sunset with orange and pink clouds, keep her exact face and pose unchanged." |
| e3_horse_w2 | woman 2 | "Show her riding a horse outdoors on a countryside trail, keep her exact face, hat, and outfit unchanged." |
| e4_night_lights_off_w2 | woman 2 | "Change the scene to nighttime, turn off any lights, dark moody night sky, keep her exact face unchanged." |
| e5_paris_w3 | woman 3 | "Place her in Paris with the Eiffel Tower visible behind her, keep her exact face, hairstyle, and outfit unchanged." |
| e6_night_lights_off_w3 | woman 3 | "Change the lighting to nighttime with all lights turned off, dark and moody atmosphere, keep her exact face unchanged." |

All six edit grids: [GALLERY.md](GALLERY.md#identity-edit-lora).

## Results

### Speed — T2I rank sweep, 10 prompts, 1024x1024, 8 steps, cfg 1.0

| checkpoint | warm (s) | vs BF16 |
|---|---|---|
| BF16 (reference) | 18.80 | 1.00x |
| W4A4, no low-rank | 6.49 | 2.90x |
| SVDQuant r16, no refine | 7.10 | 2.65x |
| SVDQuant r16, refined | 6.95 | 2.71x |
| SVDQuant r32, no refine | 6.74 | 2.79x |
| SVDQuant r32, refined | 6.86 | 2.74x |
| SVDQuant r64, no refine | 7.16 | 2.63x |
| SVDQuant r64, refined | 7.16 | 2.63x |
| SVDQuant r128, no refine | 7.22 | 2.61x |
| SVDQuant r128, refined | 7.22 | 2.60x |
| SVDQuant r256, no refine | 7.54 | 2.49x |
| SVDQuant r256, refined | 7.77 | 2.42x |

### Speed — krea2edit LoRA, 6 edits, ref_boost=4, 1024x1024, 10 steps, cfg 1.0

| checkpoint | warm (s) | vs BF16 |
|---|---|---|
| BF16 (reference) | 44.86 | 1.00x |
| W4A4, no low-rank | 18.89 | 2.37x |
| SVDQuant r16, no refine | 23.54 | 1.91x |
| SVDQuant r16, refined | 23.96 | 1.87x |
| SVDQuant r32, no refine | 24.04 | 1.87x |
| SVDQuant r32, refined | 23.46 | 1.91x |
| SVDQuant r64, no refine | 23.51 | 1.91x |
| SVDQuant r64, refined | 23.44 | 1.91x |
| SVDQuant r128, no refine | 23.85 | 1.88x |
| SVDQuant r128, refined | 23.87 | 1.88x |
| SVDQuant r256, no refine | 25.09 | 1.79x |
| SVDQuant r256, refined | 25.10 | 1.79x |

One of the six `r16, refined` edit runs hit unrelated GPU contention on the test machine
(extra load from other applications) and was excluded from that row's average as a
measurement artifact, not a property of the checkpoint.

**Speed takeaway:** `W4A4, no low-rank` is the fastest checkpoint in both tests (~2.9x on
plain t2i, ~2.4x on the edit LoRA). The branch costs ~9-10% of step time and barely varies
with rank — rank 256 is only ~4% slower than rank 16.

## Test 3 — paired LPIPS fidelity, with and without a LoRA

Run with [`tools/fidelity_bench.py`](tools/fidelity_bench.py), which exists because the judge
above could not be trusted. **16 prompts x 2 seeds = 32 paired cells per arm**, LPIPS(AlexNet)
against a BF16 reference *generated in the same arm*, 1024x1024 / 8 steps / cfg 1.0 /
euler+simple, `qwen_image_vae`. Five arms:

| arm | LoRA | rank | reseed floor |
|---|---|---|---|
| `base` | none | — | 0.5468 |
| `lora` | `canon_krea2`, photographic style | 16 | 0.5193 |
| `lora2` | `bloomgirls-ultrarealism`, realism style | 32 | 0.5505 |
| `lora3` | `lenovo_krea2` | 16 | 0.5430 |
| `lora4` | `nicegirls_krea2` | 16 | 0.5008 |

`base`/`lora`/`lora2` cover the full checkpoint sweep; `lora3`/`lora4` were added later and
cover r64/r256/r256-actaware only (Test 4).

Two rules this test follows and the old one did not:

* **Paired, not marginal.** Prompt difficulty varies far more than checkpoints do (LPIPS
  0.23-0.42 across this set), so comparing two checkpoints' *means* buries the effect. Every
  number below is `mean(A_i - B_i)` over the same prompt+seed cells.
* **Each arm has its own reseed floor** (table above), because a LoRA changes how
  seed-sensitive the model is. Raw LPIPS is therefore **not comparable across arms** — only
  within one. Divide by the arm's own floor to compare across.

### The result: rank saturates at 64 without a LoRA, and much later with one

Mean LPIPS vs BF16 (lower = closer to the unquantized model):

| checkpoint | base | lora | lora2 |
|---|---|---|---|
| W4A4, no low-rank | 0.3954 | 0.3944 | 0.3808 |
| SVDQuant r16 | 0.3758 | 0.3453 | 0.3672 |
| SVDQuant r64 | 0.3325 | 0.3611 | 0.3684 |
| SVDQuant r128 | 0.3324 | 0.3378 | 0.3299 |
| SVDQuant r256 | 0.3378 | **0.2993** | **0.3215** |

Paired `r256 vs r64` (negative = r256 closer to BF16, `***` is |t| > 3):

| arm | mean | t | r256 wins |
|---|---|---|---|
| base | +0.0053 | 0.39 | 16/32 — a coin flip |
| lora | -0.0618 | **3.53*** | 26/32 |
| lora2 | -0.0468 | **2.79** | 23/32 |

Without a LoRA, **r64, r128 and r256 are indistinguishable from each other** (`r128 vs r64`
t=0.01 at 16/32; `r128 vs r256` t=0.31 at 17/32). The branch matters — all three beat
`no low-rank` at t=3.2-5.2 — but past rank 64 it stops buying anything.

Load a LoRA and that ceiling moves. r256 wins clearly in both LoRA arms, and the effect
replicates across two adapters of different rank and different style, and separately across
both halves of the prompt set (objects/scenes n=20, t=2.99; people n=12, t=2.29).

The sharpest way to see it — does the branch beat having no branch at all?

| | base | lora | lora2 |
|---|---|---|---|
| no-low-rank vs r64 | t=4.45*** | t=1.72 | **t=0.73** (12/32) |
| no-low-rank vs r256 | t=5.22*** | t=5.71*** | t=4.15*** |

**Under a LoRA a rank-64 branch is worth close to nothing over no branch at all, while a
rank-256 branch keeps its full advantage.** A sweep run without a LoRA would clear rank 64 as
"enough" — and be wrong for the way most people actually use these checkpoints.

**Recommendation: rank 256 if you use LoRAs, rank 64 if you never do.**

Two honest caveats. First, r64 lands slightly *below* r16 in both LoRA arms, which no simple
"more rank is better" story explains; individually neither comparison is significant (t=0.92
and t=0.06) so it may be noise, but it is consistent and we cannot account for it. Second,
this is two LoRAs at strength 1.0 and two seeds per prompt — enough to show the effect is not
adapter-specific, not enough to map how it scales with LoRA strength or rank.

Ruled out as confounds: r16/r64/r128/r256 all carry `refine_iters=100`, `groupsize=256`, the
same source `turbo.safetensors` and 224 branches each. Rank is the only variable between them.

Raw data: `fb/fidelity_bench.csv` and `fb/fidelity_bench_summary.json` in the ComfyUI output
directory, plus per-prompt contact sheets from
[`tools/contact_sheet.py`](tools/contact_sheet.py) — LPIPS says how far an image moved, not
whether it got worse, so the sheets are there to be looked at rather than trusted.

## Test 4 — activation-aware low-rank objective

`svdquant_split()` fits the low-rank branch by minimising `||W - (Q + L1 L2)||_F`. That
weights every input channel equally, which is only the right objective if every input channel
carries the same activation energy — and in this model they do not. `--act-stats` replaces it
with `||(W - (Q + L1 L2)) * d||_F`, where `d` is the per-input-channel activation RMS measured
on a real calibration pass, normalised to mean 1 and floor-clamped at 0.05. The branch then
spends its rank where the activations actually are.

Cost at inference: **zero**. Same tensor shapes, same format, same kernels — only the numbers
in `svdq_l1`/`svdq_l2` differ. It is a build-time change only.

Calibration: 8 prompts (deliberately disjoint from the 16 benchmark prompts, so the
checkpoint is not tuned on what judges it), BF16 model, hooks on all 224 branched linears via
`Krea2SVDQuantCaptureStart` / `Krea2SVDQuantCaptureSave`.

Paired `r256 vs r256-actaware`, same 32 cells per arm (**positive = act-aware is closer to
BF16**):

| arm | LoRA | mean | t | actaware wins |
|---|---|---|---|---|
| `base` | none | **+0.0553** | **4.68*** | 27/32 |
| `lora` | canon | -0.0220 | 1.67 | 9/32 |
| `lora2` | bloomgirls | +0.0118 | 0.67 | 19/32 |
| `lora3` | lenovo | +0.0098 | 0.76 | 15/32 |
| `lora4` | nicegirls | +0.0120 | 0.84 | 14/32 |

Without a LoRA the gain is large and unambiguous: LPIPS 0.3378 to **0.2825**, PSNR 15.20 to
**16.48**, SSIM 0.6339 to **0.6760**, and it beats *every* other checkpoint in the sweep
including r256. It also beats `no low-rank` at t=7.86 (2/32) — the widest margin any
checkpoint reaches in this benchmark.

Under a LoRA the gain shrinks to roughly nothing: three of the four LoRA arms are positive and
one (`canon`) is slightly negative at t=1.67 — inside the noise, and the only arm of the five
pointing that way. **Four of five agree, so this is not evidence of an act-aware/LoRA
incompatibility**, and nothing here says avoid it with a LoRA.

The plausible mechanism for the shrinkage was calibration mismatch — statistics captured with
no LoRA loaded describe activation energy the adapter then shifts. **That has now been tested,
and it is not the explanation.**

A second rank-256 checkpoint was built from the same BF16 source with the same settings, the
only difference being that its `--act-stats` file was captured with `bloomgirls` loaded (8
calibration prompts, disjoint from the 16 scored here, 264k tokens per layer). Scored on the
`lora2` arm it was calibrated for, against a BF16+LoRA reference, 16 prompts x 2 seeds:

| checkpoint | LPIPS | PSNR | SSIM |
|---|---|---|---|
| rank 256 | 0.3322 | 15.94 | 0.6849 |
| rank 256, act-aware (no LoRA during capture) | **0.3123** | **16.54** | **0.6968** |
| rank 256, act-aware (LoRA loaded during capture) | 0.3204 | 16.15 | 0.6931 |

| comparison | mean | se | t | wins |
|---|---|---|---|---|
| act-aware vs LoRA-calibrated act-aware | -0.0080 | 0.0159 | **-0.51** | 18/32 |
| plain rank 256 vs LoRA-calibrated | +0.0118 | 0.0198 | 0.60 | 14/32 |

Calibrating with the adapter loaded is, if anything, marginally *worse*, and every difference
is far under the arm's reseed floor of 0.5447. So the LoRA shrinkage is not a calibration
mismatch; what causes it is still open. Worth stating plainly because this was written up here
as the obvious next experiment, and it did not pay.

`r256-actaware` also beats `r64` in every arm (t = 3.44 / 2.69 / 3.49 / 3.06 / 1.89), so
nothing here reverses the Test 3 recommendation either.

**Recommendation: build with `--act-stats`, whether or not you use LoRAs.** Free at runtime,
the largest fidelity gain measured in this repo without one, no worse with one — there is no
configuration in which the plain objective is the better choice.

```bash
python quantize_krea2.py turbo.safetensors --format svdq --rank 256 \
  --act-stats krea2_act_stats.safetensors
```

Ruled out as confounds: `r256` and `r256-actaware` share source, rank, `refine_iters=100`,
`groupsize=256` and all 224 branch sites. The activation weighting is the only variable.
With `act_rms=None` the code path is bit-identical to the old one given the same RNG seed.

## Speed and per-layer accuracy

All numbers measured on an **RTX 3090 24GB**, 1024x1024, 8-step Euler/simple sampling,
`cfg=1.0` (Krea 2 Turbo distilled schedule), from the same BF16 source checkpoint, on a
**cu130 torch build** (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — on an older build every
one of these numbers gets worse, and the ordering inverts).

These are Turbo numbers. The base model at ~50 steps with CFG does roughly 12x the
sampling work per image, so the absolute seconds do not transfer; the *ratios* between
formats do, since they come from the same per-layer kernels.

#### End to end, per image

Two numbers matter and are easy to conflate: **first run after switching checkpoints**
(pays disk-to-VRAM load time, ~9-15s here) and **warm run** (model already resident,
what you get generating multiple images back to back). ComfyUI's own progress bar
("`8/8 [00:07<00:00, 1.09it/s]`") only covers the KSampler loop; "`Prompt executed in
X seconds`" is CLIP load/encode + model staging + sampling + VAE decode + save combined
— the two numbers can differ by 2x on a cold run.

| checkpoint | size | first run (cold) | warm run | vs. BF16 |
|---|---|---|---|---|
| BF16 (unquantized reference) | 24.48 GB | 25.3 s | 21.3 s | 1.0x |
| FP8 e4m3, scaled (emulated on Ampere) | 12.24 GB | 22.2 s | 19.2 s | 1.1x |
| INT8 tensorwise + convrot (not in this upload) | 13.16 GB | 13.3 s | 10.4 s | 2.0x |
| **W4A4 + convrot, no low-rank branch** | 7.50 GB | 10.3 s | **10.1 s** | 2.1x |
| **W4A4 + SVDQuant low-rank, rank 16/64/128** | 7.6-8.3 GB | ~19.3 s | **10.1-10.2 s** | 2.1x |

Rank does not measurably change warm speed — CLIP text-encode (Qwen3-VL 4B) and VAE
decode overhead dominate a single 1024x1024/8-step/batch-1 image and mask the low-rank
branch's cost.

**That masking is why this table should not be used to compare formats**, and it is how the
row above ended up reading "INT8 10.4 s against W4A4 10.1 s" — a 3% gap that is almost
entirely fixed cost. `tools/speed_bench.py` measures the sampling loop on its own, by timing
each arm at two step counts and taking the slope, so everything that does not scale with
steps cancels. Same 3090, one session, three timed runs per cell:

| checkpoint | s/step | + `TorchCompileModel` |
|---|---|---|
| INT8 + convrot (`int8-fast`, per-row) | 0.993 | 0.886 |
| **W4A4 + SVDQuant low-rank, rank 256 actaware** | 0.845 | **0.718** |
| **W4A4 + convrot, no low-rank branch** | **0.678** | crashes (TROUBLESHOOTING.md) |

Read against those numbers: W4A4 is **1.18x** faster than INT8 per step, **1.23x** with both
compiled, and the **low-rank branch is ~25% of a step at rank 256** (0.845 against 0.678) —
not the 9-10% quoted elsewhere, which was measured at rank 64. TorchCompileModel is worth
1.10x on the svdq checkpoint and 1.12x on INT8; the "~20-25%" figure below predates the
per-step harness and describes the sampling portion of an end-to-end run.

Sessions drift: INT8 measured 0.948 s/step in one ComfyUI process and 0.993 in another, so
only rows from a single run are comparable. Each `speed_bench.py` invocation runs its whole
arm matrix in one process for that reason.

**FP8 is not faster than BF16 on Ampere** — there are no FP8 tensor cores on this
architecture, so ComfyUI casts to bf16 and calls cuBLAS. It's included here because it's
the most common recommendation online for "quantizing Krea 2," and the numbers show why
that advice doesn't hold on 30-series cards. **INT8 is the fastest *accurate* option**
measured, but is not part of this upload (available via `quantize_krea2.py --format
int8` on your own BF16 checkpoint).

#### Per-layer accuracy (cosine similarity / relative error vs. BF16 original)

Measured on real captured activations from a Krea 2 Turbo forward pass (not synthetic
noise), across representative attention and MLP layers:

| format | cosine | relative error | per-layer time |
|---|---|---|---|
| bf16 (reference) | 1.00000 | - | 1.22 - 3.48 ms |
| **int8 + convrot (Hadamard rotation)** | 0.99999 | 0.35 - 0.63% | 0.39 - 1.09 ms |
| int8 per-channel (no rotation) | 0.99993 | 0.45 - 1.47% | 0.35 - 1.01 ms |
| fp8 e4m3, scaled | 0.99996 | 0.39 - 1.28% | 1.95 - 5.14 ms |
| nvfp4 | 0.99968 | 0.74 - 4.00% | 1.49 - 3.93 ms |
| w4a4 + convrot, rank-64 low-rank branch | 0.99933 - 0.99997 | 0.72 - 8.38% | 0.39 - 1.09 ms |
| w4a4 + convrot, no low-rank branch | 0.99569 - 0.99908 | 1.49 - 9.29% | 0.23 - 0.67 ms |

The Hadamard rotation used by `convrot` already does most of what SVDQuant's low-rank
branch does (both are outlier-mitigation strategies), so on top of `convrot_w4a4` the
low-rank branch buys noticeably less than in the original SVDQuant paper — it roughly
halves the error rather than eliminating it. **`int8` is the more accurate choice if
quality matters more than raw speed; `svdq` is the faster, smaller choice.**

#### Rank sweep

`--format svdq --rank N` was run for N = 16, 32, 64, 128, 256. Checkpoint sizes:

| rank | size |
|---|---|
| 16 | 7.60 GB |
| 32 | 7.70 GB |
| 64 | 7.90 GB |
| 128 | 8.30 GB |
| 256 | 9.10 GB |

This is an experimental project — the rank sweep is deliberately shipped so people can
try the tradeoff themselves rather than take one number on faith. If you benchmark other
ranks or find a case where one clearly wins, open a discussion on this repo.

To measure any of this yourself against a BF16 reference:

```bash
python tools/fidelity_bench.py generate --output-dir <ComfyUI/output>
python tools/fidelity_bench.py score    --output-dir <ComfyUI/output>
python tools/contact_sheet.py           --output-dir <ComfyUI/output> --jpeg 90
```

`generate` is idempotent, so it can be grown one `--seeds` or `--checkpoints` at a time.
`score` prints the reseed floor next to every result, which is the number that decides whether
a difference is claimable. `contact_sheet` builds the sheets in [GALLERY.md](GALLERY.md) --
the numbers rank, the sheets adjudicate.

#### Where the remaining time goes (profiled, `svdq r64`, single denoise step, 175.7 ms)

| component | share |
|---|---|
| W4A4 GEMM (native `comfy_kitchen` cutlass kernel) | 37% |
| elementwise / norm / RoPE / dtype casts | 34% |
| attention (cuDNN flash) | 9% |
| low-rank branch (2 bf16 GEMMs per quantized layer) | 9% |
| W4A4 activation quantization | 8% |

A third of a step is small elementwise kernels, which is why `torch.compile` (backend
`inductor`) helps: add a **TorchCompileModel** node after the loader. First run after loading
pays ~50s of compilation; subsequent runs are warm.

Getting inductor to see that third of the step took two goes. Dynamo cannot trace
`F.linear(x, QuantizedTensor)` at all — it reaches the `comfy_kitchen` kernel with fake
tensors, which have no data pointer. The first workaround marked every quantized Linear as a
deliberate graph break, which compiled but cost **two breaks per layer, 448 across the 224
blocks** (`diagnose.py --mode compile`): inductor never saw two consecutive layers in one
graph, and cudagraphs was off entirely. The loader now registers the kernel call as an opaque
custom op instead (`krea2::w4a4_linear`), which is what PyTorch's own error message asks for,
and a compiled step becomes **one graph, zero breaks**. Measured against `KREA2_W4A4_OP=0`,
which forces the old path:

| | graph breaks | s/step eager | s/step compiled |
|---|---|---|---|
| graph-break workaround | 448 | 0.846 | 0.730 |
| opaque custom op | **0** | **0.833** | **0.696** |

The eager gain is not the op — an eager call skips it — but the memoized backend resolution
that came with it: `convrot_w4a4_linear` re-resolves its backend on every call, revalidating
seven constraints 1792 times per image, with no caching anywhere in kitchen's registry.

Fidelity is unchanged, and checking that needed a noise floor: the kernel is not
run-to-run deterministic. Two renders of the same seed through the *same* path in two
processes differ by max 32/255 (mean 0.143); op-on against op-off differ by **less** than
that (max 28, mean 0.108). The difference is inside the floor.
