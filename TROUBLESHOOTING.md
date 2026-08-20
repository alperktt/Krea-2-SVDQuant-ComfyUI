# Troubleshooting

Split out of the README so the getting-started page stays short. Start with the **Krea2
SVDQuant Diagnostics** node (drop it between the loader and the KSampler, `mode=dispatch`),
or from a terminal:

```bash
python diagnose.py --no-load
```

`--mode all` adds the memory, dispatch, benchmark and profile tables in one go, which is
what to paste into a bug report.

## "It's slower than FP8 / slower than BF16"

Almost always this: **ComfyUI disables `comfy_kitchen`'s CUDA backend when torch was built
against CUDA < 13**, in `comfy/quant_ops.py`:

```python
if cuda_version < (13,):
    ck.registry.disable("cuda")
    logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.")
```

`convrot_w4a4_linear` resolves its backend per call, so with `cuda` disabled it falls
through to the eager implementation — which unpacks int4 to bf16 in Python and runs an
ordinary matmul. That is strictly slower than just running bf16, and the more aggressive
the format the worse it gets. The tell is that the ordering **inverts**: fp8 fastest, int8
middling, w4a4/svdq slowest, the exact opposite of the benchmark table above.

Check with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

If that prints anything below `13.0`, install a cu130+ torch build. The loader now prints
the resolved backend on every load and shouts if it isn't `cuda`.

## "It's no faster than my INT8 model"

This one is mostly **expected**, and the README used to invite the disappointment by
publishing a "2.9x vs BF16" column with no INT8 row next to it. Measured per sampling step on
a 3090 at 1024x1024, in one ComfyUI session (`tools/speed_bench.py`):

| checkpoint | s/step | + `TorchCompileModel` |
|---|---|---|
| INT8 + convrot | 0.993 | 0.886 |
| SVDQuant rank 256, actaware | 0.845 | 0.718 |
| W4A4, no low-rank branch | 0.678 | crashes, see below |

1.18x over INT8, 1.23x if both are compiled. At 8 steps that is 5.9 s against 7.3 s — a real
difference, and a small enough one that "I didn't notice any" is a fair report rather than a
misconfiguration. **The decisive win over INT8 is VRAM**: 8.50 GiB resident against 12.84 GiB.

Before concluding it is only that, check the two things that *are* misconfigurations:

1. `python diagnose.py --no-load` — if the backend is not `cuda`, read the section above;
   that costs far more than the INT8 gap.
2. If you are on `rank 256`, the low-rank branch is ~25% of a step (0.845 against 0.678).
   `Krea2-Turbo-W4A4-noLowRank` is the fastest checkpoint here by a wide margin if you can
   accept the fidelity cost — see BENCHMARKS.md for what that cost is.

## "TorchCompileModel crashes on the w4a4 / int8 checkpoint"

Known limitation, and it is specifically the checkpoints **without** a low-rank branch — the
ones that load through the stock `UNETLoader` rather than this pack's loader. The error is:

```
torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
  ... Cannot access data pointer of Tensor (e.g. FakeTensor ...). If you're using
  torch.compile/export/fx, it is likely that we are erroneously tracing into a custom
  kernel. To fix this, please wrap the custom kernel into an opaque custom op.
```

Dynamo reaches the `comfy_kitchen` kernel with fake tensors and it wants real pointers. The
SVDQuant loader in this pack avoids it by registering that call as an opaque custom op
(`krea2::w4a4_linear`), which is why `--format svdq` checkpoints compile and these do not.
Doing the same for a stock-loaded model needs a hook this pack does not currently have — the
same gap as the sage-attention guard further down this page.

Workarounds: use a `--format svdq` checkpoint (rank 16 costs ~0.4 GB over noLowRank), or drop
the TorchCompileModel node. Uncompiled, the branchless checkpoint is still the fastest option
in the table above.

If the custom op itself ever misbehaves — a `comfy_kitchen` update that moves the layout API
is the plausible way — the loader falls back to the old graph-break path on its own and says
so in its status output. `KREA2_W4A4_OP=0` in the environment forces that fallback, which is
also how the two rows in BENCHMARKS.md's compile table were measured against each other.

## "No speedup at all on my RTX 20-series" (Turing)

Different problem, and this one has no fix on our side. The backend resolves to `cuda`, the
kernel runs, nothing is misconfigured — int4 is simply not much faster than int8 on Turing:

- The instruction the fast path is built around, `mma.m16n8k64` (s4×s4→s32), is **Ampere and
  newer**. `comfy_kitchen`'s default convrot path targets `Sm89` and gates that instruction
  behind `__CUDA_ARCH__ >= 800`.
- SM 7.5 is served by separate kernels (`turing_int4.cu`, `turing_int8.cu`) built on a
  smaller MMA tile — `GemmShape<8, 8, 32>` versus the default int8 path's
  `GemmShape<16, 8, 32>` — and without `cp.async` prefetch, which is also SM80+.

Both formats run weaker kernels there, so switching to int8 does not dodge it either.
Rebuilding `comfy_kitchen` yourself will not change it: the SM75 kernels are what you get.

The int4 checkpoint is still worth downloading on these cards for the **smaller VRAM
footprint** — just do not expect the speed column of the benchmark table.

The diagnostics node and `diagnose.py --no-load` now say this explicitly when they detect a
compute-capability-7.x device, so you can tell "my setup is broken" apart from "my card
predates the instruction". Those kernels live in
[comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), not here — this repo ships no
CUDA build pipeline, and we have no Turing hardware to validate a replacement against, so
the honest answer is to report it upstream rather than have us ship an untested kernel.

## "Pin error." in the console

Harmless. It comes from ComfyUI core (`comfy/model_management.py`), not from this repo,
and means a weight could not be page-locked so a normal (unpinned) host copy was used
instead. Results are identical; you lose a little load/offload bandwidth. Windows caps
locked pages aggressively — `MAX_PINNED_MEMORY` there is 40% of system RAM — so it fires
routinely with a model this size. It is not specific to `svdq`; INT8 checkpoints trigger it
too. The diagnostics node prints your pinned-memory budget under `mode=env`.

## Iteration time jumps to 30-100 s once VRAM passes ~12 GB (and FP8/INT8 do not)

**Fixed.** If you are on a build that still does this, update.

The `svdq` loader used to opt the model out of ComfyUI's dynamic VRAM management:
`load_svdquant_w4a4` passed `disable_dynamic=True`, which pinned it to the classic
`ModelPatcher`. The stock `UNETLoader` does not do that, so the FP8, INT8 and `noLowRank`
checkpoints got the streaming patcher and the `svdq` ones did not. Past the point where the
model no longer fits, the pinned path falls back to streaming weights per module per step,
and a ~1 s iteration becomes 30-100 s. Nothing about the quantization was involved — it was
which patcher the loader asked for.

Both loaders now default to whatever ComfyUI would do for any other model, which on an
ordinary launch is the dynamic patcher. The **status output** on the loader node says which
one you got:

```
w4a4 + low-rank: attached 224 branches (rank 256, variant turbo), model_size 9.10 GiB,
ModelPatcherDynamic (dynamic vram: on), compile: 224 layers in-graph via krea2::w4a4_linear
```

`ModelPatcher (dynamic vram: off)` there means you are on the old path — either you set
`vram_management` to `classic`, or ComfyUI itself has dynamic VRAM off (`--highvram`,
`--novram`, `--gpu-only`, `--cpu`, `--disable-dynamic-vram`, or no working `comfy-aimdo`
install; the startup log says `DynamicVRAM support detected and enabled` when it is on).

### If you need the old behaviour back

One thing here is genuinely unmeasured: the streaming patcher moves the low-rank branch
buffers with a plain device copy rather than through its own allocator, so at rank 256 there
is 1.6 GiB of real VRAM its internal bookkeeping does not account for. If that turns out to
bite on your card, set the loader's **`vram_management`** input to `classic`, or run the
whole server with `KREA2_DISABLE_DYNAMIC=1`. Please open an issue if you have to — that is
the case this escape hatch exists to hear about.

### Still tight on a 12 GB card

The branch is real memory on top of the checkpoint — measured off the published files,
1.6 GiB at rank 256 and 0.4 GiB at rank 64 across the 224 blocks — and the rank-256 build is
9.10 GB against 7.50 GB for `noLowRank`. So the older advice still holds as a *size*
argument, just no longer as a workaround for a 30x slowdown:

1. **`Krea2-Turbo-W4A4-noLowRank.safetensors`** (7.50 GB) is ~9% faster per step than any
   rank, and gives up the low-rank branch's fidelity.
2. **`Krea2-Turbo-SVDQuant-W4A4-rank64.safetensors`** (7.90 GB) is 1.2 GB smaller than rank
   256 and statistically identical to it *as long as you never load a LoRA*.
3. Watch the loader's `model_size` line — it reports the real number, branch included.

## My LoRA renders changed after updating (and the image moves when VRAM gets tight)

Measured, real, and only half explained. Read this before concluding a LoRA is broken.

With **no LoRA loaded**, the two patchers are bit-identical — same seed, same graph, same
pixels, verified at three different VRAM budgets. With a **LoRA loaded** they are not:

| pair | PSNR | SSIM |
|---|---|---|
| `auto` vs `auto`, different VRAM budgets, three runs | ∞ | 1.000000 |
| `classic` full-load vs `classic` under pressure | 17.86 dB | 0.712 |
| `classic` vs `auto` | 15.4-15.9 dB | 0.61-0.65 |

So the instability is on the `classic` side: it renders a LoRA differently depending on how
much of the model got offloaded, because ComfyUI's classic path applies patches through
`LowVramPatch` at cast time when a layer is not resident and in place when it is. The
dynamic patcher has one path and uses it always, which is why all three `auto` runs agree
to the bit.

Until this release the loader pinned to `classic`, so **your previous LoRA output was the
classic one** — and, if you were near your VRAM limit, not necessarily reproducible either.
Now the default is `auto` and the result is stable. It is also a different image.

**What has not been established: which one is numerically right.** Determinism is not
correctness. Settling it needs the same LoRA applied to a reference the quantized path can
be compared against, and that reference does not exist here yet — see
[ROADMAP.md](ROADMAP.md). If you preferred the old look, `vram_management=classic` on the
loader node reproduces it, with the pressure-dependence that comes with it.

## Out of memory on a small card (and int8 works fine)

Fixed. The low-rank factors were attached as non-persistent buffers, which ComfyUI's
`module_size()` — the basis of every VRAM decision, including the lowvram split — could
not see, while `.to(device)` moved them anyway. Worse, the old branch cached its own
device move back onto the module, so once ComfyUI offloaded a layer the factors quietly
came back to the GPU and stayed there, outside all accounting. 0.4 GiB at rank 64 and
1.6 GiB at rank 256, which is the difference between fitting and not on an 8 GB card. INT8
checkpoints carry no branch, so they were never affected.

They are now published into `state_dict()` under their own `svdq_l1` / `svdq_l2` keys and
staged per call via `comfy.model_management.cast_to`, so they are budgeted and offloaded
like any other weight.

`mode=env` on the diagnostics node reports the factor devices, and **what you should see
depends on which patcher you are on** — read it after a render, not before, because at load
time both arms say `cpu`:

| `vram_management` | healthy `factor devices` |
|---|---|
| `auto` (dynamic, the default) | all on `cuda:0` — 448 of them at rank 256 |
| `classic` | mostly `cpu`, staged per call |

`cuda` under `auto` is not the old bug coming back; it is the streaming patcher keeping the
small always-needed branch resident and streaming the 4-bit weights instead, which is what
makes it fast. Measured both ways on a 3090 under `--reserve-vram 15`.

One gap remains and it is upstream, not here: `QuantizedTensor.nbytes` reports only the
packed weight, so the W4A4 `weight_scale` (~3 MB/layer) is still invisible to ComfyUI's
accounting for *any* w4a4 checkpoint, branch or no branch.

## LoKr / LoHa / OFT LoRAs, and adapters that touch no quantized layer

Supported since the loader stopped parsing LoRAs itself. Parsing is now
`comfy.lora.load_lora`, which returns one of ComfyUI's weight adapters per layer and already
knows every format and naming convention ComfyUI supports. From there:

* **plain up/down LoRA** -> folded into the low-rank branch, one pair of GEMMs for the whole
  stack no matter how many LoRAs are chained
* **LoKr, LoHa, OFT, GLoRA** -> ComfyUI's bypass contract `g(f(x) + h(x))`, which exists for
  exactly this case ("designed for quantized models where weights may not be accessible")
* **a `diff`/`set` patch, or anything with no bypass form** -> ComfyUI's normal path, with a
  warning, because that route rewrites the weight and requantizes the delta to 4 bits

**LoKr is not free at runtime, and the `adapters` input is the dial.** The Kronecker math
is the cost, not the plumbing: a `w2` of 1536x1536 across 4 groups is ~38.7 GMAC per layer,
~0.5 s over 224 layers. Caching the adapter weights on the GPU instead of staging them per
call was measured and changed nothing (2.01 vs 2.02 s/step), so the loader does not hold that
memory. Measured on a 3090 at **1440x1920**, base rank-256 checkpoint, a plain LoRA plus a
LoKr plus a txtfusion `.diff`:

| how | s/step | what you get |
|---|---|---|
| `adapters = bypass` (default) | 5.21 | the 4-bit weight is never touched, LoKr is exact |
| `adapters = bake` | 3.55 | ComfyUI rewrites the weight once; the LoKr delta is quantized to 4 bits with it |
| stock `LoraLoaderModelOnly` for everything | 3.22 | same, and the plain LoRA is requantized too |
| `tools/bake_adapter.py`, LoKr baked before quantization | 3.74* | no adapter at runtime at all, and the branch is refit around the merged weight |

\* that arm still runs one plain LoRA through the exact branch, which is the 0.35 s/step
between it and the stock row. Baking that one too takes it to the no-LoRA floor of 3.39.

If your sampler evaluates the model twice per step (`res_2s` and friends), double every
figure -- a user's 6-step `res_2s` graph came to 104 s on `bypass` against 82 s on the stock
loader, which is exactly this table times fourteen model calls.

**Which to use.** Occasional LoKr, quality first: leave it on `bypass`. Swapping LoKrs
constantly: `bake`. Always the same LoKr: bake it into a checkpoint once with
`tools/bake_adapter.py` -- it adds the delta to the **bf16** weight and then runs the
SVDQuant split, so the low-rank branch is fitted against the merged weight instead of the
LoRA being requantized on top of a finished checkpoint.

Two more consequences worth knowing:

**A LoRA that targets no quantized layer is no longer an error.** A txtfusion-only adapter
(`diffusion_model.txtfusion.projector.diff`, for instance) is a legitimate thing to load; the
old guard refused it on the theory that ComfyUI could not patch quantized weights at all,
which was measured and found false.

**Alpha follows ComfyUI's rule, not a heuristic.** For LoKr the scale is `alpha / dim` only
when `w1` or `w2` was rebuilt from an a/b pair; when both are full matrices the alpha is
ignored. That matters because some trainers write a sentinel alpha -- one real file carries
`9999220736.0` -- and dividing by a rank would either explode the activations or silence the
LoRA entirely, depending on which way you got it wrong.

## "no layer of X matched this model" (and the same LoRA "works" on the stock loader)

The LoRA's key prefix. Krea2 LoRAs come with either `diffusion_model.blocks.N.attn.wq...`
(ComfyUI native) or `transformer.blocks.N.attn.wq...` — a diffusers prefix in front of native
module names, which some PEFT-based extraction tools write. ComfyUI does accept a
`transformer.` prefix for Krea2, but only with *diffusers* module names
(`transformer.transformer_blocks.N.attn.to_q`), so the hybrid form matches nothing in
`comfy/lora.py`:

```
krea2_raw_to_turbo_r256.safetensors  (transformer.blocks.N...)  -> stock loader: 0 patches
same file, keys renamed to diffusion_model.                     -> stock loader: 224 patches
```

Zero patches means the stock loader is not failing — it is applying nothing at all and
generating as if no LoRA were loaded. That is why one of these can look like it "works
without the SVDQuant node": it does not, it is a silent no-op. This loader hard-fails on the
same file instead, which is the error above.

Both prefixes are accepted now, so the LoRA loads as-is. If you also want it usable from the
stock `LoraLoaderModelOnly`, rename the keys in the file itself:

```python
from safetensors import safe_open
from safetensors.torch import save_file

with safe_open("in.safetensors", "pt") as f:
    meta = dict(f.metadata() or {})
    sd = {k.replace("transformer.", "diffusion_model.", 1): f.get_tensor(k) for k in f.keys()}
save_file(sd, "out.safetensors", metadata=meta)
```

## Sage attention: `OutOfResources` or a broken image with krea2edit's `ref_boost`

Fixed, and it was never about the LoRA. `comfyui-krea2edit` turns `ref_boost` into an
*additive float* attention bias, and a float mask is the one input sage's kernels do not
take well:

* `sageattn_qk_int8_pv_fp16_triton` stages the mask tile through shared memory on top of the
  K/V pipeline. At head_dim 128 it asks for 139276 bytes against the 101376 an Ampere or Ada
  SM offers, so the launch dies with
  `triton.runtime.errors.OutOfResources: out of resource: shared memory`.
* the CUDA kernels launch, but the masked result drifts ~50x further from a BF16 reference
  than the same call unmasked — over 224 layers that is the black or scrambled image.

Boolean masks are fine on both, and so is the ordinary no-mask text-to-image step, which is
why this only ever showed up on the edit workflow. The loader now installs a guard that sends
float-mask attention calls to ComfyUI's stock attention and leaves everything else on sage:
you keep the speedup on every normal block and pay full price only on the biased ones. The
console logs `float attention mask -> stock attention` once per run when it fires.

## Random all-black frames with sage `auto` / `..._fp16_cuda`

Also fixed, and it is a sage kernel bug, not a quantization one. `sageattn_qk_int8_pv_fp16_cuda`
returns NaN for sequences shorter than one K/V block. Standalone, no ComfyUI, same inputs on
every call, RTX 3090, head_dim 128, bf16 and fp16 alike:

```
B=64  H=20  N=4..48    ->  7-10 of 10 calls non-finite
B=64  H=20  N=63..128  ->  0 of 10
B=592 H=20  N=12 / 32  ->  0-3 of 20, and it flips between processes
triton, every shape    ->  0 of 20
```

Krea2's `txtfusion` layerwise blocks attend over **12 tokens**, so every sampling step rolls
these dice, and a single NaN there poisons the latent: a completely black frame. `auto` picks
this kernel on sm80/86/87 (`sageattention/core.py`), which is why "auto" and "cuda fp16" were
the two settings people reported black images on.

The intermittency is what made it look like a LoRA or SVDQuant problem: the same graph is
clean on one model load and black on the next, and it survives across runs until the model is
reloaded. It is neither. Measured before the fix, 768px edit workflow, cold load each time:
roughly one black frame in three or four. After it: 10 of 10 clean, and byte-identical
outputs run to run.

The guard now routes any attention shorter than one K/V block to stock attention. Sage has
nothing to win on a 12-token attention, so this costs no measurable speed, and it applies
regardless of which sage kernel is selected. Console logs
`attention shorter than one K/V block ... -> stock attention` once per run.

One more thing worth knowing about `auto`: on sm80/86/87 its dispatch calls the kernel
*without* forwarding `attn_mask` at all, so krea2edit's `ref_boost` would be silently ignored
even when the output is not black. Prefer `sageattn_qk_int8_pv_fp16_triton` if you want the
bias honoured on the blocks the guard does not intercept.

Merging the LoRA into the checkpoint appeared to fix it only because that comparison also
dropped the `ref_boost` bias; the quantized model and its LoRA branch are not involved. If
you load a plain `--format w4a4` / `int8` checkpoint through the stock `UNETLoader`, the
guard is not installed — use the SVDQuant loader, or turn sage off for edit workflows.

## A re-saved checkpoint logs "left over keys in diffusion model"

Expected. Saving the model out of ComfyUI now includes the `svdq_l1` / `svdq_l2` keys, which
is what lets the file round-trip back into this loader — but the stock `UNETLoader` doesn't
know them and says so. Harmless.
