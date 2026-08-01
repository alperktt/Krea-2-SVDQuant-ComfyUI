# Roadmap

What is worth doing next, why, and what it is expected to buy — with the measurement behind
each estimate so none of them is a guess. Ordered by value against effort.

Everything here is *open*. Things that were tried and did not work live in
[BENCHMARKS.md](BENCHMARKS.md) as negative results, because a failed experiment nobody wrote
down gets run twice.

---

## 1. Act-aware at ranks other than 256

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

## 2. Make `TorchCompileModel` work on branchless checkpoints

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

## 3. Fold LoKr/LoHa/OFT deltas into the low-rank branch by SVD

**Status:** idea, not started. Would become a third value for the LoRA node's `adapters`
input, alongside `bypass` and `bake`.

**The problem it solves.** Today's two options both cost something:

| mode | s/step (3090, 1440x1920, r256 + LoRA + LoKr + a `.diff`) | what it costs you |
|---|---|---|
| `bypass` (default) | 5.21 | exact, but the adapter runs every forward |
| `bake` | 3.55 | fast, but the LoKr delta is requantized to **4 bits** |
| stock loader | 3.22 | same, and the plain LoRA is requantized too |

The bypass cost is inherent, not an implementation flaw — ComfyUI's `h()` already does the
efficient nested contraction and never materialises the Kronecker product
(`comfy/weight_adapter/lokr.py:146-163`). For a 6144→6144 layer with 4 groups it is ~38.7
GMAC per layer in **bf16**, against the base layer's 154 GMAC in **int4**. A quarter of the
arithmetic on hardware that is ~8x slower for it, which is why it roughly doubles the layer.

**The idea.** At load time, extract the adapter's weight delta via ComfyUI's own
`calculate_weight` (applied to a zero weight, so the result is the pure ΔW), take a truncated
SVD with `torch.svd_lowrank`, and concatenate the factors onto the layer's existing
`svdq_l1`/`svdq_l2`. Runtime cost then drops to *zero* beyond a slightly wider branch, and the
delta is **truncated rather than quantized to 4 bits** — the same trade already accepted for
the base weight, and a much gentler one than `bake` makes.

**Open questions:** what rank the delta needs (a Kronecker product is not inherently
low-rank, so this may need measuring per adapter); ~15 s of SVD at load across 224 layers;
and whether the widened branch's VRAM is acceptable, given the branch is already 24% of a step
at rank 256. Wants a fidelity run against `bypass` before it could become the default.

## 4. Mixed precision: keep the first and last blocks at W8A8

**Status:** standard SVDQuant lever, never tried here. Costs file size, buys fidelity. No
estimate — nothing has been measured, which is exactly why it is on this list rather than
above the items that have been.

## 5. Benchmark the base checkpoint

`Krea2-Base-SVDQuant-W4A4-rank256-actaware` is published and has **never been through the
paired benchmark** (README says so). `tools/fidelity_bench.py --variant base` already
supports it. Not a quality lever — a claim with no evidence behind it.

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
