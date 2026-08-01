"""Sampler-only speed benchmark: this repo's W4A4 against int8 and bf16, with and without compile.

    python tools/speed_bench.py --server http://127.0.0.1:8188
    python tools/speed_bench.py --arms int8 r256aa --compile off on --out speed.json

Why this exists. Every speed number this repo has published is either *end to end* (which
BENCHMARKS.md measures at 10.4 s for int8 against 10.1 s for W4A4 -- a 3% gap that says
almost nothing about the kernels) or *per layer* (`svdquant_diag.py --mode bench`, which
times one Linear in isolation against bf16 and never sees int8 at all). Neither answers the
question users actually ask, which is "how much faster is a step".

Two things make that hard to measure through ComfyUI, and this script handles both:

* **Fixed cost swamps the signal.** A 1024x1024 8-step turbo render spends several seconds
  on CLIP text-encode, model staging and VAE decode. So instead of timing one render, every
  arm is run at *two* step counts and the per-step cost is the **slope**,
  `(t_hi - t_lo) / (steps_hi - steps_lo)`. Everything that does not scale with step count
  cancels out. The intercept is reported too, because a suspicious intercept is how you
  catch a run where the model was silently reloaded mid-measurement.
* **Cold runs are a different measurement.** The first render after switching checkpoints
  pays disk-to-VRAM load. Each arm therefore does one discarded warmup at each step count
  before anything is timed.

`--compile on` inserts a **TorchCompileModel** node (backend `inductor`) between the loader
and the sampler, which is what the report this script was written for was running. That arm
is the one that matters: on `int8-fast`'s own numbers compile is worth 1.58x to them
(1.64 -> 1.04 s/it), while this repo shields all 224 quantized linears from Dynamo
(`svdquant_w4a4.py:_shield_from_dynamo`) and gets far less. Measuring both sides with the
same harness is the point.

Needs a running ComfyUI on `--server`, and for the `int8` arm the `int8-fast` custom nodes
(node `OTUNetLoaderW8A8`) -- the checkpoint it loads is per-row int8 + convrot in that pack's
own format, not one ComfyUI's `UNETLoader` can read.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The graph is the same one fidelity_bench builds, minus the parts that would add variance:
# one fixed prompt, one fixed seed, no LoRA. Sharing the module keeps the sampler settings,
# CLIP and VAE names from drifting apart between the two benchmarks.
import fidelity_bench as fb  # noqa: E402

# One prompt, deliberately mid-length: text-encode cost is constant across arms and cancels
# in the slope anyway, but a pathologically short prompt would make the intercept unrealistic
# and hide a staging regression.
PROMPT = fb.PROMPTS_OBJECTS[9][1]

# Every render gets its own seed. ComfyUI caches node outputs by input hash, so submitting
# the same graph twice returns in ~20 ms without sampling anything -- the first version of
# this script measured `2.01 0.02 0.03` and would have reported a per-step cost of zero.
# Bumping the seed invalidates the KSampler and everything downstream while leaving the
# CLIP encode cached, which is what we want: text-encode is constant across arms anyway,
# and keeping it out of the timing makes the intercept easier to read.
SEED_BASE = 987654321
_seed_counter = 0

# Two step counts far enough apart that the difference is many times the timing noise, and
# low enough that the long arm still finishes in a sane time on a 3090.
STEPS_LO = 8
STEPS_HI = 24

# An arm is a checkpoint plus the loader node that can read it. `int8` is not a ComfyUI-native
# format: per-tensor `comfy_quant` blobs written by int8-fast, per-row scales, convrot baked
# into the weight. Its loader also insists on a `model_type`, which for Krea 2 is only used
# for on-the-fly quantization -- this file is already quantized, so the value is inert.
ARMS = {
    "bf16": {
        "node": "UNETLoader",
        "inputs": {"unet_name": "turbo.safetensors", "weight_dtype": "default"},
    },
    "int8": {
        "node": "OTUNetLoaderW8A8",
        "inputs": {"unet_name": "krea2turboint8convrot.safetensors",
                   "weight_dtype": "default", "model_type": "qwen",
                   "on_the_fly_quantization": False, "enable_convrot": True,
                   "lora_mode": "None"},
    },
    "nolowrank": {
        "node": "UNETLoader",
        "inputs": {"unet_name": "Krea2-Turbo-W4A4-noLowRank.safetensors",
                   "weight_dtype": "default"},
    },
    "r256aa": {
        "node": "Krea2SVDQuantW4A4Loader",
        "inputs": {"model_name": "Krea2-Turbo-SVDQuant-W4A4-rank256-actaware.safetensors"},
    },
    # The rank sweep exists to price the low-rank branch. `--act-stats` cannot change any of
    # these numbers -- same shapes, same kernels, only different values inside the factors --
    # so the plain builds stand in for their act-aware equivalents.
    "r16": {
        "node": "Krea2SVDQuantW4A4Loader",
        "inputs": {"model_name": "Krea2-Turbo-SVDQuant-W4A4-rank16.safetensors"},
    },
    "r64": {
        "node": "Krea2SVDQuantW4A4Loader",
        "inputs": {"model_name": "Krea2-Turbo-SVDQuant-W4A4-rank64.safetensors"},
    },
    "r128": {
        "node": "Krea2SVDQuantW4A4Loader",
        "inputs": {"model_name": "Krea2-Turbo-SVDQuant-W4A4-rank128.safetensors"},
    },
}

# bf16 is 24.5 GB and does not fit a 3090 alongside the text encoder, so it spends the run
# being shuttled over PCIe. It stays out of the default set: it is a reference point for the
# README table, not something to pay for on every run.
DEFAULT_ARMS = ["int8", "r256aa", "nolowrank"]


def build_graph(arm: str, steps: int, compile_backend: str | None, seed: int) -> dict:
    spec = ARMS[arm]
    g = {
        "1": {"class_type": spec["node"], "inputs": dict(spec["inputs"])},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": fb.CLIP_NAME, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": fb.VAE_NAME}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": fb.WIDTH, "height": fb.HEIGHT, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": fb.SAMPLER, "scheduler": fb.SCHEDULER,
                         "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
    }
    if compile_backend:
        # Note for whoever reads a FAILED row: a branchless w4a4 checkpoint arrives through
        # the stock UNETLoader, which never registers the kitchen kernel as a custom op, and
        # compiling one *raises* rather than running slowly. That is a real limitation of the
        # project, not of this script -- see TROUBLESHOOTING.md.
        g["11"] = {"class_type": "TorchCompileModel",
                   "inputs": {"model": ["1", 0], "backend": compile_backend}}
        g["7"]["inputs"]["model"] = ["11", 0]
    return g


def check_nodes(server: str, arms: list[str], want_compile: bool) -> list[str]:
    """Names of graph nodes this run needs that the server does not have.

    Checked up front rather than discovered on the first submit: a missing custom node comes
    back as a validation error against node id "1", which reads like a bad checkpoint path.
    """
    needed = {ARMS[a]["node"] for a in arms}
    if want_compile:
        needed.add("TorchCompileModel")
    try:
        info = fb._get(server, "/object_info")
    except Exception as exc:
        raise SystemExit("could not reach ComfyUI at {}: {}".format(server, exc))
    return sorted(n for n in needed if n not in info)


def _server_seconds(entry: dict) -> float | None:
    """Execution time as ComfyUI itself measured it, from the history entry's timestamps.

    Polling `/history` on a 2 s timer put up to 2 s of quantization noise on every sample --
    an 8.27 s render came back as 10.11 s. That bias mostly cancels in the slope, but its
    *variance* does not, and at a 15 s difference between the two step counts a couple of
    seconds of jitter is several percent on the answer. ComfyUI already records exactly what
    we want: `execution_start` and `execution_success` in milliseconds.
    """
    stamps = {}
    for name, payload in entry.get("status", {}).get("messages", []):
        if name in ("execution_start", "execution_success"):
            stamps[name] = payload.get("timestamp")
    start, end = stamps.get("execution_start"), stamps.get("execution_success")
    if start is None or end is None:
        return None
    return (end - start) / 1000.0


def time_run(server: str, arm: str, steps: int, compile_backend: str | None,
             timeout: int) -> float:
    global _seed_counter
    _seed_counter += 1
    graph = build_graph(arm, steps, compile_backend, SEED_BASE + _seed_counter)

    try:
        result = fb._post(server, "/prompt", {"prompt": graph})
    except urllib.error.HTTPError as exc:
        raise RuntimeError("{} @ {} steps: submit failed: {}".format(
            arm, steps, exc.read().decode("utf-8", "replace")[:400]))
    if "error" in result:
        raise RuntimeError("{} @ {} steps: {}".format(
            arm, steps, json.dumps(result["error"])[:400]))

    pid = result["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = fb._get(server, "/history/" + pid).get(pid)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("{} @ {} steps: {}".format(
                    arm, steps, json.dumps(status)[:400]))
            if status.get("completed") or status.get("status_str") == "success":
                seconds = _server_seconds(entry)
                if seconds is None:
                    raise RuntimeError(
                        "{} @ {} steps: history entry carries no execution timestamps; this "
                        "ComfyUI is too old for a per-step measurement".format(arm, steps))
                return seconds
        time.sleep(1)
    raise RuntimeError("{} @ {} steps: timeout".format(arm, steps))


def measure(server: str, arm: str, compile_backend: str | None, reps: int,
            timeout: int) -> dict:
    """Median wall time at two step counts, and the per-step slope between them.

    The warmup at each step count is what makes the numbers "warm": it pays the checkpoint
    load, the CLIP encode that ComfyUI then caches, and -- when compiling -- inductor's
    first-run compilation, which is tens of seconds and would otherwise land entirely in the
    first timed rep.
    """
    samples: dict[int, list[float]] = {}
    for steps in (STEPS_LO, STEPS_HI):
        time_run(server, arm, steps, compile_backend, timeout)  # warmup, discarded
        runs = [time_run(server, arm, steps, compile_backend, timeout) for _ in range(reps)]
        samples[steps] = runs
        print("    {:>2} steps: {}  median {:.2f}s".format(
            steps, " ".join("{:.2f}".format(r) for r in runs), statistics.median(runs)),
            flush=True)

    lo = statistics.median(samples[STEPS_LO])
    hi = statistics.median(samples[STEPS_HI])
    per_step = (hi - lo) / (STEPS_HI - STEPS_LO)
    return {
        "arm": arm,
        "compile": compile_backend or "off",
        "steps_lo": STEPS_LO, "steps_hi": STEPS_HI,
        "median_lo": lo, "median_hi": hi,
        "runs_lo": samples[STEPS_LO], "runs_hi": samples[STEPS_HI],
        "s_per_step": per_step,
        # Everything that does not scale with steps: text-encode, staging, VAE decode, queue.
        # It should be near-identical across arms of the same checkpoint size; a big outlier
        # means that arm was reloading the model between runs and its slope is not trustworthy.
        "fixed_overhead": lo - per_step * STEPS_LO,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=sorted(ARMS))
    ap.add_argument("--compile", nargs="+", default=["off", "on"],
                    choices=["off", "on", "cudagraphs"],
                    help="'on' is TorchCompileModel with backend inductor")
    ap.add_argument("--reps", type=int, default=3, help="timed runs per (arm, step count)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per render; the first compiled run pays inductor's compilation")
    ap.add_argument("--out", default=None, help="write the raw results here as JSON")
    args = ap.parse_args()

    backends = {"off": None, "on": "inductor", "cudagraphs": "cudagraphs"}
    missing = check_nodes(args.server, args.arms, any(c != "off" for c in args.compile))
    if missing:
        raise SystemExit("ComfyUI at {} has no node(s): {}. The int8 arm needs the int8-fast "
                         "custom nodes; drop it with --arms.".format(args.server,
                                                                     ", ".join(missing)))

    results = []
    for arm in args.arms:
        for mode in args.compile:
            print("{} / compile={}".format(arm, mode), flush=True)
            t0 = time.time()
            try:
                results.append(measure(args.server, arm, backends[mode], args.reps,
                                       args.timeout))
            except RuntimeError as exc:
                # One arm failing (a checkpoint that will not load, compile blowing up on a
                # kernel) must not throw away the arms that already ran.
                print("    SKIPPED: {}".format(exc), flush=True)
                results.append({"arm": arm, "compile": mode, "error": str(exc)})
                continue
            print("    -> {:.3f} s/step  (fixed {:.2f}s, {:.0f}s elapsed)".format(
                results[-1]["s_per_step"], results[-1]["fixed_overhead"], time.time() - t0),
                flush=True)

    ok = [r for r in results if "error" not in r]
    print("\n{:<12} {:>8} {:>10} {:>10} {:>9}".format(
        "arm", "compile", "s/step", "fixed s", "vs best"))
    best = min((r["s_per_step"] for r in ok), default=None)
    for r in results:
        if "error" in r:
            print("{:<12} {:>8} {:>10}".format(r["arm"], r["compile"], "FAILED"))
            continue
        print("{:<12} {:>8} {:>10.3f} {:>10.2f} {:>8.2f}x".format(
            r["arm"], r["compile"], r["s_per_step"], r["fixed_overhead"],
            r["s_per_step"] / best))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"resolution": [fb.WIDTH, fb.HEIGHT], "sampler": fb.SAMPLER,
                       "scheduler": fb.SCHEDULER, "prompt": PROMPT, "seed_base": SEED_BASE,
                       "results": results}, fh, indent=2)
        print("\nwrote {}".format(args.out))

    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
