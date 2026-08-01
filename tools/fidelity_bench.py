"""Multi-seed, paired fidelity benchmark against a BF16 reference -- with and without a LoRA.

    python tools/fidelity_bench.py generate --output-dir <ComfyUI/output> [--variant base] [--seeds 4]
    python tools/fidelity_bench.py score    --output-dir <ComfyUI/output>

Why this exists. The measurement it replaced scored one seed per prompt and reported each
checkpoint's *marginal* mean. Two things go wrong with that:

* **One seed cannot resolve the differences we care about.** Re-analysing its own 130-pair
  output as paired differences: `rank256 vs nolowrank` is a clean t=-4.73 (10/10 prompts),
  but `rank256 vs rank64` -- a *fourfold* rank increase -- is t=-1.29 and wins only 5 of 10
  prompts. Anything subtler than the branch itself is invisible at n=10, which is how a rank
  reallocation (`--rank-alloc gqa`, t=0.55) can look like a null result without that being
  provable either way.
* **Marginal means hide the effect behind prompt difficulty.** Prompt LPIPS ranges 0.23-0.42
  in this set; between-prompt spread swamps between-checkpoint spread. The same prompt scored
  under two checkpoints cancels that -- so this reports `mean(A_i - B_i)` over shared cells
  with its own standard error, never two independent means eyeballed side by side.

So: several seeds per prompt, paired statistics, and a reseed noise floor printed next to
every result. A difference smaller than the floor is a difference you cannot claim.

The LoRA arm exists because the repo's headline use case is unmeasured. `BENCHMARKS.md`
Test 2 scored LoRA output with a vision judge that had already saturated (71 of 110 cells a
flat 10/10 in the run that preceded it), and it never compared quantized+LoRA against
*BF16+LoRA* -- only against each other. The question that arm answers: does a LoRA amplify
quantization error, i.e. is LPIPS(quant+LoRA vs BF16+LoRA) worse than LPIPS(quant vs BF16)?
If it is, low-rank branch size matters more for LoRA users than the plain t2i sweep suggests.
Both arms therefore run the *same* prompts and seeds, so the two are directly subtractable.

Generation is idempotent: a cell whose PNG already exists is skipped, so the dataset can be
grown one `--seeds` or `--checkpoints` at a time without redoing work.

Needs `lpips` (`pip install lpips`) for the score stage; the generate stage needs only a
running ComfyUI on `--server`.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The metric implementations are shared rather than reimplemented -- two copies of SSIM in
# one repo is two chances to have a different window or a different epsilon.
import torch  # noqa: E402

import pixel_metrics  # noqa: E402

SUBDIR = "fb"
REFERENCE = "bf16"

# 8 steps / cfg 1.0 / euler+simple is what Krea 2 Turbo is distilled for; the base model is
# not distilled and wants 50 steps at cfg 3.5 (SAMPLER_HINTS in quantize_krea2.py). Changing
# either set changes what "the same image" means and invalidates comparison with earlier runs,
# so they are picked by `--variant` rather than edited.
STEPS = 8
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "simple"
WIDTH = 1024
HEIGHT = 1024

CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"
LORA_STRENGTH = 1.0

# An "arm" is one LoRA configuration, `base` being none. Each is scored against its *own*
# BF16 reference, because a LoRA changes how seed-sensitive the model is -- the reseed floor
# is 0.547 without one and 0.519 with -- so raw LPIPS is not comparable across arms and only
# the within-arm paired numbers mean anything.
#
# Both LoRAs here cover the same layers (448 block + 64 txtfusion keys) and differ in rank,
# which is the point: a second LoRA is what separates "this LoRA behaves like that" from
# "LoRAs behave like that". A txtfusion-only adapter would be useless here -- it never
# touches the 224 quantized blocks, so it cannot exercise the branch at all.
#
# The names must match what ComfyUI puts in the node's dropdown, which for a LoRA in a
# subfolder is a *native* relative path -- `Krea2\canon_krea2.safetensors` on Windows. A
# forward slash there fails validation with "value_not_in_list", so the subfolder is joined
# rather than written in. `LORA_SUBDIR = ""` for a flat `models/loras`.
LORA_SUBDIR = "Krea2"


def _lora(name: str) -> str:
    return os.path.join(LORA_SUBDIR, name) if LORA_SUBDIR else name


ARM_LORA = {
    "base": None,
    "lora": _lora("canon_krea2.safetensors"),                      # rank 16, photographic style
    "lora2": _lora("bloomgirls-ultrarealism-krea2_4k.safetensors"), # rank 32, realism style
    "lora3": _lora("lenovo_krea2.safetensors"),                    # rank 16
    "lora4": _lora("nicegirls_krea2.safetensors"),                 # rank 16
}

# Loader choice is a property of the file, so it lives with the file rather than being
# guessed from the name at call time. `quantized` drives the *LoRA* node choice: applying a
# LoRA to a quantized weight through the stock path means dequantize -> add -> requantize,
# which re-quantizes the LoRA delta to 4 bits, so quantized files go through this repo's node.
CHECKPOINTS_TURBO = {
    "bf16":      {"file": "turbo.safetensors",                            "loader": "unet", "quantized": False},
    "nolowrank": {"file": "Krea2-Turbo-W4A4-noLowRank.safetensors",       "loader": "unet", "quantized": True},
    "r16":       {"file": "Krea2-Turbo-SVDQuant-W4A4-rank16.safetensors", "loader": "svdq", "quantized": True},
    "r64":       {"file": "Krea2-Turbo-SVDQuant-W4A4-rank64.safetensors", "loader": "svdq", "quantized": True},
    "r128":      {"file": "Krea2-Turbo-SVDQuant-W4A4-rank128.safetensors","loader": "svdq", "quantized": True},
    "r256":      {"file": "Krea2-Turbo-SVDQuant-W4A4-rank256.safetensors","loader": "svdq", "quantized": True},
    # Same rank, same size, same kernel -- built with --act-stats so the low-rank split is
    # weighted by measured activation RMS instead of plain weight magnitude.
    "r256aa":    {"file": "Krea2-Turbo-SVDQuant-W4A4-rank256-actaware.safetensors", "loader": "svdq", "quantized": True},
    # Same build as r256aa except the activation statistics were captured with `lora2`
    # loaded. BENCHMARKS.md's open question is whether act-aware goes null under a LoRA
    # because the calibration saw no adapter; this arm is what answers it, and it is only
    # meaningful when scored on the `lora2` arm it was calibrated for.
    "r256aal2":  {"file": "Krea2-Turbo-SVDQuant-W4A4-rank256-actaware-lora2.safetensors", "loader": "svdq", "quantized": True},
}

CHECKPOINTS_BASE = {
    "bf16":      {"file": "raw.safetensors",                             "loader": "unet", "quantized": False},
    "nolowrank": {"file": "Krea2-Base-W4A4-convrot.safetensors",         "loader": "unet", "quantized": True},
    "r16":       {"file": "Krea2-Base-SVDQuant-W4A4-rank16.safetensors", "loader": "svdq", "quantized": True},
    "r64":       {"file": "Krea2-Base-SVDQuant-W4A4-rank64.safetensors", "loader": "svdq", "quantized": True},
    "r128":      {"file": "Krea2-Base-SVDQuant-W4A4-rank128.safetensors","loader": "svdq", "quantized": True},
    "r256":      {"file": "Krea2-Base-SVDQuant-W4A4-rank256.safetensors","loader": "svdq", "quantized": True},
    "r256aa":    {"file": "Krea2-Base-SVDQuant-W4A4-rank256-actaware.safetensors", "loader": "svdq", "quantized": True},
}

# Renders go to a per-variant subfolder so a base run cannot be scored against turbo images:
# the scorer indexes a directory, and two variants in one directory would silently pair
# 8-step turbo renders with 50-step base ones.
# The negative matters only above cfg 1.0. Turbo runs at cfg 1.0, where the negative branch
# is never evaluated, so a zeroed-out conditioning costs nothing there. The base model runs at
# cfg 3.5 and CFG then pushes *away* from whatever the negative is -- away from a zeroed
# conditioning means away from nothing, which comes out as grainy, over-contrasted garbage.
# `None` = ConditioningZeroOut; a string = a real encoded negative, as the shipped base
# workflow uses.
NEGATIVE_BASE = "blurry, low resolution, jpeg artifacts, watermark, deformed, extra limbs"

VARIANTS = {
    "turbo": {"steps": 8,  "cfg": 1.0, "subdir": "fb",      "checkpoints": CHECKPOINTS_TURBO,
              "negative": None},
    "base":  {"steps": 50, "cfg": 3.5, "subdir": "fb_base", "checkpoints": CHECKPOINTS_BASE,
              "negative": NEGATIVE_BASE},
}

CHECKPOINTS = CHECKPOINTS_TURBO
NEGATIVE = None


def select_variant(name: str) -> None:
    """Point the module's sampler settings and checkpoint table at one model variant."""
    global STEPS, CFG, SUBDIR, CHECKPOINTS, NEGATIVE
    spec = VARIANTS[name]
    STEPS, CFG = spec["steps"], spec["cfg"]
    SUBDIR, CHECKPOINTS = spec["subdir"], spec["checkpoints"]
    NEGATIVE = spec["negative"]


# The repo's established prompt set (the same ids BENCHMARKS.md uses), kept verbatim so new
# numbers stay comparable with the old ones.
PROMPTS_OBJECTS = [
    ("01_dense_text", "A rain-soaked neon diner sign at night, below it a handwritten chalkboard menu with three lines of text reading 'SOUP $4 / PIE $6 / COFFEE $2', reflections on wet asphalt, cinematic"),
    ("02_curved_text", "Close-up of a person holding a paper coffee cup with large bold curved text 'STAY WARM' printed around the cup, soft morning light, shallow depth of field"),
    ("03_hands_detail", "A violinist's hands mid-performance, fingers pressed on the strings, bow in motion with visible blur, studio lighting, extreme close-up, photorealistic"),
    ("04_crowd_faces", "A busy Tokyo street crossing at dusk, dozens of pedestrians with distinct faces and expressions, neon signage in the background, wide angle, high detail"),
    ("05_symmetry_pattern", "A perfectly symmetrical Islamic geometric tile mosaic, intricate repeating star and polygon pattern, deep blue and gold, overhead flat lighting, ultra sharp"),
    ("06_multi_subject", "Two chefs in white uniforms plating a dish together in a busy kitchen, one holding tweezers placing a garnish, the other pouring sauce, steam rising, low angle shot"),
    ("07_reflections_glass", "A glass of iced whiskey on a dark wood bar, condensation droplets, warm bokeh lights reflected in the glass and the liquid, macro photography"),
    ("08_logo_typography", "A vintage motorcycle fuel tank with a hand-painted logo reading 'IRON WOLF GARAGE' in bold serif letters, chrome and scratched paint texture, studio product shot"),
    ("09_counting_objects", "A wooden table from directly above with exactly seven red apples arranged in a neat row next to three green pears, soft natural light, flat lay photography"),
    ("10_complex_scene", "A fantasy marketplace street at golden hour, merchant stalls with hanging fabrics and baskets of spices, a dragon perched on a rooftop in the background, dense crowd, painterly digital art"),
]

# People, deliberately. The set above is mostly objects and scenes, which under-tests the
# kind of LoRA most people actually install: `canon_krea2` is a photographic style adapter,
# and a style adapter shows its hand on skin, hair and fabric far more than on a tile mosaic.
# Faces are also the failure mode viewers are most sensitive to, so a quantization difference
# that is invisible on a marketplace scene can be obvious on a portrait.
PROMPTS_PEOPLE = [
    ("11_portrait_woman", "Close-up portrait of a woman with freckles and green eyes, windswept auburn hair, wearing a chunky knitted wool sweater, standing on a rainy city street at blue hour, shallow depth of field, natural skin texture with visible pores, catchlights in both eyes, 85mm lens, photorealistic"),
    ("12_person_holding_sign", "A young woman in a bright yellow raincoat standing at the end of a wooden pier, holding up a handwritten cardboard sign that reads 'BACK IN 5 MIN' in thick black marker, seagulls circling behind her, overcast diffused light, full body shot, 35mm documentary photography"),
    ("13_two_people_interaction", "A barista with tattooed forearms handing a paper cup across a marble counter to a bearded man in a denim jacket, both mid-conversation and smiling, morning light through a tall window, steam rising from the cup, candid documentary photography, natural skin tones"),
    ("14_hands_and_face", "A woman applying red lipstick while looking into a small round handheld mirror, her fingers wrapped around the mirror rim, the reflection of one eye visible in the glass, warm bathroom lighting, extreme close-up on face and hands, photorealistic detail"),
    ("15_full_body_fashion", "Full body editorial fashion photograph of a woman in a flowing emerald green silk dress mid-stride on a marble staircase, one hand trailing on the brass railing, dramatic hard side lighting, sharp fabric texture and folds, 50mm, high fashion magazine style"),
    ("16_group_selfie", "Four friends of different ethnicities crowded around a restaurant table taking a group selfie, one holding the phone at arm's length, another leaning in making a peace sign, plates of pasta and half-full wine glasses on the table, warm indoor lighting, natural candid expressions"),
]

PROMPTS = PROMPTS_OBJECTS + PROMPTS_PEOPLE

# Fixed rather than random so a re-run reproduces the same dataset. The first is the seed the
# older single-seed benchmarks used, which keeps those results inside this one.
SEEDS = [987654321, 424242424, 12345, 777]


# --------------------------------------------------------------------------- generation

def cell_name(arm: str, ckpt: str, prompt_id: str, seed: int) -> str:
    """Filename stem for one cell. `__` separates fields because prompt ids contain `_`."""
    return "{}__{}__{}__s{}".format(arm, ckpt, prompt_id, seed)


def build_graph(ckpt: str, prompt: str, seed: int, arm: str, prefix: str) -> dict:
    spec = CHECKPOINTS[ckpt]
    if spec["loader"] == "svdq":
        loader = {"class_type": "Krea2SVDQuantW4A4Loader",
                  "inputs": {"model_name": spec["file"]}}
    else:
        loader = {"class_type": "UNETLoader",
                  "inputs": {"unet_name": spec["file"], "weight_dtype": "default"}}

    g = {
        "1": loader,
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP_NAME, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": ({"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
              if NEGATIVE is None else
              {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}}),
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER,
                         "scheduler": SCHEDULER, "denoise": 1.0,
                         "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }

    lora = ARM_LORA[arm]
    if lora:
        # A quantized checkpoint needs this repo's LoRA node whether or not it carries a
        # low-rank branch -- `nolowrank` included. ComfyUI's own loader would report success
        # and patch nothing on the 224 quantized blocks.
        if spec["quantized"]:
            g["10"] = {"class_type": "Krea2SVDQuantLoraLoader",
                       "inputs": {"model": ["1", 0], "lora_name": lora,
                                  "strength": LORA_STRENGTH}}
        else:
            g["10"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": ["1", 0], "lora_name": lora,
                                  "strength_model": LORA_STRENGTH}}
        g["7"]["inputs"]["model"] = ["10", 0]

    return g


def _post(server: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(server + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _get(server: str, path: str) -> dict:
    with urllib.request.urlopen(server + path, timeout=60) as resp:
        return json.loads(resp.read())


def run_cell(server: str, graph: dict, timeout: int) -> tuple[bool, float, str]:
    t0 = time.time()
    try:
        result = _post(server, "/prompt", {"prompt": graph})
    except urllib.error.HTTPError as exc:
        return False, 0.0, "submit failed: {}".format(exc.read().decode("utf-8", "replace")[:400])
    if "error" in result:
        return False, 0.0, json.dumps(result["error"])[:400]
    pid = result["prompt_id"]
    while time.time() - t0 < timeout:
        hist = _get(server, "/history/" + pid)
        entry = hist.get(pid)
        if entry:
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                return True, time.time() - t0, ""
            if status.get("status_str") == "error":
                return False, time.time() - t0, json.dumps(status)[:400]
        time.sleep(2)
    return False, time.time() - t0, "timeout"


def generate(args) -> int:
    out_root = os.path.join(args.output_dir, SUBDIR)
    os.makedirs(out_root, exist_ok=True)
    seeds = SEEDS[: args.seeds]
    if args.prompt_ids:
        wanted = set(args.prompt_ids)
        prompts = [p for p in PROMPTS if p[0] in wanted]
    else:
        prompts = PROMPTS[: args.prompts]

    todo = []
    for arm, ckpt, (pid, text), seed in itertools.product(
            args.arms, args.checkpoints, prompts, seeds):
        stem = cell_name(arm, ckpt, pid, seed)
        # ComfyUI's SaveImage bumps the counter when a name is taken, so an existing file
        # would silently become _00002_ and never be found by the scorer. Skipping is both
        # the resume mechanism and the guard against that.
        if os.path.exists(os.path.join(out_root, stem + "_00001_.png")):
            continue
        todo.append((arm, ckpt, pid, text, seed, stem))

    print("{} cells to generate ({} already present)".format(
        len(todo), len(args.arms) * len(args.checkpoints) * len(prompts) * len(seeds) - len(todo)),
        flush=True)

    failures = []
    for i, (arm, ckpt, pid, text, seed, stem) in enumerate(todo, start=1):
        graph = build_graph(ckpt, text, seed, arm, "{}/{}".format(SUBDIR, stem))
        ok, elapsed, err = run_cell(args.server, graph, args.timeout)
        print("[{}/{}] {} {:.1f}s{}".format(i, len(todo), stem, elapsed,
                                            "" if ok else "  FAILED: " + err), flush=True)
        if not ok:
            failures.append((stem, err))

    if failures:
        print("\n{} cells failed:".format(len(failures)))
        for stem, err in failures:
            print("  {}: {}".format(stem, err))
        return 1
    return 0


# --------------------------------------------------------------------------- scoring

def index(out_root: str) -> dict[tuple[str, str], dict[tuple[str, int], str]]:
    """{(arm, checkpoint): {(prompt_id, seed): path}} from the files actually on disk."""
    found: dict[tuple[str, str], dict[tuple[str, int], str]] = {}
    if not os.path.isdir(out_root):
        return found
    for name in sorted(os.listdir(out_root)):
        if not name.endswith("_00001_.png"):
            continue
        stem = name[: -len("_00001_.png")]
        parts = stem.split("__")
        if len(parts) != 4 or not parts[3].startswith("s"):
            continue
        arm, ckpt, pid, seed_s = parts
        try:
            seed = int(seed_s[1:])
        except ValueError:
            continue
        found.setdefault((arm, ckpt), {})[(pid, seed)] = os.path.join(out_root, name)
    return found


def paired(a_scores: dict, b_scores: dict) -> dict | None:
    """Paired difference A-B over the cells both have. Negative means A is closer to BF16."""
    cells = sorted(set(a_scores) & set(b_scores))
    if len(cells) < 2:
        return None
    diffs = [a_scores[c] - b_scores[c] for c in cells]
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(len(diffs))
    return {"mean": mean, "sd": sd, "se": se,
            "t": mean / se if se else float("inf"),
            "a_wins": sum(1 for d in diffs if d < 0), "n": len(diffs)}


def score(args) -> int:
    try:
        import lpips
    except ImportError:
        raise SystemExit("pip install lpips")

    out_root = os.path.join(args.output_dir, SUBDIR)
    images = index(out_root)
    if not images:
        raise SystemExit("no fidelity_bench renders in {}; run the generate stage first"
                         .format(out_root))

    # A partially-finished generate leaves some cells with more seeds than others, which
    # would weight the reseed floor unevenly across prompts. Restricting to a seed prefix
    # keeps the analysed subset rectangular while the dataset keeps growing.
    if args.seeds:
        keep = set(SEEDS[: args.seeds])
        images = {k: {c: p for c, p in cells.items() if c[1] in keep}
                  for k, cells in images.items()}
        images = {k: cells for k, cells in images.items() if cells}

    # The two prompt groups answer different questions -- a photographic style LoRA barely
    # touches a tile mosaic but rewrites skin and hair -- so they are separable.
    if args.group:
        wanted = {p for p, _ in (PROMPTS_PEOPLE if args.group == "people"
                                 else PROMPTS_OBJECTS)}
        images = {k: {c: p for c, p in cells.items() if c[0] in wanted}
                  for k, cells in images.items()}
        images = {k: cells for k, cells in images.items() if cells}

    device = torch.device(args.device)
    net = lpips.LPIPS(net="alex").to(device).eval()

    # Only the reference renders are cached. They are read once per candidate checkpoint and
    # again for every reseed pair, so caching them is most of the win; candidates are read
    # exactly once. Caching everything costs ~12 MB of VRAM per 1024x1024 image, which on a
    # full 400-cell run is ~5 GB sitting next to LPIPS for no benefit.
    ref_paths = {p for (_, ck), cells in images.items() if ck == REFERENCE
                 for p in cells.values()}
    cache: dict[str, "torch.Tensor"] = {}

    def load(path):
        if path in cache:
            return cache[path]
        tensor = pixel_metrics.load_image(path, device)
        if path in ref_paths:
            cache[path] = tensor
        return tensor

    rows = []
    # {(arm, ckpt): {(prompt, seed): lpips}} -- the shape `paired` consumes.
    lpips_by = {}
    with torch.no_grad():
        for (arm, ckpt), cells in sorted(images.items()):
            if ckpt == REFERENCE:
                continue
            refs = images.get((arm, REFERENCE))
            if not refs:
                print("skipping arm {!r}: no {} reference renders".format(arm, REFERENCE))
                continue
            for cell, path in sorted(cells.items()):
                ref_path = refs.get(cell)
                if ref_path is None:
                    continue
                a, b = load(ref_path), load(path)
                if a.shape != b.shape:
                    print("  skipping {} {} {}: shape mismatch".format(arm, ckpt, cell))
                    continue
                value = net(a, b).item()
                rows.append({"arm": arm, "checkpoint": ckpt, "prompt_id": cell[0],
                             "seed": cell[1], "lpips": round(value, 5),
                             "psnr": round(pixel_metrics.psnr(a, b), 3),
                             "ssim": round(pixel_metrics.ssim(a, b), 5)})
                lpips_by.setdefault((arm, ckpt), {})[cell] = value

        # Reseed floor: the same checkpoint at two seeds on the same prompt. Anything below
        # this is a difference the sampler produces on its own.
        floors = {}
        for arm in sorted({a for a, _ in images}):
            refs = images.get((arm, REFERENCE), {})
            per_prompt = {}
            for (pid, seed), path in refs.items():
                per_prompt.setdefault(pid, []).append(path)
            dists = []
            for pid, paths in per_prompt.items():
                for p, q in itertools.combinations(sorted(paths), 2):
                    dists.append(net(load(p), load(q)).item())
            if dists:
                floors[arm] = {"mean": statistics.mean(dists),
                               "sd": statistics.stdev(dists) if len(dists) > 1 else 0.0,
                               "n": len(dists)}

        # Sanity: the instrument must return 0 comparing an image with itself.
        any_path = next(iter(next(iter(images.values())).values()))
        self_lpips = net(load(any_path), load(any_path).clone()).item()

    out_csv = args.out or os.path.join(out_root, "fidelity_bench.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "checkpoint", "prompt_id", "seed",
                                           "lpips", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)

    print("\nsanity: reference vs itself LPIPS = {:.6f} (must be ~0)".format(self_lpips))
    for arm, f in sorted(floors.items()):
        print("reseed floor [{}]: LPIPS {:.4f} +-{:.4f} over {} same-prompt seed pairs"
              .format(arm, f["mean"], f["sd"], f["n"]))

    print("\n{:<6} {:<12} {:>8} {:>8} {:>8} {:>5}".format(
        "arm", "checkpoint", "LPIPS", "PSNR", "SSIM", "n"))
    for (arm, ckpt) in sorted(lpips_by):
        sub = [r for r in rows if r["arm"] == arm and r["checkpoint"] == ckpt]
        print("{:<6} {:<12} {:>8.4f} {:>8.2f} {:>8.4f} {:>5}".format(
            arm, ckpt, statistics.mean(r["lpips"] for r in sub),
            statistics.mean(r["psnr"] for r in sub),
            statistics.mean(r["ssim"] for r in sub), len(sub)))

    print("\npaired differences within each arm (negative = first is closer to BF16)")
    print("{:<40} {:>9} {:>8} {:>7} {:>9}".format("A vs B", "mean", "se", "t", "A wins"))
    summary = []
    for arm in sorted({a for a, _ in lpips_by}):
        ckpts = sorted(c for a, c in lpips_by if a == arm)
        for a_ck, b_ck in itertools.combinations(ckpts, 2):
            st = paired(lpips_by[(arm, a_ck)], lpips_by[(arm, b_ck)])
            if st is None:
                continue
            mark = "  ***" if abs(st["t"]) > 3 else ("  *" if abs(st["t"]) > 2 else "")
            label = "[{}] {} vs {}".format(arm, a_ck, b_ck)
            print("{:<40} {:>+9.4f} {:>8.4f} {:>7.2f} {:>5}/{}{}".format(
                label, st["mean"], st["se"], st["t"], st["a_wins"], st["n"], mark))
            summary.append(dict(arm=arm, a=a_ck, b=b_ck, **st))

    # The reason the LoRA arms exist: same checkpoint, same cells, LoRA vs no LoRA. A
    # positive mean means the LoRA made that checkpoint drift further from its own
    # reference, i.e. LoRAs amplify quantization error. One block per LoRA arm present --
    # this used to be hardcoded to "lora", so extra arms were generated and then silently
    # left out of the analysis.
    base_ckpts = {c for a, c in lpips_by if a == "base"}
    for arm in sorted({a for a, _ in lpips_by} - {"base"}):
        both = sorted({c for a, c in lpips_by if a == arm} & base_ckpts)
        if not both:
            continue
        print("\nLoRA amplification ({} - base, same checkpoint/prompt/seed)".format(arm))
        print("{:<40} {:>9} {:>8} {:>7} {:>9}".format("checkpoint", "mean", "se", "t", "worse"))
        for ckpt in both:
            st = paired(lpips_by[(arm, ckpt)], lpips_by[("base", ckpt)])
            if st is None:
                continue
            mark = "  ***" if abs(st["t"]) > 3 else ("  *" if abs(st["t"]) > 2 else "")
            print("{:<40} {:>+9.4f} {:>8.4f} {:>7.2f} {:>5}/{}{}".format(
                ckpt, st["mean"], st["se"], st["t"], st["n"] - st["a_wins"], st["n"], mark))
            summary.append(dict(arm="amplification:" + arm, a=ckpt, b=ckpt, **st))

    json_path = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"reference": REFERENCE, "variant": args.variant,
                   "reference_self_lpips": self_lpips,
                   "reseed_floor": floors, "paired": summary,
                   "settings": {"steps": STEPS, "cfg": CFG, "negative": NEGATIVE,
                                "sampler": SAMPLER,
                                "scheduler": SCHEDULER, "width": WIDTH, "height": HEIGHT,
                                "loras": ARM_LORA, "lora_strength": LORA_STRENGTH}},
                  fh, indent=2)
    print("\nwrote {}\n      {}".format(out_csv, json_path))
    return 0


# --------------------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    for name in ("generate", "score"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", required=True,
                       help="the ComfyUI output directory (renders land in a subfolder of it)")
        p.add_argument("--variant", default="turbo", choices=sorted(VARIANTS),
                       help="which model family to bench: turbo (8 steps, cfg 1.0) or base "
                            "(50 steps, cfg 3.5). Must match between generate and score")

    gen = sub.choices["generate"]
    gen.add_argument("--server", default="http://127.0.0.1:8188")
    gen.add_argument("--seeds", type=int, default=len(SEEDS),
                     help="how many of the fixed seeds to use (max {})".format(len(SEEDS)))
    gen.add_argument("--prompts", type=int, default=len(PROMPTS),
                     help="how many of the fixed prompts to use (max {})".format(len(PROMPTS)))
    gen.add_argument("--prompt-ids", nargs="+", default=None,
                     metavar="ID", choices=[p for p, _ in PROMPTS],
                     help="explicit prompt ids, overriding --prompts. Use this rather than a "
                          "count when the set has grown: a count silently changes meaning "
                          "when prompts are appended")
    gen.add_argument("--arms", nargs="+", default=["base"], choices=sorted(ARM_LORA))
    # No `choices=` here: the legal set depends on --variant, which argparse has not read
    # yet. Validated against the selected variant below instead.
    gen.add_argument("--checkpoints", nargs="+", default=None)
    gen.add_argument("--timeout", type=int, default=900)

    sc = sub.choices["score"]
    sc.add_argument("--out", default=None)
    # Scoring is LPIPS over already-rendered PNGs -- it has no GPU requirement of its own,
    # so it must not fail on a machine that only has the images.
    sc.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sc.add_argument("--seeds", type=int, default=None,
                    help="analyse only the first N fixed seeds, so a half-finished generate "
                         "does not make the cell grid ragged")
    sc.add_argument("--group", default=None, choices=["objects", "people"],
                    help="restrict to one prompt group (01-10 vs 11-16)")

    args = ap.parse_args()
    select_variant(args.variant)
    if args.stage == "generate":
        if args.checkpoints is None:
            args.checkpoints = sorted(CHECKPOINTS)
        unknown = [c for c in args.checkpoints if c not in CHECKPOINTS]
        if unknown:
            raise SystemExit("unknown checkpoint(s) for variant {}: {} (have: {})".format(
                args.variant, ", ".join(unknown), ", ".join(sorted(CHECKPOINTS))))
        if REFERENCE not in args.checkpoints:
            print("note: {!r} is not in --checkpoints, so nothing will be scorable until it "
                  "is generated".format(REFERENCE))
        return generate(args)
    return score(args)


if __name__ == "__main__":
    sys.exit(main())
