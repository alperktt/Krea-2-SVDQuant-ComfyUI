"""Generate the UI-format workflow JSONs under ``workflows/``.

Run this, do not hand-edit the output:

    python tools/build_workflows.py

ComfyUI has two JSON dialects and they are not interchangeable.

* **API format** -- ``{"1": {"class_type": ..., "inputs": {...}}, ...}``. What you POST to
  ``/prompt``. Carries no layout, no titles, no notes, no colours.
* **UI format** -- ``{"nodes": [...], "links": [...], ...}``. What the editor saves and what
  drag-and-drop expects.

The repo used to ship only API format while the README told people to drag the file in, so
every new user's first experience was a pile of untitled nodes stacked on the origin. Both
dialects come out of this script now -- ``*_api.json`` for scripting, the plain name for the
editor -- from one graph definition, so they cannot drift apart. They previously could, and
did: the committed API files carried titles the generator had never produced.

Positions are computed from a column/row grid rather than written by hand, so adding a node
does not mean renumbering everything below it.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "workflows")

# Widget order matters: ComfyUI matches `widgets_values` positionally against the node's
# required inputs, so these lists follow each node class's INPUT_TYPES order exactly.
COL = 420          # horizontal gap between columns
ROW = 200          # vertical gap between rows

TEAL = ("#233", "#355")        # loaders
GREEN = ("#232", "#353")       # prompts
PURPLE = ("#323", "#535")      # sampling
YELLOW = ("#432", "#653")      # notes / read me first / diagnostics
ORANGE = ("#653", "#874")      # LoRA

# Mirrors svdquant_lora.ADAPTER_BYPASS. Not imported: this script must run without ComfyUI
# on sys.path, and the loader module imports comfy at module level. Asserted in main().
ADAPTER_BYPASS = "bypass (exact, slower)"


class Graph:
    """Minimal UI-format graph builder.

    Only what these workflows need: typed slots, links, groups, colours. Not a general
    LiteGraph serializer.
    """

    def __init__(self):
        self.nodes = []
        self.links = []
        self._next_node = 1
        self._next_link = 1
        self._slots = {}   # node_id -> {"in": {name: idx}, "out": {name: idx}}
        self._api = {}     # node_id -> what the API dialect needs about that node

    def add(self, class_type, col, row, inputs=None, outputs=None, widgets=None,
            title=None, colour=None, size=None, api=True, mode=0, widget_inputs=()):
        """`widgets` is a list of (input_name, value).

        The name is what the API dialect needs and the UI dialect ignores; `None` marks a
        UI-only widget (`control_after_generate` has no API input). Keeping them as pairs is
        what lets one graph definition emit both dialects, instead of a hand-written API file
        drifting away from the generator -- which is exactly what had happened.

        `api=False` drops the node from the API output: Notes and the optional Diagnostics
        node are editor furniture with nothing to execute.

        `mode=4` mutes the node in the editor. A muted node must also be `api=False` -- the
        API dialect has no mute flag, so leaving it in would execute something the editor
        shows as off.

        `widget_inputs` names inputs that are widgets on the node class but are being wired
        from another node instead. LiteGraph needs the `widget` key on the slot or the editor
        draws a plain input the frontend then refuses to reconnect to its widget.
        """
        if mode == 4 and api:
            raise ValueError("a muted node must be api=False: {}".format(class_type))
        nid = self._next_node
        self._next_node += 1
        inputs = inputs or []
        outputs = outputs or []
        widgets = widgets or []
        self._slots[nid] = {
            "in": {name: i for i, (name, _) in enumerate(inputs)},
            "out": {name: i for i, (name, _) in enumerate(outputs)},
        }
        node = {
            "id": nid,
            "type": class_type,
            "pos": [60 + col * COL, 60 + row * ROW],
            "size": size or [380, 100],
            "flags": {},
            "order": self._next_node - 2,
            "mode": mode,
            "inputs": [dict({"name": n, "type": t, "link": None},
                            **({"widget": {"name": n}} if n in widget_inputs else {}))
                       for n, t in inputs],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(outputs)],
            "properties": {"Node name for S&R": class_type},
            "widgets_values": [v for _name, v in widgets],
        }
        if title:
            node["title"] = title
        if colour:
            node["color"], node["bgcolor"] = colour
        self.nodes.append(node)
        self._api[nid] = {"class_type": class_type, "title": title, "api": api,
                          "widgets": [(n, v) for n, v in widgets if n is not None]}
        return nid

    def link(self, src_node, src_name, dst_node, dst_name):
        src_slot = self._slots[src_node]["out"][src_name]
        dst_slot = self._slots[dst_node]["in"][dst_name]
        link_type = next(n for n in self.nodes if n["id"] == src_node)["outputs"][src_slot]["type"]
        lid = self._next_link
        self._next_link += 1
        self.links.append([lid, src_node, src_slot, dst_node, dst_slot, link_type])
        for n in self.nodes:
            if n["id"] == src_node:
                n["outputs"][src_slot]["links"].append(lid)
            elif n["id"] == dst_node:
                n["inputs"][dst_slot]["link"] = lid
        return lid

    def note(self, text, col, row, size=(400, 260), title="READ ME"):
        """A built-in Note node. Its text lives in widgets_values[0]."""
        return self.add("Note", col, row, widgets=[(None, text)], title=title, colour=YELLOW,
                        size=list(size), api=False)

    def group(self, title, bounding, colour="#3f789e"):
        return {"title": title, "bounding": list(bounding), "color": colour,
                "font_size": 24, "flags": {}}

    def serialize(self, groups=()):
        return {
            "id": "00000000-0000-4000-8000-000000000000",
            "revision": 0,
            "last_node_id": self._next_node - 1,
            "last_link_id": self._next_link - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": list(groups),
            "config": {},
            "extra": {},
            "version": 0.4,
        }

    def serialize_api(self):
        """The same graph in API dialect: {"1": {"class_type", "inputs", "_meta"}, ...}.

        Ids are reassigned sequentially over the executable nodes, so dropping the Note (which
        is always added first) leaves the remaining nodes numbered from 1 with no gaps.
        """
        keep = [n["id"] for n in self.nodes if self._api[n["id"]]["api"]]
        renum = {old: str(i + 1) for i, old in enumerate(keep)}

        # dst_node -> {input_name: [src_id, src_slot]}
        wired: dict[int, dict] = {}
        for _lid, src, src_slot, dst, dst_slot, _type in self.links:
            if src not in renum or dst not in renum:
                continue
            name = next(n for n, i in self._slots[dst]["in"].items() if i == dst_slot)
            wired.setdefault(dst, {})[name] = [renum[src], src_slot]

        out = {}
        for old in keep:
            meta = self._api[old]
            inputs = {name: value for name, value in meta["widgets"]}
            inputs.update(wired.get(old, {}))
            entry = {"class_type": meta["class_type"], "inputs": inputs}
            if meta["title"]:
                entry["_meta"] = {"title": meta["title"]}
            out[renum[old]] = entry
        return out


TURBO_NOTE = """KREA 2 TURBO - SVDQuant W4A4

1. Checkpoint goes in ComfyUI/models/diffusion_models/.
   This graph expects an svdq checkpoint (it carries *.svdq_l1 /
   *.svdq_l2 tensors). The --format w4a4 / int8 / fp8 files have no
   low-rank branch and load with the stock UNETLoader instead.

2. cfg MUST be 1.0. Krea 2 Turbo is cfg-distilled, so the negative
   prompt is zeroed out (ConditioningZeroOut) rather than encoded.
   Raising cfg here degrades the image, it does not sharpen it.
   8 steps is what the checkpoint was distilled for.

3. Text encoder: any Qwen3-VL 4B in ComfyUI/models/text_encoders/,
   loaded with CLIPLoader type "krea2".
   VAE: qwen_image_vae.safetensors.

4. SLOW? Read the loader's "status" output. It names the kernel that
   will actually run. If it does not say cuda, ComfyUI has disabled
   comfy_kitchen's CUDA backend because torch was built against
   CUDA < 13 -- and the fallback dequantizes int4 in Python, so the
   checkpoint ends up slower than fp8. Install a cu130+ torch build.
   The Krea2 SVDQuant Env Check node answers this with no model loaded.

5. LoRA: use the Krea2 SVDQuant LoRA Loader, not the stock one. The
   stock loader cannot patch a quantized weight and skips the blocks."""

BASE_NOTE = """KREA 2 BASE (non-turbo) - SVDQuant W4A4

The loader node will be outlined in RED when you open this. That is
expected: the filename it wants does not exist until you build it in
step 1, and ComfyUI flags any dropdown value it cannot find. Build the
checkpoint, then reselect it in the loader.

1. There is no pre-built base checkpoint to download. Make one:

     python quantize_krea2.py raw.safetensors \\
         --format svdq --rank 64 --variant base

   That writes Krea2-Base-SVDQuant-W4A4-rank64.safetensors next to
   the source. ~54s without refinement, ~5.7min with it.
   The Krea2 SVDQuant Quantize node does the same thing from inside
   ComfyUI if you would rather not touch a terminal.

2. Unlike Turbo, this model is NOT cfg-distilled, so it uses real
   classifier-free guidance: cfg 3.5 and a real negative prompt
   (a second CLIPTextEncode, not ConditioningZeroOut).
   ~50 steps. These are starting points, tune them.

3. This is roughly 12x the sampling work of the 8-step Turbo graph.
   Every benchmark number in the README is Turbo at 8 steps - the
   ratios carry over, the absolute seconds do not.

4. Text encoder and VAE are the same as the Turbo graph.
   Same slow-generation checklist too: read the loader's status
   output first."""

LORA_NOTE = """KREA 2 TURBO + LoRA - SVDQuant W4A4

Same graph as the plain Turbo one, with LoRA loaders inserted between
the checkpoint loader and the sampler.

1. USE THIS NODE, NOT THE STOCK ONE. On a quantized checkpoint the
   stock LoraLoaderModelOnly has to dequantize the 4-bit weight, add
   the LoRA, and requantize it -- which throws away the format and
   quantizes the LoRA delta to 4 bits along with it. This node
   attaches the LoRA as a parallel branch instead:
   (W + BA)x == Wx + B(Ax), so the quantized weight is never touched.

2. CHECK THE CONSOLE. The node prints what it matched, e.g.
     224 quantized layers, 32 normal layers
   If it says 0 quantized layers, the LoRA does not target this model
   and nothing will change. That line is the first thing to look at
   when a LoRA "does nothing".

3. STACKING. Chain more of these nodes, one per LoRA. The whole stack
   is refolded each time, so strengths stay exact no matter how many
   you chain, and the cost stays one pair of GEMMs. The second loader
   here is bypassed (ctrl-B) -- unmute it to use it.

4. RANK MATTERS HERE. Without a LoRA, branch ranks 64/128/256 measure
   the same. With one, rank 256 wins clearly and rank 64 loses most of
   its advantage. If you use LoRAs, download rank 256.

5. adapters INPUT. Only affects LoKr / LoHa / OFT, which cannot fold
   into the branch. "bypass" is exact and costs ~1.8s per model call
   at 1440x1920; "bake" is free per-step but requantizes the delta.
   A plain up/down LoRA ignores this setting entirely."""

DIAG_NOTE = """KREA 2 SVDQUANT - DIAGNOSTICS

Run this before reporting a bug, and before downloading 8 GB.
Nothing here generates an image.

ENV CHECK (left, no model needed)
   Answers the single most common report: "the quantized checkpoint
   is slower than fp8". Almost always the cause is that ComfyUI
   disabled comfy_kitchen's CUDA backend because torch was built
   against CUDA < 13, and the fallback dequantizes int4 in Python --
   slower than bf16, not faster. Needs no model, so run it first.

DIAGNOSTICS (right, needs a checkpoint) - the mode widget:
   dispatch  which kernel actually runs. START HERE.
   env       versions + memory accounting, including where the
             low-rank factors currently live. Under lowvram they
             should read cpu between steps, not cuda.
   bench     quantized vs bf16 timing per layer.
   profile   torch.profiler table.
   compile   graph breaks a TorchCompileModel run would pay.

   tokens = sequence length. 4096 is 1024x1024. Kernel selection is
   shape-dependent, so match your real resolution.

WHAT TO PASTE INTO AN ISSUE
   Your GPU, the checkpoint filename, and the report output of both
   nodes. Both are OUTPUT_NODEs, so the text shows in the node itself
   and in the console. Neither is cached -- they re-measure the live
   process every run, which is the point."""

PROMPT = ("A cluttered antique clockmaker's workshop seen through a cracked magnifying "
          "glass, brass gears laid out in a spiral on the workbench, a ginger cat asleep "
          "on a stack of leather-bound books, warm afternoon light with visible dust "
          "motes, photorealistic, 85mm lens, shallow depth of field")
NEGATIVE = "blurry, low resolution, jpeg artifacts, watermark, deformed, extra limbs"


def build_turbo():
    g = Graph()
    g.note(TURBO_NOTE, 0, 0, size=(400, 620), title="READ ME FIRST")

    loader = g.add("Krea2SVDQuantW4A4Loader", 1, 0,
                   outputs=[("MODEL", "MODEL"), ("STRING", "STRING")],
                   widgets=[("model_name", "Krea2-Turbo-SVDQuant-W4A4-rank64.safetensors")],
                   title="Krea2 SVDQuant W4A4 Loader", colour=TEAL, size=[400, 120])
    clip = g.add("CLIPLoader", 1, 1,
                 outputs=[("CLIP", "CLIP")],
                 widgets=[("clip_name", "qwen3vl_4b_fp8_scaled.safetensors"), ("type", "krea2"),
                          ("device", "default")],
                 title="Text encoder (Qwen3-VL 4B)", colour=TEAL, size=[400, 120])
    vae = g.add("VAELoader", 1, 2, outputs=[("VAE", "VAE")],
                widgets=[("vae_name", "qwen_image_vae.safetensors")], title="VAE", colour=TEAL,
                size=[400, 80])

    pos = g.add("CLIPTextEncode", 2, 0, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[("text", PROMPT)],
                title="Prompt", colour=GREEN, size=[400, 220])
    neg = g.add("ConditioningZeroOut", 2, 1.4, inputs=[("conditioning", "CONDITIONING")],
                outputs=[("CONDITIONING", "CONDITIONING")],
                title="Negative (zeroed - required at cfg 1.0)", colour=GREEN,
                size=[400, 60])
    latent = g.add("EmptySD3LatentImage", 2, 2.2, outputs=[("LATENT", "LATENT")],
                   widgets=[("width", 1024), ("height", 1024), ("batch_size", 1)],
                   title="Latent 1024x1024", colour=GREEN,
                   size=[400, 120])

    sampler = g.add("KSampler", 3, 0,
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                            ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                    outputs=[("LATENT", "LATENT")],
                    widgets=[("seed", 987654321), (None, "randomize"), ("steps", 8), ("cfg", 1.0),
                             ("sampler_name", "euler"), ("scheduler", "simple"),
                             ("denoise", 1.0)],
                    title="KSampler - 8 steps, cfg 1.0", colour=PURPLE, size=[400, 280])
    decode = g.add("VAEDecode", 4, 0, inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")], title="VAE Decode", colour=PURPLE,
                   size=[300, 60])
    save = g.add("SaveImage", 4, 0.7, inputs=[("images", "IMAGE")],
                 widgets=[("filename_prefix", "krea2_turbo_svdq")], title="Save", colour=PURPLE, size=[400, 300])

    diag = g.add("Krea2SVDQuantDiagnostics", 3, 2.4, inputs=[("model", "MODEL")],
                 outputs=[("MODEL", "MODEL"), ("STRING", "STRING")],
                 widgets=[("mode", "dispatch"), ("tokens", 4096)],
                 title="Diagnostics (optional - run if slow)", colour=YELLOW,
                 size=[400, 130], api=False)

    g.link(loader, "MODEL", sampler, "model")
    g.link(clip, "CLIP", pos, "clip")
    g.link(pos, "CONDITIONING", neg, "conditioning")
    g.link(pos, "CONDITIONING", sampler, "positive")
    g.link(neg, "CONDITIONING", sampler, "negative")
    g.link(latent, "LATENT", sampler, "latent_image")
    g.link(sampler, "LATENT", decode, "samples")
    g.link(vae, "VAE", decode, "vae")
    g.link(decode, "IMAGE", save, "images")
    g.link(loader, "MODEL", diag, "model")

    groups = [
        g.group("Load", (500, 20, 420, 620)),
        g.group("Prompt", (940, 20, 420, 620)),
        g.group("Sample", (1360, 20, 420, 640), colour="#8a4"),
    ]
    return g.serialize(groups), g.serialize_api()


def build_base():
    g = Graph()
    g.note(BASE_NOTE, 0, 0, size=(400, 560), title="READ ME FIRST")

    loader = g.add("Krea2SVDQuantW4A4Loader", 1, 0,
                   outputs=[("MODEL", "MODEL"), ("STRING", "STRING")],
                   widgets=[("model_name", "Krea2-Base-SVDQuant-W4A4-rank64.safetensors")],
                   title="Krea2 SVDQuant W4A4 Loader (base)", colour=TEAL, size=[400, 120])
    clip = g.add("CLIPLoader", 1, 1, outputs=[("CLIP", "CLIP")],
                 widgets=[("clip_name", "qwen3vl_4b_fp8_scaled.safetensors"), ("type", "krea2"),
                          ("device", "default")],
                 title="Text encoder (Qwen3-VL 4B)", colour=TEAL, size=[400, 120])
    vae = g.add("VAELoader", 1, 2, outputs=[("VAE", "VAE")],
                widgets=[("vae_name", "qwen_image_vae.safetensors")], title="VAE", colour=TEAL,
                size=[400, 80])

    pos = g.add("CLIPTextEncode", 2, 0, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[("text", PROMPT)],
                title="Positive prompt", colour=GREEN, size=[400, 200])
    neg = g.add("CLIPTextEncode", 2, 1.3, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[("text", NEGATIVE)],
                title="Negative prompt (base uses real CFG)", colour=GREEN,
                size=[400, 140])
    latent = g.add("EmptySD3LatentImage", 2, 2.3, outputs=[("LATENT", "LATENT")],
                   widgets=[("width", 1024), ("height", 1024), ("batch_size", 1)],
                   title="Latent 1024x1024", colour=GREEN,
                   size=[400, 120])

    sampler = g.add("KSampler", 3, 0,
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                            ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                    outputs=[("LATENT", "LATENT")],
                    widgets=[("seed", 987654321), (None, "randomize"), ("steps", 50), ("cfg", 3.5),
                             ("sampler_name", "euler"), ("scheduler", "simple"),
                             ("denoise", 1.0)],
                    title="KSampler - 50 steps, cfg 3.5", colour=PURPLE, size=[400, 280])
    decode = g.add("VAEDecode", 4, 0, inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")], title="VAE Decode", colour=PURPLE,
                   size=[300, 60])
    save = g.add("SaveImage", 4, 0.7, inputs=[("images", "IMAGE")],
                 widgets=[("filename_prefix", "krea2_base_svdq")], title="Save", colour=PURPLE, size=[400, 300])

    g.link(loader, "MODEL", sampler, "model")
    g.link(clip, "CLIP", pos, "clip")
    g.link(clip, "CLIP", neg, "clip")
    g.link(pos, "CONDITIONING", sampler, "positive")
    g.link(neg, "CONDITIONING", sampler, "negative")
    g.link(latent, "LATENT", sampler, "latent_image")
    g.link(sampler, "LATENT", decode, "samples")
    g.link(vae, "VAE", decode, "vae")
    g.link(decode, "IMAGE", save, "images")

    groups = [
        g.group("Load", (500, 20, 420, 560)),
        g.group("Prompt", (940, 20, 420, 620)),
        g.group("Sample", (1360, 20, 420, 640), colour="#8a4"),
    ]
    return g.serialize(groups), g.serialize_api()


def build_lora():
    """The Turbo graph with two LoRA loaders spliced in, the second one muted.

    Two rather than one because stacking is the question people actually ask, and a muted
    second node shows the answer (chain them) without changing what the graph does on first
    run.
    """
    g = Graph()
    g.note(LORA_NOTE, 0, 0, size=(400, 720), title="READ ME FIRST")

    loader = g.add("Krea2SVDQuantW4A4Loader", 1, 0,
                   outputs=[("MODEL", "MODEL"), ("STRING", "STRING")],
                   widgets=[("model_name", "Krea2-Turbo-SVDQuant-W4A4-rank256.safetensors")],
                   title="Krea2 SVDQuant W4A4 Loader (rank 256 for LoRAs)", colour=TEAL,
                   size=[400, 120])
    clip = g.add("CLIPLoader", 1, 1, outputs=[("CLIP", "CLIP")],
                 widgets=[("clip_name", "qwen3vl_4b_fp8_scaled.safetensors"), ("type", "krea2"),
                          ("device", "default")],
                 title="Text encoder (Qwen3-VL 4B)", colour=TEAL, size=[400, 120])
    vae = g.add("VAELoader", 1, 2, outputs=[("VAE", "VAE")],
                widgets=[("vae_name", "qwen_image_vae.safetensors")], title="VAE", colour=TEAL,
                size=[400, 80])

    lora1 = g.add("Krea2SVDQuantLoraLoader", 2, 0, inputs=[("model", "MODEL")],
                  outputs=[("model", "MODEL")],
                  widgets=[("lora_name", "your_krea2_lora.safetensors"), ("strength", 1.0),
                           ("adapters", ADAPTER_BYPASS)],
                  title="Krea2 SVDQuant LoRA Loader", colour=ORANGE, size=[400, 140])
    lora2 = g.add("Krea2SVDQuantLoraLoader", 2, 1, inputs=[("model", "MODEL")],
                  outputs=[("model", "MODEL")],
                  widgets=[("lora_name", "your_krea2_lora.safetensors"), ("strength", 0.6),
                           ("adapters", ADAPTER_BYPASS)],
                  title="Second LoRA (muted - ctrl-B to enable)", colour=ORANGE,
                  size=[400, 140], mode=4, api=False)

    pos = g.add("CLIPTextEncode", 3, 0, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[("text", PROMPT)],
                title="Prompt (add your LoRA trigger words)", colour=GREEN, size=[400, 220])
    neg = g.add("ConditioningZeroOut", 3, 1.4, inputs=[("conditioning", "CONDITIONING")],
                outputs=[("CONDITIONING", "CONDITIONING")],
                title="Negative (zeroed - required at cfg 1.0)", colour=GREEN, size=[400, 60])
    latent = g.add("EmptySD3LatentImage", 3, 2.2, outputs=[("LATENT", "LATENT")],
                   widgets=[("width", 1024), ("height", 1024), ("batch_size", 1)],
                   title="Latent 1024x1024", colour=GREEN, size=[400, 120])

    sampler = g.add("KSampler", 4, 0,
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                            ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                    outputs=[("LATENT", "LATENT")],
                    widgets=[("seed", 987654321), (None, "randomize"), ("steps", 8), ("cfg", 1.0),
                             ("sampler_name", "euler"), ("scheduler", "simple"), ("denoise", 1.0)],
                    title="KSampler - 8 steps, cfg 1.0", colour=PURPLE, size=[400, 280])
    decode = g.add("VAEDecode", 5, 0, inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")], title="VAE Decode", colour=PURPLE, size=[300, 60])
    save = g.add("SaveImage", 5, 0.7, inputs=[("images", "IMAGE")],
                 widgets=[("filename_prefix", "krea2_turbo_lora")], title="Save", colour=PURPLE,
                 size=[400, 300])

    g.link(loader, "MODEL", lora1, "model")
    g.link(lora1, "model", lora2, "model")
    g.link(lora1, "model", sampler, "model")
    g.link(clip, "CLIP", pos, "clip")
    g.link(pos, "CONDITIONING", neg, "conditioning")
    g.link(pos, "CONDITIONING", sampler, "positive")
    g.link(neg, "CONDITIONING", sampler, "negative")
    g.link(latent, "LATENT", sampler, "latent_image")
    g.link(sampler, "LATENT", decode, "samples")
    g.link(vae, "VAE", decode, "vae")
    g.link(decode, "IMAGE", save, "images")

    groups = [
        g.group("Load", (500, 20, 420, 620)),
        g.group("LoRA - chain one node per LoRA", (940, 20, 420, 420), colour="#a85"),
        g.group("Prompt", (1360, 20, 420, 620)),
        g.group("Sample", (1780, 20, 420, 640), colour="#8a4"),
    ]
    return g.serialize(groups), g.serialize_api()


def build_diagnostics():
    """Env Check + Diagnostics. No sampler: nothing here generates an image."""
    g = Graph()
    g.note(DIAG_NOTE, 0, 0, size=(430, 700), title="READ ME FIRST")

    env = g.add("Krea2SVDQuantEnvCheck", 1, 0, outputs=[("report", "STRING")],
                title="1. Env Check - no model needed, run this first", colour=YELLOW,
                size=[400, 80])
    env_view = g.add("PreviewAny", 1, 0.6, inputs=[("source", "*")],
                     title="Env Check report", colour=YELLOW, size=[400, 300])

    loader = g.add("Krea2SVDQuantW4A4Loader", 2, 0,
                   outputs=[("MODEL", "MODEL"), ("STRING", "STRING")],
                   widgets=[("model_name", "Krea2-Turbo-SVDQuant-W4A4-rank256-actaware.safetensors")],
                   title="2. Load the checkpoint you are diagnosing", colour=TEAL,
                   size=[400, 120])
    loader_view = g.add("PreviewAny", 2, 0.7, inputs=[("source", "*")],
                        title="Loader status - names the kernel that will run", colour=TEAL,
                        size=[400, 200])

    diag = g.add("Krea2SVDQuantDiagnostics", 3, 0, inputs=[("model", "MODEL")],
                 outputs=[("model", "MODEL"), ("report", "STRING")],
                 widgets=[("mode", "dispatch"), ("tokens", 4096)],
                 title="3. Diagnostics - start with mode=dispatch", colour=YELLOW,
                 size=[400, 130])
    diag_view = g.add("PreviewAny", 3, 0.8, inputs=[("source", "*")],
                      title="Diagnostics report - paste this into an issue", colour=YELLOW,
                      size=[400, 400])

    g.link(env, "report", env_view, "source")
    g.link(loader, "STRING", loader_view, "source")
    g.link(loader, "MODEL", diag, "model")
    g.link(diag, "report", diag_view, "source")

    groups = [
        g.group("Step 1 - does the int4 kernel exist at all?", (500, 20, 420, 480),
                colour="#a85"),
        g.group("Step 2 - load it", (940, 20, 420, 480)),
        g.group("Step 3 - what actually runs", (1360, 20, 420, 620), colour="#8a4"),
    ]
    return g.serialize(groups), g.serialize_api()


def _check_adapter_constant():
    """The `adapters` widget value must match svdquant_lora.ADAPTER_BYPASS exactly.

    ComfyUI matches combo widgets by string. A drifted copy here would write a workflow whose
    dropdown silently falls back to the first option, so read it out of the source rather than
    trusting that two literals stayed equal.
    """
    src = os.path.join(os.path.dirname(HERE), "svdquant_lora.py")
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ADAPTER_BYPASS"):
                actual = line.split("=", 1)[1].strip().strip('"')
                if actual != ADAPTER_BYPASS:
                    raise SystemExit(
                        "ADAPTER_BYPASS drifted: svdquant_lora.py has {!r}, this script has "
                        "{!r}".format(actual, ADAPTER_BYPASS))
                return
    raise SystemExit("could not find ADAPTER_BYPASS in " + src)


QUANTIZE_NOTE = """MAKE YOUR OWN QUANTIZED CHECKPOINT - no terminal needed

This graph is the whole quantizer. Set the widgets, press Queue,
wait. It does exactly what `python quantize_krea2.py` does (same
function, not a copy), and writes the result next to the source in
ComfyUI/models/diffusion_models/.

BEFORE YOU START
- You need the BF16 Krea 2 checkpoint (~24 GB) in
  ComfyUI/models/diffusion_models/. FP8 also works as a source; any
  other already-quantized file does not.
- ~12 GB free on that drive. The output is ~8 GB.
- Run the Env Check node first (it needs nothing wired). If it says
  the cuda backend is not live, the file you are about to build will
  run SLOWER than fp8 - fix torch before spending the disk.

WHICH FORMAT
  svdq  4-bit + a low-rank bf16 correction branch. Best quality of
        the 4-bit options. Loads with the Krea2 SVDQuant W4A4 Loader.
  w4a4  same without the branch. Smaller, ~9% faster per step.
  int8  most faithful, still ~2x fp8 on Ampere. Stock UNETLoader.
  fp8   storage only, no speedup here.
  w4a4 / int8 / fp8 ignore rank, rank_alloc and refine_iters.

WHICH RANK (svdq only)
  No LoRA planned -> 64. 128 and 256 measured the same.
  LoRAs planned  -> 256. It wins clearly there; 64 loses most of
                    its advantage once a LoRA is patched in.
  Keep refine_iters at 100. At 0 the split is single-shot (~54s
  instead of ~5.7min) and raising rank then buys nothing.

WHAT TO EXPECT WHILE IT RUNS
- The queue is BLOCKED. Nothing else generates. 54s to ~6 min.
- Every loaded model is unloaded first, so your next generation
  pays a reload. This is deliberate: the dequantize step needs the
  card to itself or it OOMs.
- overwrite is off on purpose. An existing file of the same name is
  an error, not 8 GB written over last night's run.

The summary appears on the node when it finishes, and says which
loader to use and what sampler settings that variant wants.

FREE QUALITY: FILL IN act_stats (svdq only)
Download the calibration file for your variant from
huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI/tree/main/calibration
into ComfyUI/output/, then type its name into act_stats:

  Turbo -> krea2_act_stats.safetensors
  base  -> krea2_act_stats_base.safetensors

6.67 MB each. They fit the low-rank branch against activations
measured from a real sampling pass instead of assuming every input
channel matters equally. LPIPS to BF16 0.3378 -> 0.2825 at rank 64.
Same rank, same file size, same kernel, same speed. There is no
reason to skip this.

Take the file that matches the model you are quantizing. A mismatch
is caught before any GPU work only if the layer NAMES differ, and
Turbo and base share their names - so the check will not save you
from feeding Turbo's statistics to a base build. Read the filename.

If you are quantizing a finetune, a merge, or anything these two
files do not describe, capture your own:
workflows/krea2_quantize_calibrated.json does the capture and the
quantize in one Queue press."""

CALIB_NOTE = """CALIBRATED QUANTIZE - measure your own activations

YOU PROBABLY DO NOT NEED THIS GRAPH.
Calibration files for stock Krea 2 Turbo and base are published at
huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI/tree/main/calibration
- 6.67 MB each. Download the one matching your variant, drop it in
ComfyUI/output/, name it in the act_stats box of the plain
krea2_quantize.json graph, and you get the identical quality lever
without running a sampling pass.

This graph is for the case those files do not cover: a finetune, a
merge, a model whose weights are not the released ones. Activation
statistics describe the WEIGHTS they were captured from. Same
architecture is not the same model - Turbo's file loads cleanly into
a base build and quietly describes the wrong thing.

WHAT IT BUYS
The low-rank branch is fitted against activations measured from a
real sampling pass instead of assuming every input channel matters
equally. Measured, no LoRA, rank 64: LPIPS to BF16 0.3378 -> 0.2825.
Same rank, same file size, same kernel, same speed.

WHAT RUNS, IN ORDER
1. The BF16 model is loaded with the stock UNETLoader and sampled
   once. Capture Start hooks every layer that will be quantized and
   records the RMS of its inputs; Capture Save writes them out.
2. Capture Save's act_stats_path feeds the Quantize node. That wire
   is not cosmetic - it is what stops ComfyUI running the quantizer
   first, which would silently produce an uncalibrated file.

So one Queue press does calibration AND quantization. Expect the
sampling pass plus 54s-6min of quantizing, with the queue blocked.

SET THIS BEFORE YOU RUN
- UNETLoader: the BF16 checkpoint. Statistics from an already
  quantized model describe the wrong thing.
- Quantize node: the SAME file as source_model. It is a separate
  dropdown, and no, it cannot be wired - the loader hands over a
  MODEL object, not a filename.
- variant: turbo or base, so the output gets the right name.
- KSampler: 20 steps / cfg 3.5 suits base. For Turbo use 8 steps
  and cfg 1.0, and swap the negative prompt for ConditioningZeroOut.

CALIBRATING ON SEVERAL PROMPTS (better, optional)
The defaults calibrate on one prompt, which is enough to beat no
calibration. To use more: set keep_capturing = true on Capture Save,
mute the Quantize node (select it, Ctrl+M), queue as many prompts as
you like with reset = false on Capture Start after the first, then
unmute Quantize and queue once more. Prompts should look like what
you actually generate.

Only svdq uses act_stats. The other formats have no branch to fit,
and wiring act_stats into them is an error rather than a no-op."""


def build_quantize():
    g = Graph()
    g.note(QUANTIZE_NOTE, 0, 0, size=(460, 1360), title="READ ME FIRST")

    g.add("Krea2SVDQuantEnvCheck", 1, 0, outputs=[("report", "STRING")],
          title="1. Env Check - run this first, it needs nothing", colour=YELLOW,
          size=[420, 120])
    g.add("Krea2SVDQuantQuantize", 1, 0.9,
                  outputs=[("summary", "STRING")],
                  widgets=[("source_model", "krea2_bf16.safetensors"), ("format", "svdq"),
                           ("rank", 64), ("rank_alloc", "uniform"), ("refine_iters", 100),
                           ("groupsize", 256), ("variant", "turbo"), ("output_name", ""),
                           ("overwrite", False), ("act_stats", ""), ("seed", 0)],
          title="2. Quantize - set source_model and act_stats, then Queue", colour=TEAL,
          size=[420, 460])

    groups = [g.group("Quantize", (500, 20, 460, 640), colour="#8a4")]
    return g.serialize(groups), g.serialize_api()


def build_quantize_calibrated():
    g = Graph()
    g.note(CALIB_NOTE, 0, 0, size=(460, 1420), title="READ ME FIRST")

    unet = g.add("UNETLoader", 1, 0, outputs=[("MODEL", "MODEL")],
                 widgets=[("unet_name", "krea2_bf16.safetensors"),
                          ("weight_dtype", "default")],
                 title="BF16 source (stock loader)", colour=TEAL, size=[400, 120])
    clip = g.add("CLIPLoader", 1, 1, outputs=[("CLIP", "CLIP")],
                 widgets=[("clip_name", "qwen3vl_4b_fp8_scaled.safetensors"), ("type", "krea2"),
                          ("device", "default")],
                 title="Text encoder (Qwen3-VL 4B)", colour=TEAL, size=[400, 120])

    start = g.add("Krea2SVDQuantCaptureStart", 2, 0, inputs=[("model", "MODEL")],
                  outputs=[("model", "MODEL"), ("status", "STRING")],
                  widgets=[("reset", True)],
                  title="Capture Start - before the sampler", colour=GREEN, size=[400, 100])
    pos = g.add("CLIPTextEncode", 2, 0.8, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[("text", PROMPT)],
                title="Calibration prompt", colour=GREEN, size=[400, 200])
    neg = g.add("CLIPTextEncode", 2, 2.0, inputs=[("clip", "CLIP")],
                outputs=[("CONDITIONING", "CONDITIONING")], widgets=[("text", NEGATIVE)],
                title="Negative (Turbo: replace with ConditioningZeroOut)", colour=GREEN,
                size=[400, 140])
    latent = g.add("EmptySD3LatentImage", 2, 3.0, outputs=[("LATENT", "LATENT")],
                   widgets=[("width", 1024), ("height", 1024), ("batch_size", 1)],
                   title="Latent 1024x1024", colour=GREEN, size=[400, 120])

    sampler = g.add("KSampler", 3, 0,
                    inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                            ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                    outputs=[("LATENT", "LATENT")],
                    widgets=[("seed", 987654321), (None, "randomize"), ("steps", 20),
                             ("cfg", 3.5), ("sampler_name", "euler"), ("scheduler", "simple"),
                             ("denoise", 1.0)],
                    title="Calibration pass - base 20/3.5, Turbo 8/1.0", colour=PURPLE,
                    size=[400, 280])

    save = g.add("Krea2SVDQuantCaptureSave", 4, 0, inputs=[("latent", "LATENT")],
                 outputs=[("latent", "LATENT"), ("status", "STRING"),
                          ("act_stats_path", "STRING")],
                 widgets=[("filename", "krea2_act_stats.safetensors"),
                          ("keep_capturing", False)],
                 title="Capture Save - after the sampler", colour=GREEN, size=[400, 120])

    quant = g.add("Krea2SVDQuantQuantize", 5, 0,
                  inputs=[("act_stats", "STRING")],
                  outputs=[("summary", "STRING")],
                  widgets=[("source_model", "krea2_bf16.safetensors"), ("format", "svdq"),
                           ("rank", 64), ("rank_alloc", "uniform"), ("refine_iters", 100),
                           ("groupsize", 256), ("variant", "turbo"), ("output_name", ""),
                           ("overwrite", False), ("act_stats", ""), ("seed", 0)],
                  title="Quantize - source_model must match the loader above", colour=TEAL,
                  size=[420, 440], widget_inputs=("act_stats",))

    g.link(unet, "MODEL", start, "model")
    g.link(start, "model", sampler, "model")
    g.link(clip, "CLIP", pos, "clip")
    g.link(clip, "CLIP", neg, "clip")
    g.link(pos, "CONDITIONING", sampler, "positive")
    g.link(neg, "CONDITIONING", sampler, "negative")
    g.link(latent, "LATENT", sampler, "latent_image")
    g.link(sampler, "LATENT", save, "latent")
    g.link(save, "act_stats_path", quant, "act_stats")

    groups = [
        g.group("Load BF16", (500, 20, 420, 420)),
        g.group("Calibration pass", (940, 20, 420, 800)),
        g.group("Quantize", (2200, 20, 460, 520), colour="#8a4"),
    ]
    return g.serialize(groups), g.serialize_api()


def _write(path, payload):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main():
    _check_adapter_constant()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, build in (("krea2_turbo_svdquant_w4a4_t2i", build_turbo),
                        ("krea2_base_svdquant_w4a4_t2i", build_base),
                        ("krea2_turbo_svdquant_w4a4_lora", build_lora),
                        ("krea2_svdquant_diagnostics", build_diagnostics),
                        ("krea2_quantize", build_quantize),
                        ("krea2_quantize_calibrated", build_quantize_calibrated)):
        graph, api = build()
        ui_path = os.path.join(OUT_DIR, name + ".json")
        api_path = os.path.join(OUT_DIR, name + "_api.json")
        _write(ui_path, graph)
        _write(api_path, api)
        print("wrote {}  ({} nodes, {} links)".format(
            ui_path, len(graph["nodes"]), len(graph["links"])))
        print("wrote {}  ({} nodes)".format(api_path, len(api)))


if __name__ == "__main__":
    main()
