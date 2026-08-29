"""gemma_vllm_readout.py — Gemma-4-26B-A4B-it OUTPUT arm.

Holds two readouts for output_arm.run:
  - GemmaReadout      : the HF (transformers .generate) output readout — sample N REGION|POINT
                        answers to the shared frozen prompt and parse the ranked coordinates
                        (mode="box" is the Gemma-native JSON box_2d ablation). Used by
                        run_coord_sample_gemma.py.
  - GemmaVllmReadout  : subclass that swaps in vLLM for THROUGHPUT at scale-up (continuous +
                        chunked cross-image batching), reusing GemmaReadout's prompt_for/parse.
                        Used by run_coord_sample_gemma_vllm.py.

Gemma is REPORTED in Instruct (non-thinking) mode. vLLM does NOT expose attention weights, so the
ATTENTION arm stays on the HF eager path (attention/gemma_attn_readout.py); this file is OUTPUT-only.

RUN THE vLLM PATH IN A vLLM ENV WITH Gemma-4 SUPPORT — its per-layer head_dim needs vLLM>=0.27.2.
26B-total MoE -> 2 GPUs (TP=2). Tunables (env, NOT
the VLLM_ namespace): GEMMA_TP, GEMMA_MAX_LEN, GEMMA_GPU_UTIL, GEMMA_MAX_SEQS, GEMMA_MAX_NEW,
GEMMA_VLLM_ID, GEMMA_ALLOW_GLOBAL_HEADDIM.

KNOWN ENV FRICTION (vLLM < 0.27.2): the Gemma4 arch is REGISTERED, but building its
ModelConfig crashes — vLLM's convertor reads config.head_dim globally while this transformers guards
head_dim as a per-layer attribute (AmbiguousGlobalPerLayerAttributeError). Resolution order:
  1. Check whether head_dim is uniform across layers:
       python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('google/gemma-4-26B-A4B-it'); \
         pl=getattr(getattr(c,'text_config',c),'per_layer_config',None); \
         print(sorted({getattr(l,'head_dim',None) for l in pl}) if pl else 'no per_layer_config')"
  2. If it prints a SINGLE value -> set GEMMA_ALLOW_GLOBAL_HEADDIM=1 (below) and rerun; the global
     value is correct and vLLM builds.
  3. If it prints MULTIPLE values -> do NOT use the flag (vLLM would mis-slice). Build a dedicated
     gemma-vllm env with a newer vLLM that reads head_dim per-layer, leaving glm-vllm-env (Qwen/GLM)
     untouched.
"""
import json
import os
import re

import numpy as np
import torch

from config import FREEVIEW_PROMPT, FROZEN_PROMPT, FROZEN_SYSTEM, MODEL_IDS

# One env var controls the Gemma model id for both output paths (default = the reported weights).
MODEL_ID = os.environ.get("GEMMA_VLLM_ID", MODEL_IDS["gemma"])

# Gemma-4's NATIVE grounding format (per ai.google.dev/gemma/docs/capabilities/vision/image):
# a JSON list of {"box_2d": [y1, x1, y2, x2], "label": ...}, Y-FIRST, normalized to a 1000x1000
# grid. mode="box" prompts in this native format (parse the JSON, take box centres) — an ablation
# for when the free-form POINT prompt's yield is flaky. Same task/wording as FROZEN_PROMPT.
BOX_PROMPT = (
    "You are searching for a {t} in this image. If a {t} is clearly visible, give its bounding "
    "box; otherwise, give the bounding boxes of all locations in this image you would examine "
    "before you can decide whether a {t} is present. Rank the locations from most to least "
    "likely. Output ONLY a JSON list in exactly this format:\n"
    '[{{"box_2d": [y1, x1, y2, x2], "label": <a short noun phrase for the place>}}, ...]\n'
    "where y1, x1, y2, x2 are integers from 0 to 1000 giving the top, left, bottom, and right "
    "edges (y top-to-bottom, x left-to-right), normalized to a 1000x1000 grid."
)

_PT = re.compile(r'POINT\s*[:=]?\s*\(?\s*(\d{1,4})\s*[,;]\s*(\d{1,4})', re.I)
# Gemma boxes are Y-FIRST [y1,x1,y2,x2]; this 4-int matcher is used for both the JSON-fallback
# and the point-mode fallback, interpreted y-first below.
_BOX = re.compile(r'\[?\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]')
_BOX2D = re.compile(r'"box_2d"\s*:\s*\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]'
                    r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?', re.I)
_REG = re.compile(r'REGION\s*[:=]\s*([^\n|]+)', re.I)


def _yx_box_centre(y1, x1, y2, x2, iw, ih):
    """Gemma box_2d is [y1,x1,y2,x2] (y-first). Return (x1000, y1000, xpx, ypx) of the centre."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if not (0 <= cx <= 1000 and 0 <= cy <= 1000):
        return None
    return (cx, cy, float(np.clip(cx / 1000.0 * iw, 0, iw - 1)),
            float(np.clip(cy / 1000.0 * ih, 0, ih - 1)))


class GemmaReadout:
    name = MODEL_ID

    def __init__(self, mode="point", max_new_tokens=420, temperature=0.7, top_p=0.8, top_k=20,
                 batch=4, min_new_tokens=64):
        self.mode = mode            # "point" = shared REGION|POINT prompt (default);
                                    # "box" = Gemma-native JSON box_2d detection (ablation)
        self.name = f"{MODEL_ID} ({mode})"
        self.model = None
        self.processor = None
        self.max_new_tokens = max_new_tokens
        # Gemma-4 tends to emit a canned "I did not find any bounding box detections for {t}."
        # and hit EOS before the ranked REGION|POINT list. min_new_tokens suppresses EOS until
        # this many tokens are generated, forcing it past that bail-out into the enumeration
        # (~15 tokens = the canned line; 64 pushes it into a few REGION lines).
        self.min_new_tokens = min_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.batch = batch          # num_return_sequences per forward pass (lower if OOM)

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)

    def prompt_for(self, task):
        if self.mode == "freeview":
            return FREEVIEW_PROMPT          # target-free memory baseline; `task` unused
        return (BOX_PROMPT if self.mode == "box" else FROZEN_PROMPT).format(t=task)

    def sample_answers(self, image, prompt, n):
        img = image.convert("RGB")
        # Gemma has no dedicated system turn; fold FROZEN_SYSTEM into the user text (identical
        # wording, delivered in the user turn).
        text = (FROZEN_SYSTEM + "\n\n" + prompt) if FROZEN_SYSTEM else prompt
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        plen = inputs["input_ids"].shape[1]
        outs = []
        remaining = n
        while remaining > 0:
            k = min(self.batch, remaining)
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs, do_sample=True, temperature=self.temperature,
                    top_p=self.top_p, top_k=self.top_k, max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens, num_return_sequences=k)
            for i in range(gen.shape[0]):
                outs.append(self.processor.decode(gen[i, plen:], skip_special_tokens=True))
            remaining -= k
        return outs

    def parse(self, text, iw, ih):
        if self.mode == "box":
            pts = self._boxes(text, iw, ih)
            if pts:
                return pts
        return self._points(text, iw, ih)      # point mode, or box-mode fallback

    def _points(self, text, iw, ih):
        regs = [m.strip()[:40] for m in _REG.findall(text)]
        pts = []
        hits = _PT.findall(text)
        if hits:
            for i, (a, b) in enumerate(hits):
                x, y = int(a), int(b)
                if not (0 <= x <= 1000 and 0 <= y <= 1000):
                    continue
                xpx = float(np.clip(x / 1000.0 * iw, 0, iw - 1))
                ypx = float(np.clip(y / 1000.0 * ih, 0, ih - 1))
                pts.append((x, y, xpx, ypx, regs[i] if i < len(regs) else ""))
        else:                                   # fall back to box centres (Gemma boxes are y-first)
            for a, b, c, d in _BOX.findall(text):
                c0 = _yx_box_centre(int(a), int(b), int(c), int(d), iw, ih)
                if c0:
                    pts.append((*c0, ""))
        return pts

    def _boxes(self, text, iw, ih):
        """Parse Gemma's native JSON detection: list of {"box_2d":[y1,x1,y2,x2],"label":...}."""
        pts = []
        # Preferred: strip ```json fences and json.loads the bracketed list.
        m = re.search(r'\[\s*\{.*\}\s*\]', text, re.S)
        if m:
            try:
                for it in json.loads(m.group(0)):
                    b = it.get("box_2d")
                    if isinstance(b, list) and len(b) == 4:
                        c0 = _yx_box_centre(int(b[0]), int(b[1]), int(b[2]), int(b[3]), iw, ih)
                        if c0:
                            pts.append((*c0, str(it.get("label", ""))[:40]))
            except Exception:                   # noqa: BLE001 - fall through to regex
                pts = []
        if not pts:                             # regex fallback over "box_2d":[...] occurrences
            for a, b, c, d, lab in _BOX2D.findall(text):
                c0 = _yx_box_centre(int(a), int(b), int(c), int(d), iw, ih)
                if c0:
                    pts.append((*c0, (lab or "")[:40]))
        return pts


class GemmaVllmReadout(GemmaReadout):
    """vLLM output readout for Gemma-4 (INSTRUCT). Reuses GemmaReadout.prompt_for / _points / _boxes;
    overrides load (vLLM), sample_answers[/ _batch], and parse. Gemma is reported in Instruct mode
    (see config); parses REGION|POINT directly."""

    def __init__(self, mode="point", max_new_tokens=None, temperature=0.7, top_p=0.8, top_k=20):
        super().__init__(mode=mode)
        self.name = f"{MODEL_ID} ({mode}, vLLM)"
        # Instruct answers are short (ranked REGION|POINT list).
        self.max_new_tokens = int(os.environ.get("GEMMA_MAX_NEW", max_new_tokens or 420))
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.llm = None
        self._proc = None

    def load(self):
        if self.llm is not None:
            return
        import torch
        from vllm import LLM
        from transformers import AutoProcessor
        self._proc = AutoProcessor.from_pretrained(MODEL_ID)
        tp = int(os.environ.get("GEMMA_TP", max(1, torch.cuda.device_count())))   # 26B MoE -> TP=2
        # vLLM 0.27.1's arch-config convertor reads config.head_dim GLOBALLY, but this transformers
        # build guards head_dim as a per-layer attribute for Gemma-4 and raises
        # (AmbiguousGlobalPerLayerAttributeError). If head_dim is actually UNIFORM across layers, the
        # global value is correct — GEMMA_ALLOW_GLOBAL_HEADDIM=1 tells the config to allow global
        # access so vLLM can build. ONLY safe when head_dim is uniform (verify first; see the header).
        overrides = None
        if os.environ.get("GEMMA_ALLOW_GLOBAL_HEADDIM") == "1":
            def overrides(cfg):                                       # noqa: E306 - callable hf_overrides
                for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "vision_config", None)):
                    if c is not None:
                        try:
                            setattr(c, "allow_global_per_layer_attribute_access", True)
                        except Exception:                            # noqa: BLE001
                            pass
                return cfg
            print("[gemma-vllm] GEMMA_ALLOW_GLOBAL_HEADDIM=1: allowing global head_dim access "
                  "(valid only if head_dim is uniform across layers)", flush=True)
        kw = {"hf_overrides": overrides} if overrides is not None else {}
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            trust_remote_code=False,                    # Gemma-4 is native in recent vLLM
            tensor_parallel_size=tp,
            max_model_len=int(os.environ.get("GEMMA_MAX_LEN", "8192")),   # image + trace + answer
            gpu_memory_utilization=float(os.environ.get("GEMMA_GPU_UTIL", "0.92")),
            limit_mm_per_prompt={"image": 1},
            max_num_seqs=int(os.environ.get("GEMMA_MAX_SEQS", "48")),
            **kw,
        )
        print(f"[gemma-vllm] LLM ready: tp={tp} max_len={os.environ.get('GEMMA_MAX_LEN','8192')} "
              f"max_new={self.max_new_tokens}", flush=True)

    def _req(self, image, prompt):
        """(image, prompt) -> vLLM request dict. Gemma has no system turn: FROZEN_SYSTEM is folded
        into the user text (identical wording)."""
        text = (FROZEN_SYSTEM + "\n\n" + prompt) if FROZEN_SYSTEM else prompt
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        ps = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": ps, "multi_modal_data": {"image": image.convert("RGB")}}

    def _sp(self, n):
        from vllm import SamplingParams
        # min_tokens mirrors the HF readout's min_new_tokens: Gemma-4 otherwise emits a canned
        # "I did not find any bounding box detections for {t}." and hits EOS BEFORE the ranked
        # REGION|POINT list; suppressing EOS for the first min_new_tokens forces it past the bail-out.
        return SamplingParams(n=n, temperature=self.temperature, top_p=self.top_p,
                              top_k=self.top_k, max_tokens=self.max_new_tokens,
                              min_tokens=int(getattr(self, "min_new_tokens", 64) or 0))

    def sample_answers(self, image, prompt, n):
        out = self.llm.generate(self._req(image, prompt), sampling_params=self._sp(n))
        return [o.text for o in out[0].outputs]

    def sample_answers_batch(self, pairs, n):
        """BATCHED: many (image, prompt) pairs in one llm.generate — cross-image continuous batching."""
        reqs = [self._req(image, prompt) for image, prompt in pairs]
        outs = self.llm.generate(reqs, sampling_params=self._sp(n))
        return [[o.text for o in ro.outputs] for ro in outs]

    def parse(self, text, iw, ih):
        """Instruct: parse REGION|POINT directly with the inherited Gemma point parser."""
        return super().parse(text, iw, ih)
