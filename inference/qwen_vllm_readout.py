"""qwen_vllm_readout.py — Qwen3-VL-32B-Instruct coordinate readout via vLLM.

Coordinates are 0-1000 normalized per axis, REGION|POINT, top-MAX_POINTS ranked. If the model
emits a <think>...</think> block, the answer is taken AFTER </think>. QWEN_VLLM_ID overrides the
model id. RUN IN A vLLM env (32B needs 2 GPUs -> TP=2).

Interface for coord_sample.run: load / prompt_for / sample_answers[/ _batch] / parse — with chunked
batching. Tunables (env, NOT in the VLLM_ namespace): QWEN_TP, QWEN_MAX_LEN, QWEN_GPU_UTIL,
QWEN_MAX_SEQS, QWEN_MAX_NEW.
"""
import os
import re

import numpy as np

from config import FROZEN_SYSTEM, FROZEN_PROMPT, FREEVIEW_PROMPT, MODEL_IDS

MODEL_ID = os.environ.get("QWEN_VLLM_ID", MODEL_IDS["qwen-instruct"])

MAX_POINTS = 12            # cap points parsed per sampled scanpath


def parse_points(t):
    ps = re.findall(r'POINT\s*[:=]?\s*\(?\s*(\d{1,4})\s*[,;]\s*(\d{1,4})', t, re.I)
    if not ps:
        ps = re.findall(r'\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)', t)
    return [(int(a), int(b)) for a, b in ps]


def parse_regions(t):
    return [m.strip()[:40] for m in re.findall(r'REGION\s*[:=]\s*([^\n|]+)', t, re.I)]


class QwenVllmReadout:
    def __init__(self, mode="point", max_new_tokens=6144, temperature=0.7, top_p=0.8, top_k=20):
        self.mode = mode
        self.name = f"{MODEL_ID} ({mode}, vLLM)"
        self.max_new_tokens = int(os.environ.get("QWEN_MAX_NEW", max_new_tokens))
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
        tp = int(os.environ.get("QWEN_TP", max(1, torch.cuda.device_count())))   # 32B -> TP=2
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            trust_remote_code=False,                    # Qwen3-VL is native in vLLM
            tensor_parallel_size=tp,
            max_model_len=int(os.environ.get("QWEN_MAX_LEN", "10240")),   # image tokens + trace+answer
            gpu_memory_utilization=float(os.environ.get("QWEN_GPU_UTIL", "0.92")),
            limit_mm_per_prompt={"image": 1},
            max_num_seqs=int(os.environ.get("QWEN_MAX_SEQS", "64")),
        )
        print(f"[qwen-vllm] LLM ready: tp={tp} max_len={os.environ.get('QWEN_MAX_LEN','10240')} "
              f"max_new={self.max_new_tokens}", flush=True)

    def prompt_for(self, task):
        if self.mode == "freeview":
            return FREEVIEW_PROMPT                       # target-free memory baseline
        return FROZEN_PROMPT.format(t=task)

    def _req(self, image, prompt):
        """(image, prompt) -> vLLM request dict. Qwen keeps a SEPARATE system turn (unlike GLM)."""
        messages = []
        if FROZEN_SYSTEM:
            messages.append({"role": "system", "content": FROZEN_SYSTEM})
        messages.append({"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
        ps = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": ps, "multi_modal_data": {"image": image.convert("RGB")}}

    def _sp(self, n):
        from vllm import SamplingParams
        return SamplingParams(n=n, temperature=self.temperature, top_p=self.top_p,
                              top_k=self.top_k, max_tokens=self.max_new_tokens)

    def sample_answers(self, image, prompt, n):
        out = self.llm.generate(self._req(image, prompt), sampling_params=self._sp(n))
        return [o.text for o in out[0].outputs]

    def sample_answers_batch(self, pairs, n):
        """BATCHED: many (image, prompt) pairs in one llm.generate — cross-image continuous batching."""
        reqs = [self._req(image, prompt) for image, prompt in pairs]
        outs = self.llm.generate(reqs, sampling_params=self._sp(n))
        return [[o.text for o in ro.outputs] for ro in outs]

    def parse(self, text, iw, ih):
        """Answer AFTER </think>; Qwen coords are 0-1000 normalized per axis (x/1000*iw)."""
        ans = text.split("</think>")[-1] if "</think>" in text else text
        pts = parse_points(ans)[:MAX_POINTS]
        regs = parse_regions(ans)
        out = []
        for i, (px, py) in enumerate(pts):
            xpx = float(np.clip(px / 1000.0 * iw, 0, iw - 1))
            ypx = float(np.clip(py / 1000.0 * ih, 0, ih - 1))
            reg = regs[i] if i < len(regs) else ""
            out.append((px, py, xpx, ypx, reg))
        return out
