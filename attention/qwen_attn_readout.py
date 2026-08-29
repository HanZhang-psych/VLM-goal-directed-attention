"""
qwen_attn_readout.py — load Qwen3-VL (HF, eager attention) and capture its attention weights
for the attention arm. Used by run_attn_qwen.py.

load() puts the model on the GPU(s) and installs an attention-capture hook. capture_raw()
runs one forward pass over an image + prompt and returns the raw per-layer, per-head
attention from the target word (and a neutral baseline) to the image patches. The maps
themselves are built offline from these arrays by rescore_attn.py.

Needs transformers >= 4.57 (Qwen3-VL support).
"""

import math

import numpy as np
import torch
from PIL import Image

from config import (ATTN_BANDS, DEFAULT_ATTN_BAND, FROZEN_ATTN_BAND,
                    FROZEN_SYSTEM, model_key)

# Constants from the Qwen3-VL config.
IMAGE_TOKEN_ID = 151655      # the <|image_pad|> placeholder token
SPATIAL_MERGE = 2            # ViT patches are merged 2x2 before entering the LLM

# Cap on input image resolution (total pixels); bounds the vision-patch count and the
# attention memory. Set by load().
_MAX_PIXELS = 768 * 768

# Captured attention, keyed by transformer layer index. Filled by the capture hook,
# read by capture_raw().
_CAPTURED = {}


def _build_messages(prompt, system=FROZEN_SYSTEM):
    msgs = []
    if system:                      # omit the system turn entirely when falsy
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
    return msgs


def _install_capture():
    """Register an 'eager_capture' attention implementation.

    It computes attention exactly like the built-in 'eager' path (model outputs unchanged)
    but also saves the attention weights for the text-decoder layers (filtered by class name
    so the vision encoder's self-attention is not captured).
    """
    import transformers.models.qwen3_vl.modeling_qwen3_vl as qvl
    reference_eager = qvl.eager_attention_forward  # the correct math, reused

    def eager_capture(module, query, key, value, attention_mask, scaling,
                      dropout=0.0, **kwargs):
        attn_output, attn_weights = reference_eager(
            module, query, key, value, attention_mask, scaling,
            dropout=dropout, **kwargs
        )
        # Only keep the language model's attention (name contains "Text"),
        # and only when weights were actually returned.
        if attn_weights is not None and "Text" in module.__class__.__name__:
            layer_idx = getattr(module, "layer_idx", len(_CAPTURED))
            _CAPTURED[layer_idx] = attn_weights.detach().to("cpu", torch.float32)
        return attn_output, attn_weights

    # Preferred: the official extension API.
    try:
        from transformers import AttentionInterface
        AttentionInterface.register("eager_capture", eager_capture)
    except Exception:
        # Fallback for older/newer variants: write into the global registry.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager_capture"] = eager_capture


def readout_band(model, model_id):
    """Decoder-layer indices in this model's FROZEN search band, derived from config
    (config.FROZEN_ATTN_BAND -> ATTN_BANDS fraction range over the model's decoder-layer count).
    e.g. Qwen-Thinking 'mid' (0.4-0.7) over 64 layers -> range(25, 44) = layers 25..43. This is the
    same band rescore_attn.attn_map_from_raw pools, so the residency guard matches scoring."""
    tc = getattr(model.config, "text_config", model.config)
    L = int(getattr(tc, "num_hidden_layers", 0))
    lo_f, hi_f = ATTN_BANDS[FROZEN_ATTN_BAND.get(model_key(model_id), DEFAULT_ATTN_BAND)]
    return range(int(lo_f * L), int(hi_f * L))


def load(model_id, device_map="auto", max_pixels=768 * 768, capture=True):
    """Load the model and processor.

    device_map="auto" shards the model across all visible GPUs (a 32B model spans 2x A40).
    max_pixels caps the image resolution, which sets both the map resolution and the eager-
    attention memory.

    capture=True installs the eager attention-capture hook (for capture_raw); capture=False
    is a plain sdpa load with no hook.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    global _MAX_PIXELS
    _MAX_PIXELS = max_pixels

    if capture:
        _install_capture()
        attn = {"text_config": "eager_capture", "vision_config": "sdpa"}
    else:
        attn = "sdpa"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn,
    )
    model.eval()

    if capture:
        # Report which decoder layers (if any) offloaded to CPU. Layers in the frozen search band
        # (config-derived; see readout_band) are the ones scored, so an offload there can affect
        # the attention map — flag it.
        import re as _re
        rb = readout_band(model, model_id)
        meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if meta:
            layers = sorted({int(m.group(1)) for n in meta
                             for m in [_re.search(r"layers\.(\d+)\.", n)] if m})
            band = [l for l in layers if l in rb]
            print(f"[!] {len(meta)} params offloaded (meta/CPU). "
                  f"decoder layers affected: {layers or 'none (lm_head/other)'}")
            if band:
                print(f"[!!] layers {band} ARE in the readout band {rb.start}-{rb.stop - 1} -> the "
                      "attention map may be affected; re-load fully resident.")
            else:
                print(f"     none in the readout band {rb.start}-{rb.stop - 1} -> attention map is "
                      "unaffected (only generation/logits were).")
        else:
            print("[i] no offload; model fully resident.")

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _cap_image(image):
    """Downscale (preserving aspect ratio) so total pixels <= _MAX_PIXELS.

    Bounds the vision-patch count — and therefore the LLM image-token count and the attention
    memory — regardless of the processor's default max resolution.
    """
    w, h = image.size
    if w * h <= _MAX_PIXELS:
        return image
    s = math.sqrt(_MAX_PIXELS / float(w * h))
    return image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def _prepare(processor, image, prompt, device, system=FROZEN_SYSTEM):
    """Turn (image, prompt) into model inputs. Returns inputs dict on `device`."""
    image = _cap_image(image)
    msgs = _build_messages(prompt, system)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return inputs.to(device)


def _first_device(model):
    try:
        return next(p.device for p in model.parameters() if p.device.type == "cuda")
    except StopIteration:
        return torch.device("cuda:0")


def _find_target_spans(input_ids, processor, target):
    """Locate EVERY occurrence of the target word. Returns (spans, matched): spans is a
    list of position lists, one per non-overlapping match; matched=True if the word was
    found, False if we fell back to the trailing prompt tokens (one span).
    """
    ids = input_ids.tolist()
    for variant in (" " + target, target):
        tgt = processor.tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not tgt:
            continue
        n = len(tgt)
        spans, i = [], 0
        while i <= len(ids) - n:
            if ids[i:i + n] == tgt:
                spans.append(list(range(i, i + n)))
                i += n
            else:
                i += 1
        if spans:
            return spans, True
    # Fallback: the last few non-image tokens (the tail of the user prompt).
    non_img = [i for i, t in enumerate(ids) if t != IMAGE_TOKEN_ID]
    return [non_img[-4:]], False


def capture_raw(model, processor, image, prompt, target, system=FROZEN_SYSTEM):
    """One forward pass -> RAW per-layer, per-head target/baseline attention to image patches.

    Returns fp32 arrays shaped [n_layers, n_heads, n_image_patches]:
      target_all   : attention averaged over ALL target mentions
      target_first : attention from the FIRST target mention only
      baseline     : the neutral (post-image non-target) tokens
    The baseline excludes EVERY target mention, so a minus-neutral map is target_* - baseline
    done offline. Also returns grid (grid_t, H, W), n_occ, matched. Every readout choice (layer
    band, head weighting, clip, first-vs-all mention) is a cheap CPU sweep over these arrays in
    rescore_attn.py. No generation.
    """
    device = _first_device(model)
    inputs = _prepare(processor, image, prompt, device, system)
    input_ids = inputs["input_ids"][0]

    grid_t, grid_h, grid_w = [int(x) for x in inputs["image_grid_thw"][0]]
    H, W = grid_h // SPATIAL_MERGE, grid_w // SPATIAL_MERGE
    img_pos = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0].cpu()

    spans, matched = _find_target_spans(input_ids, processor, target)
    all_tgt = [i for s in spans for i in s]
    qa = torch.tensor(all_tgt, dtype=torch.long)
    qf = torch.tensor(spans[0], dtype=torch.long)
    last_img = int(img_pos.max())
    tgt_set = set(all_tgt)
    base_pos = [i for i in range(last_img + 1, int(input_ids.shape[0])) if i not in tgt_set]
    qb = torch.tensor(base_pos, dtype=torch.long)

    _CAPTURED.clear()
    with torch.no_grad():
        model(**inputs, use_cache=False)

    Ta, Tf, B = [], [], []
    for li in sorted(_CAPTURED.keys()):
        A = _CAPTURED[li][0]                              # [heads, seq, seq], cpu fp32
        Ta.append(A[:, qa, :].mean(dim=1)[:, img_pos].numpy())
        Tf.append(A[:, qf, :].mean(dim=1)[:, img_pos].numpy())
        B.append(A[:, qb, :].mean(dim=1)[:, img_pos].numpy())
    return {
        "target_all": np.stack(Ta).astype(np.float32),   # [L, heads, H*W]
        "target_first": np.stack(Tf).astype(np.float32),
        "baseline": np.stack(B).astype(np.float32),
        "grid": (grid_t, H, W),
        "n_occ": len(spans),
        "target_matched": matched,
    }
