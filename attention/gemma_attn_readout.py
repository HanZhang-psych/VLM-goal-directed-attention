"""
gemma_attn_readout.py — load Gemma-4-26B-A4B-it (HF, eager attention) and capture its attention
weights for the attention arm. Used by run_attn_gemma.py.

load(capture=True) installs an attention-capture hook and returns (model, processor).
capture_raw() runs one forward pass over an image + prompt and returns the raw per-layer,
per-head attention from the target word (and a neutral baseline) to the image patches, in the
same .npz schema attention/rescore_attn.py reads. Image tokens come from mm_token_type_ids;
the pooled (H, W) grid from the official aspect-preserving sizing.
"""
import importlib
import math

import numpy as np
import torch
from PIL import Image

from config import FROZEN_SYSTEM, MODEL_IDS

MODEL_ID = MODEL_IDS["gemma"]

IMAGE_TOKEN_ID = None        # from processor.image_token_id (NOT model config) in load()
SLIDING_WINDOW = None
_IMGPROC = {}                # patch_size / pooling_kernel_size / max_soft_tokens, from processor
_MAX_PIXELS = 768 * 768
_CAPTURED = {}
_REF = [None]                # lazily-resolved reference eager_attention_forward


def _build_messages(prompt):
    # Gemma expects image BEFORE text; FROZEN_SYSTEM folded into the user turn.
    text = (FROZEN_SYSTEM + "\n\n" + prompt) if FROZEN_SYSTEM else prompt
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


def _install_capture():
    """Register 'eager_capture': same math as Gemma's eager path, but stash TEXT-decoder
    attention only. Filter requires 'Text' in the class name (Gemma4TextAttention) — this
    excludes Gemma4VisionAttention AND Gemma4AudioAttention. The reference
    eager_attention_forward is resolved lazily from the module that defines the attention
    class, so registration works before the model is loaded."""
    def eager_capture(module, query, key, value, attention_mask, **kw):
        if _REF[0] is None:
            _REF[0] = importlib.import_module(type(module).__module__).eager_attention_forward
        attn_output, attn_weights = _REF[0](module, query, key, value, attention_mask, **kw)
        if attn_weights is not None and "Text" in module.__class__.__name__:
            layer_idx = getattr(module, "layer_idx", len(_CAPTURED))
            _CAPTURED[layer_idx] = attn_weights.detach().to("cpu", torch.float32)
        return attn_output, attn_weights

    try:
        from transformers import AttentionInterface
        AttentionInterface.register("eager_capture", eager_capture)
    except Exception:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager_capture"] = eager_capture


def load(model_id=MODEL_ID, device_map="auto", max_pixels=768 * 768, capture=True):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    global _MAX_PIXELS, IMAGE_TOKEN_ID, SLIDING_WINDOW
    _MAX_PIXELS = max_pixels
    if capture:
        _install_capture()
        attn = {"text_config": "eager_capture", "vision_config": "sdpa"}
    else:
        attn = "sdpa"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device_map, attn_implementation=attn)
    model.eval()

    # Image token id lives on the processor/tokenizer in Gemma-4 (NOT the model config).
    IMAGE_TOKEN_ID = int(getattr(processor, "image_token_id",
                                 getattr(processor.tokenizer, "image_token_id")))
    ip = processor.image_processor
    _IMGPROC.update(patch_size=int(ip.patch_size),
                    pooling_kernel_size=int(ip.pooling_kernel_size),
                    max_soft_tokens=int(ip.max_soft_tokens))
    tc = getattr(model.config, "text_config", model.config)
    SLIDING_WINDOW = int(getattr(tc, "sliding_window", 0) or 0)
    print(f"[gemma-attn] image_token_id={IMAGE_TOKEN_ID}  sliding_window={SLIDING_WINDOW}  "
          f"text_layers={num_text_layers(model)}  imgproc={_IMGPROC}", flush=True)
    return model, processor


def num_text_layers(model):
    tc = getattr(model.config, "text_config", model.config)
    return int(getattr(tc, "num_hidden_layers", 0))


def _cap_image(image):
    w, h = image.size
    if w * h <= _MAX_PIXELS:
        return image
    s = math.sqrt(_MAX_PIXELS / float(w * h))
    return image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def _first_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare(processor, image, prompt, device):
    """(image, prompt) -> model inputs (image-first ordering)."""
    image = _cap_image(image).convert("RGB")
    messages = _build_messages(prompt)
    for c in messages[0]["content"]:
        if c.get("type") == "image":
            c["image"] = image
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    return inputs.to(device), image


def _image_grid_hw(image):
    """(H, W) of Gemma-4's pooled image-token grid for this image, via the OFFICIAL
    aspect-preserving sizing function and the processor's own parameters. Row-major
    (Gemma4VisionPooler: slot = x + W*y), so reshape(H, W) is correct."""
    from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
    p, k, mst = _IMGPROC["patch_size"], _IMGPROC["pooling_kernel_size"], _IMGPROC["max_soft_tokens"]
    w, h = image.size
    th, tw = get_aspect_ratio_preserving_size(height=h, width=w, patch_size=p,
                                              max_patches=mst * k * k, pooling_kernel_size=k)
    return (th // p) // k, (tw // p) // k


def _find_target_spans(input_ids, processor, target):
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
    non_img = [i for i, t in enumerate(ids) if t != IMAGE_TOKEN_ID]
    return [non_img[-4:]], False


def capture_raw(model, processor, image, prompt, target):
    """One forward pass -> RAW per-layer/per-head target/baseline attention to image patches.
    Returns the same dict shape as attention/qwen_attn_readout.capture_raw. Grid is aspect-preserving (H, W)."""
    device = _first_device(model)
    inputs, capped = _prepare(processor, image, prompt, device)
    input_ids = inputs["input_ids"][0]

    # Image-token positions: mm_token_type_ids==1 (canonical for Gemma-4), fallback to token id.
    if "mm_token_type_ids" in inputs:
        img_pos = (inputs["mm_token_type_ids"][0] == 1).nonzero(as_tuple=True)[0].cpu()
    else:
        img_pos = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0].cpu()
    n_img = int(len(img_pos))
    H, W = _image_grid_hw(capped)
    assert H * W == n_img, (
        f"Gemma image-token count {n_img} != computed grid {H}x{W} "
        f"(image {capped.size}, imgproc {_IMGPROC}) — sizing mismatch; do not score.")
    if SLIDING_WINDOW and int(input_ids.shape[0]) > SLIDING_WINDOW:
        print(f"[gemma-attn][!] seq_len {int(input_ids.shape[0])} > sliding_window {SLIDING_WINDOW}: "
              "sliding layers are windowed; check target-image distance.", flush=True)

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
        "grid": (1, H, W),
        "n_occ": len(spans),
        "target_matched": matched,
    }
