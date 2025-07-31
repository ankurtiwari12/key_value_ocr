#!/usr/bin/env python
"""
Inference script — give an invoice image, receive extracted field–value pairs.

Usage
-----
python infer_single.py path/to/invoice.jpg
"""

import sys
import json
import torch
import numpy as np
from PIL import Image
import pytesseract
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)

# ───────────────── CONFIGURATION ─────────────────
MODEL_DIR      = "model_out/best"   # path to your fine-tuned model
CONF_THRESHOLD = 40                 # ignore OCR tokens below this confidence
MAX_LEN        = 512                # LayoutLMv3 maximum sequence length
# ─────────────────────────────────────────────────


def ocr_with_pytesseract(img_path):
    """
    Run Tesseract OCR and return:
      words : list[str]
      boxes : list[[x0,y0,x1,y1]] (absolute pixel coords)
    """
    data = pytesseract.image_to_data(
        Image.open(img_path),
        output_type=pytesseract.Output.DICT,
        config="--psm 6",  # treat the image as a single text block
    )

    words, boxes = [], []
    for text, conf, x, y, w, h in zip(
        data["text"],
        data["conf"],
        data["left"],
        data["top"],
        data["width"],
        data["height"],
    ):
        text = text.strip()
        if text and int(conf) >= CONF_THRESHOLD:
            words.append(text)
            boxes.append([x, y, x + w, y + h])

    return words, boxes


def group_entities(words, preds, id2label):
    # Handles labels as simple ints or strings (e.g., "0", "1", ...)
    id2label_int = {int(k): v for k, v in id2label.items()}
    entities = {}
    for word, pred_id in zip(words, preds):
        label = id2label_int.get(pred_id, "O")
        if label == "O":
            continue
        entities.setdefault(label, []).append(word)
    # Join consecutive tokens for each field
    entities = {k: [" ".join(v)] for k, v in entities.items()}
    return entities



def main(img_path):
    # 1. OCR
    words, boxes = ocr_with_pytesseract(img_path)
    if not words:
        print("No OCR words detected above confidence threshold.")
        return

    # 2. Load processor & model
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
    model     = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # 3. Truncate overly long inputs (LayoutLMv3 max 512 tokens)
    if len(words) > MAX_LEN:
        words = words[:MAX_LEN]
        boxes = boxes[:MAX_LEN]

    # 4. Encode
    image   = Image.open(img_path).convert("RGB")
    encoded = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )

    # 5. Predict
    with torch.no_grad():
        logits   = model(**encoded).logits
    pred_ids = logits.argmax(-1).squeeze().tolist()

    # 6. Post-process
    entities = group_entities(
        words,
        pred_ids,
        model.config.id2label,  # pass mapping for cleaner code
    )

    # 7. Present results
    if entities:
        print(json.dumps(entities, indent=2, ensure_ascii=False))
    else:
        print("No entities extracted.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_single.py <invoice_image>")
        sys.exit(1)
    main(sys.argv[1])