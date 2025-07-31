#!/usr/bin/env python
"""
Prepare Hugging-Face DatasetDict from FATURA layoutlm_HF_format annotations.

Run:
    python scripts/2_prepare_hf_dataset.py
"""

import json
import random
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Sequence, ClassLabel

# -------------------------------------------------
RAW_ROOT = Path("data/fatura_raw/invoices_dataset_final/Annotations/layoutlm_HF_format")
IMAGES_DIR = RAW_ROOT.parent.parent / "images"   # <- ensure this is the correct images folder
SPLIT_PERC = dict(train=0.8, validation=0.1, test=0.1)
# -------------------------------------------------

# ----- 1. Auto-discover all tag IDs in the dataset -----
all_tag_ids = set()
for jf in RAW_ROOT.glob("*.json"):
    rec = json.load(open(jf))
    all_tag_ids.update(rec["ner_tags"])

max_id = max(all_tag_ids)
num_ids = max_id + 1

# ----- 2. Build label names -----
# Start with default names, then overwrite with your preferred mapping
LABELS = [f"LABEL_{i}" for i in range(num_ids)]
LABELS[0] = "O"   # index 0 should always be 'Outside' or 'Other'

# Overwrite with real field names where provided
nice_names = {
    1: "B-VENDOR",   2: "I-VENDOR",
    3: "B-DATE",     4: "I-DATE",
    5: "B-TOTAL",    6: "I-TOTAL",
    7: "B-INVOICE_NO", 8: "I-INVOICE_NO",
    9: "B-SUB_TOTAL", 10: "I-SUB_TOTAL",
    11: "B-TAX",     12: "I-TAX",
    13: "B-ADDRESS", 14: "I-ADDRESS"
    # Extend this dict as needed to cover all present field types in your dataset!
}
for idx, name in nice_names.items():
    if idx < len(LABELS):
        LABELS[idx] = name

print(f"Detected {num_ids} label IDs (0 – {max_id}).")
for idx, name in enumerate(LABELS):
    print(f"  {idx}: {name}")

# ----- 3. Load all examples -----
def load_examples():
    for jf in RAW_ROOT.glob("*.json"):
        rec = json.load(open(jf))
        yield dict(
            image_path=str(IMAGES_DIR / rec["path"]),
            words=rec["words"],
            bboxes=rec["bboxes"],     # absolute pixel coords
            ner_tags=rec["ner_tags"]  # already integers
        )

examples = list(load_examples())
random.shuffle(examples)

# ----- 4. Split into train/val/test -----
n = len(examples)
n_train = int(SPLIT_PERC['train'] * n)
n_val = int(SPLIT_PERC['validation'] * n)
split_dict = {
    'train': examples[:n_train],
    'validation': examples[n_train:n_train + n_val],
    'test': examples[n_train + n_val:]
}

# ----- 5. Define Hugging Face features -----
features = Features({
    "image_path": Value("string"),
    "words":      Sequence(Value("string")),
    "bboxes":     Sequence(Sequence(Value("int64"), length=4)),
    "ner_tags":   Sequence(ClassLabel(num_classes=num_ids, names=LABELS))
})

# ----- 6. Construct DatasetDict and save -----
ds_dict = DatasetDict({
    split: Dataset.from_list(records, features=features)
    for split, records in split_dict.items()
})

out_dir = "data/fatura_hf"
ds_dict.save_to_disk(out_dir)
print(f"✅  Saved dataset to {out_dir} "
      f"({len(ds_dict['train'])} train / "
      f"{len(ds_dict['validation'])} val / "
      f"{len(ds_dict['test'])} test)")