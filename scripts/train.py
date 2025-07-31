#!/usr/bin/env python
"""
Fine-tune LayoutLMv3 on the FATURA Hugging-Face dataset (data/fatura_hf)
with memory-saving tweaks for small GPUs (≈3–4 GB).

Run:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0 python scripts/3_train.py
"""

import os, torch
from datasets import load_from_disk
from transformers import (LayoutLMv3Processor,
                          LayoutLMv3ForTokenClassification,
                          TrainingArguments, Trainer)
from transformers import default_data_collator
from PIL import Image
data_collator = lambda features: {k: v.to(torch.float16) if v.dtype==torch.float else v
                                  for k, v in default_data_collator(features).items()}

# ───────── CONFIG ─────────
DATA_DIR   = "data/fatura_hf"
MODEL_NAME = "microsoft/layoutlmv3-base"
OUT_DIR    = "model_out"
MAX_LEN    = 320          # shorter sequence → lower VRAM
BSZ        = 1            # per-device batch size
GRAD_ACC   = 4            # effective batch = 4
EPOCHS     = 8
LR         = 1e-5
# ──────────────────────────


# 1. Load dataset & derive label names dynamically
ds        = load_from_disk(DATA_DIR)
LABELS    = ds["train"].features["ner_tags"].feature.names
num_labels = len(LABELS)

# 2. Processor & model (bf16 > fp16 if your GPU supports it)
processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=dict(enumerate(LABELS)),
    label2id={v: k for k, v in enumerate(LABELS)},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# model.gradient_checkpointing_enable()      # recompute activations to save RAM

# 3. Encode function
def encode(batch):
    images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
    enc = processor(
        images,
        batch["words"],
        boxes=batch["bboxes"],
        word_labels=batch["ner_tags"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return {k: v for k, v in enc.items()}

encoded = ds.map(encode, batched=True, remove_columns=ds["train"].column_names)
encoded.set_format(type="torch")

# 4. TrainingArguments with lightweight optimiser
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BSZ,
    per_device_eval_batch_size=BSZ,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=50,
    save_strategy="epoch",
    optim="paged_adamw_8bit",     # bitsandbytes 8-bit optimiser
    fp16=not torch.cuda.is_bf16_supported(),  # use fp16 if bf16 unavailable
    report_to="none",
)

# 5. Simple seqeval metric
def compute_metrics(pred):
    import numpy as np, evaluate
    metric = evaluate.load("seqeval")
    preds  = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    true_preds, true_labels = [], []
    for p, l in zip(preds, labels):
        mask = l != -100
        true_preds.append([LABELS[i] for i in p[mask]])
        true_labels.append([LABELS[i] for i in l[mask]])
    res = metric.compute(predictions=true_preds, references=true_labels)
    return {"precision": res["overall_precision"],
            "recall":    res["overall_recall"],
            "f1":        res["overall_f1"]}

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    data_collator=data_collator,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"{OUT_DIR}/best")
processor.save_pretrained(f"{OUT_DIR}/best")
print("✅  Training complete. Model saved to", f"{OUT_DIR}/best")