import torch, numpy as np, evaluate
from datasets import load_from_disk
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
from tqdm.auto import tqdm


DATA_DIR = "data/fatura_hf"
MODEL_DIR = "model_out/best"

ds = load_from_disk(DATA_DIR)["test"]
processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
LABELS    = list(model.config.id2label.values()) 
model.eval()

metric = evaluate.load("seqeval")
for ex in tqdm(ds, desc="Evaluating"):
    image = Image.open(ex["image_path"]).convert("RGB")
    inputs = processor(
        image,
        ex["words"],
        boxes=ex["bboxes"],
        word_labels=None,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Only match the number of gold tokens
    labels = ex["ner_tags"]
    pred_ids = outputs.logits.argmax(-1).squeeze().tolist()
    pred_ids = pred_ids[:len(labels)]  # Trim to actual sequence length

    ids = np.array(labels) != -100
    preds = [LABELS[i] for i in np.array(pred_ids)[ids]]
    refs  = [LABELS[i] for i in np.array(labels)[ids]]
    metric.add(predictions=[preds], references=[refs])


print(metric.compute())