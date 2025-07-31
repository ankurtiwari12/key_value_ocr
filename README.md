# Invoice Field-Value Extraction with LayoutLMv3 (FATURA Dataset, Label Indices)

## Project Objective

Extract field-value pairs from invoice images using the LayoutLMv3 transformer, focusing on direct field index labels (e.g., `"1"`, `"2"`, etc.)—no business field mapping—for out-of-the-box deployment or testing on unknown schemas.

**Model:** LayoutLMv3  
**Dataset:** FATURA [https://zenodo.org/records/8261508] (layoutlm_HF_format, with raw indices or default label names as tags)  
**Use Case:** Benchmarking extraction with integer-label fields, or where explicit field-names are unknown/unmapped.

---

## Directory Structure

project-root/
│
├── data/
│ ├── fatura_raw/ # FATURA raw data (images & annotations)
│ │ ├── Images/ # Invoice images (.jpg)
│ │ └── invoices_dataset_final/
│ │ └── Annotations/
│ │ ├── layoutlm_HF_format/ # JSONs with path, words, indices/bboxes/ner_tags
│ └── fatura_hf/ # Hugging Face DatasetDict (ready for LayoutLM)
│
├── model_out/ # Fine-tuned model checkpoint directory
│ └── best/
│
├── scripts/
│ ├── 1_download_fatura.py # Download/unzip FATURA data
│ ├── 2_prepare_hf_dataset.py # Prepare dataset from layoutlm_HF_format (no mapping)
│ ├── 3_train.py # Fine-tune LayoutLMv3 on dataset
│ ├── 4_evaluate.py # Evaluate performance on test set
│ └── 5_infer_single.py # Inference: extract values from a single invoice image
│
├── requirements.txt
└── README.md

text

---

## Getting Started

### 1. Environment Setup

pip install -r requirements.txt

text

**Requirements include:**  
- Python 3.8+
- torch, torchvision  
- transformers, datasets, pillow, pytesseract, tqdm  
- Install Tesseract OCR on your OS (`sudo apt install tesseract-ocr` on Ubuntu)

### 2. Dataset Preparation

#### a. Download and Unpack FATURA

python scripts/1_download_fatura.py

text

#### b. Create HF Dataset with Label Indices

Edit `scripts/2_prepare_hf_dataset.py` to include:

LABELS = ["O"] + [str(i) for i in range(1, num_ids)]
...do not set up or use a nice_names dictionary...

text

Then build the dataset:

python scripts/2_prepare_hf_dataset.py

text

### 3. Fine-tune the Model

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python scripts/3_train.py

text

*You can reduce the number of training examples or batch size if on a small GPU.*

### 4. Evaluation

python scripts/4_evaluate.py

text
Outputs token-level metrics for each class index.

### 5. Inference on a Single Image

python scripts/5_infer_single.py path/to/invoice.jpg

text

**You will see outputs like:**

{
"1": ["Acme Corporation"],
"2": ["2025-08-01"],
...
}

text
Each key (`"1"`, `"2"`, etc.) corresponds to a field index as in your dataset.

---

## Notes

- The system does **not require or use explicit field names**; it will output label indices as specified in the input annotations.
- To turn indices into field names (e.g., `"1"` → `VENDOR`), supply a mapping file and update the `LABELS` list in `2_prepare_hf_dataset.py` before re-training.
- If you switch to business field labels, adjust only the dataset prep script; all other scripts will use the new names automatically.
- For custom visualizations, you may post-process the output as needed by referencing your `LABELS` list.

---

## References

- FATURA dataset: see project data directory and user instructions
- [LayoutLMv3 model (Hugging Face)](https://huggingface.co/microsoft/layoutlmv3-base)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

## Troubleshooting

- **"not enough values to unpack" error in inference**: Your `5_infer_single.py` should aggregate words only by the predicted label index (see latest update in group_entities function; do NOT attempt to split on "-").
- **KeyError in field mapping**: This cannot happen in this mode; all keys are simple label indices.

---
