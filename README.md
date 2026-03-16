# AlgoShield: Cross-Platform Algospeak & Toxicity Detection via Domain-Adapted DistilBERT

> **Fine-tuning a compact transformer for robust detection of evasive toxic language across decentralized social media platforms.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

Automated content moderation systems face a fundamental challenge: users increasingly exploit **Algospeak** — the intentional substitution of characters, phonetic distortions, and coded slang — to evade toxicity classifiers. Standard models trained on single-platform, lexically explicit datasets fail to generalize across communities where norms and evasion strategies differ substantially.

This project addresses the challenge through **domain-adaptive fine-tuning** of a DistilBERT-based classifier on the [MADOC dataset](https://zenodo.org/records/14637314) — a multi-platform corpus spanning Reddit, Koo, Bluesky, and Voat. A **Toxicity-Balanced Stratified Sampling** strategy ensures uniform representation across fine-grained toxicity intensity bins, preventing any single severity level from dominating the training signal. Evaluated on an entirely held-out set of platforms never seen during training (Bluesky + Voat), the fine-tuned model achieves **F1 = 66.7%** and **Recall = 73.2%**, compared to **F1 = 45.1%** and **Recall = 33.2%** for the untuned baseline — a **+107% improvement in Recall** with no retraining on the target platforms.

---

## Key Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (`martin-ha/toxic-comment-model`) | 59.0% | **70.3%** | 33.2% | 45.1% |
| **Fine-tuned — Exp 1: Tox-balanced** (ours) | **62.8%** | 61.2% | **73.2%** | **66.7%** |
| Fine-tuned — Exp 2: Tox+Length-balanced† | 62.1% | 62.3% | 70.1% | 66.0% |

> *Evaluated on 98,455 out-of-domain test instances (Bluesky + Voat). In-domain validation: Acc=67.5%, Prec=64.8%, Rec=76.6%, F1=70.2%.*
>
> <sup>†</sup> **Exp 2 ablation:** Adding post-length balancing alongside toxicity-bin stratification reduces the val→test generalization gap from 3.5pt (Exp 1) to 0.3pt, at a marginal cost of 0.7pt F1. Exp 1 remains the primary reported model.

---

## Qualitative Analysis Highlights

Structured error analysis on 500 stratified test instances revealed three dominant detection patterns where fine-tuning succeeded:

### Representative FN-Fixed Examples (Cases Resolved by Fine-Tuning)

| Type | Example (Abridged) | Base Prob | FT Prob |
|---|---|---|---|
| Implicit attack | *"Are you an insufferable douchebag? Drive Rivian."* | 0.01 | 0.73 |
| Algospeak (phonetic) | *"...it looks like a pedo daycare... **gheyyy**"* | 0.17 | 0.95 |
| Leet-encoded slur | *"You need to be able to pass as a **n[---]er**."* | 0.09 | 0.89 |
| Body-shaming (statistical framing) | *"Fat Americans are liars. <10% meet activity guidelines."* | 0.27 | 0.89 |
| Predatory content | *"...children are the **sex toy payments** to the smugglers."* | 0.06 | 0.84 |
| Implicit threat | *"The other deserves **curbstomping**."* | 0.09 | 0.71 |

### Key Findings

- **93 False Negatives resolved** by fine-tuning (37.2% of all toxic test instances missed by baseline)
- **4 Algospeak substitutions** correctly flagged: 3× leet-encoded racial slurs + 1× censored gendered epithet (`wh*re`)
- **85 new False Positives** introduced — predominantly short posts where brevity removes disambiguating context
- Fine-tuned model shows stronger sensitivity to **implicit toxicity**, **dehumanizing framing**, and **community-specific hate vocabulary**

---

## Methodology

### 1. Data Construction
- **Source:** [MADOC v1.0](https://zenodo.org/records/14637314) — 236M annotated social media comments
- **Train/Val platforms:** Reddit + Koo (in-domain)
- **Test platforms:** Bluesky + Voat (entirely held-out, never seen during training)
- **Preprocessing:** URL removal, emoji stripping, minimum length filtering (≥10 chars)

### 2. Toxicity-Balanced Stratified Sampling
Toxicity scores are discretized into **10 equal-width bins**. Sampling draws an equal number of instances from each bin × platform combination, ensuring no single toxicity intensity level dominates training:

```
10 bins × 2 platforms × 4,500 train + 500 val samples
→ 90,000 train | 10,000 validation (50/50 toxic/non-toxic)
```

### 3. Model & Training
- **Base model:** [`martin-ha/toxic-comment-model`](https://huggingface.co/martin-ha/toxic-comment-model) — DistilBERT pre-trained on the Jigsaw Unintended Bias dataset
- **Architecture:** DistilBERT (6 layers, 768 hidden dim, 12 attention heads)
- **Fine-tuning hyperparameters:**

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 (linear warmup, ratio=0.06) |
| Batch size | 8 per GPU (effective: 16 with gradient accumulation) |
| Max epochs | 10 (early stopping, patience=3) |
| Actual stopping epoch | 7 (best checkpoint: epoch 4) |
| Optimizer | AdamW (weight decay=0.01) |
| Max sequence length | 512 tokens |

- **Training infrastructure:** TAU SLURM HPC cluster (~4.8 hrs, 52.1 samples/sec)

---

## Repository Structure

```
AlgoShield/
│
├── scripts/
│   ├── exp1/                              # Experiment 1 — Tox-balanced sampling
│   │   ├── finetune.py                    # Main fine-tuning pipeline
│   │   ├── balanced_train_val_ds.py       # Toxicity-bin stratified sampler
│   │   ├── qualitative_analysis.py        # Error analysis & example extraction
│   │   └── analyze_ds.py                  # Dataset statistics & visualization
│   └── exp2/                              # Experiment 2 — Tox+Length-balanced (ablation)
│       ├── finetune.py                    # Same pipeline, different dataset
│       ├── lengthXtox_DS.py               # Joint tox×length stratified sampler
│       └── DS_analyzer.py                 # Dataset diagnostic tool
│
├── data/
│   ├── exp1/                              # Preprocessed splits — Experiment 1
│   │   ├── train_2x.parquet               # 90,000 samples, tox-balanced
│   │   ├── val_2x.parquet                 # 10,000 samples, tox-balanced
│   │   └── test_2x.parquet                # 98,455 samples (Bluesky + Voat)
│   ├── exp2/                              # Preprocessed splits — Experiment 2
│   │   ├── train_2x_length_balanced.parquet
│   │   ├── val_2x_length_balanced.parquet
│   │   └── test_2x_length_balanced.parquet
│   ├── samples/                           # Qualitative error analysis (Exp 1)
│   │   ├── fn_fixed.csv                   # 93 FN cases resolved by fine-tuning
│   │   ├── fn_still.csv                   # 72 persistent FN cases
│   │   └── fp_new.csv                     # 85 new FP cases introduced
│   └── README.md                          # Instructions to download raw MADOC from Zenodo
│
├── results/
│   ├── exp1/                              # Experiment 1 results
│   │   ├── figures/                       # Training curves, confusion matrices, dataset plots
│   │   ├── qualitative_report.txt         # Summary statistics from error analysis
│   │   └── results.json                   # Full metrics (baseline + fine-tuned)
│   └── exp2/                              # Experiment 2 results (ablation)
│       ├── figures/                       # Training curves, confusion matrices
│       └── results.json                   # Full metrics (baseline + fine-tuned)
│
├── README.md
├── requirements.txt
└── LICENSE
```

> **Note:** Raw MADOC parquet files are not included (dataset available at [Zenodo](https://zenodo.org/records/14637314)). Model weights available at [HuggingFace Hub — link TBD].

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- CUDA-capable GPU recommended (training); CPU sufficient for inference

### 1. Clone the repository
```bash
git clone https://github.com/odeliyach/AlgoShield-Algospeak-Detection.git
cd AlgoShield
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the MADOC dataset
```bash
# Download from Zenodo and place parquet files in data/madoc/
# Dataset DOI: https://doi.org/10.5281/zenodo.14637314
```

### 4. Run the balanced sampling pipeline
```bash
python scripts/balanced_train_val_ds.py \
  --madoc_dir data/madoc/ \
  --output_dir data/balanced/
```

### 5. Fine-tune the model
```bash
python scripts/finetune.py \
  --train_path data/balanced/train.parquet \
  --val_path data/balanced/val.parquet \
  --test_path data/madoc/test.parquet \
  --output_dir models/algoshield-finetuned/
```

### 6. Run qualitative analysis
```bash
python scripts/qualitative_analysis.py \
  --model_dir models/algoshield-finetuned/checkpoint-best/ \
  --test_path data/madoc/test.parquet \
  --output_dir results/
```

---

## Links

| Resource | Link |
|---|---|
| 📄 Full Paper (ACL format) | *[Link to paper — TBD upon publication]* |
| 🗃️ MADOC Dataset | [Zenodo — DOI: 10.5281/zenodo.14637314](https://zenodo.org/records/14637314) |
| 🤗 Base Model | [martin-ha/toxic-comment-model](https://huggingface.co/martin-ha/toxic-comment-model) |
| 🤗 Fine-tuned Model Weights | *[HuggingFace Hub — TBD]* |

---

## Citation

If you use this work, please cite:
```bibtex
@misc{algoshield2026,
  title        = {AlgoShield: Cross-Platform Algospeak Detection via Domain-Adapted DistilBERT},
  author       = {Charitonova, Odeliya and Loshevsky, Alin and Pernik, Lior},
  year         = {2026},
  howpublished = {NLP Course Final Project, Tel Aviv University},
  note         = {GitHub: https://github.com/odeliyach/AlgoShield-Algospeak-Detection}
}
```

If you use the MADOC dataset, please cite the original dataset:
```bibtex
@dataset{fillies_paschke_2024_madoc,
  author    = {Fillies, Sebastian and Paschke, Adrian},
  title     = {The MADOC Dataset: Multi-Platform Aggregated Dataset of Online Communities},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.14637314},
  url       = {https://doi.org/10.5281/zenodo.14637314}
}
```

---

## Authors

Odeliya Charitonova · Alin Loshevsky · Lior Pernik  
*NLP Final Project — Tel Aviv University, 2026*  
*Supervisor: Dr. Tal Wagner*
