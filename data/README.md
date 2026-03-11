# Data

The raw MADOC dataset files are **not included** in this repository due to their size.

## Download Instructions

1. Go to the MADOC dataset page on Zenodo: [https://zenodo.org/records/XXXXXXX](https://zenodo.org/records/XXXXXXX)
2. Download the following parquet files and place them in a `data/madoc/` folder:
   - `reddit_*_madoc.parquet` (multiple files)
   - `koo_madoc.parquet`
   - Bluesky and Voat files for the test set

## Preprocessed Splits

The balanced train/val/test splits used in our experiments are included in this repository:

| File | Folder | Experiment | Description |
|------|--------|-----------|-------------|
| `train_2x.parquet` | `data/exp1/` | Exp 1 | 90,000 samples, tox-balanced |
| `val_2x.parquet` | `data/exp1/` | Exp 1 | 10,000 samples, tox-balanced |
| `test_2x.parquet` | `data/exp1/` | Both | 98,455 samples (Bluesky + Voat) |
| `train_2x_length_balanced.parquet` | `data/exp2/` | Exp 2 | 90,000 samples, tox+length-balanced |
| `val_2x_length_balanced.parquet` | `data/exp2/` | Exp 2 | 10,000 samples, tox+length-balanced |
| `test_2x_length_balanced.parquet` | `data/exp2/` | Exp 2 | 98,455 samples (Bluesky + Voat) |

To regenerate the splits from scratch, run:
```bash
# Experiment 1
python scripts/exp1/balanced_train_val_ds.py --madoc_dir data/madoc/ --output_dir data/

# Experiment 2
python scripts/exp2/lengthXtox_DS.py --madoc_dir data/madoc/ --output_dir data/
```

## Qualitative Samples

The `samples/` folder contains the error analysis outputs from Experiment 1:
- `fn_fixed.csv` — 93 toxic posts the baseline missed but the fine-tuned model caught
- `fn_still.csv` — 72 toxic posts both models missed
- `fp_new.csv` — 85 benign posts incorrectly flagged by the fine-tuned model
