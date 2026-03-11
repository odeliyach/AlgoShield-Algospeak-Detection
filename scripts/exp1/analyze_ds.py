import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# --- CONFIG ---
DATA_DIR = Path("/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/project/data")
TOX_COL = "tox_bin"
PLAT_COL = "platform"
CONTENT_COL = "content"

print("=" * 70)
print("GENERATING DATASET ANALYSIS PLOTS (8 PNGs)")
print("=" * 70)

# Load datasets
files = {"Train": "train_2x.parquet", "Val": "val_2x.parquet", "Test": "test_2x.parquet"}
datasets = {}
for name, filename in files.items():
    path = DATA_DIR / filename
    if path.exists():
        df = pl.read_parquet(path)
        datasets[name] = df.with_columns([pl.col(CONTENT_COL).str.len_chars().alias("content_length")])
        print(f"✅ Loaded {name}")

combined_df = pl.concat(datasets.values())
colors = ['#4682B4', '#FF7F50', '#3CB371', '#FFD700']

# 1. Overall Toxicity
plt.figure(figsize=(10,6))
plt.hist(combined_df[TOX_COL].to_list(), bins=10, color='steelblue', edgecolor='black', alpha=0.7)
plt.title("Overall Toxicity Bin Distribution"); plt.savefig("1_toxicity_overall.png")

# 2. Toxicity by Platform (Overlapping)
plt.figure(figsize=(10,6))
for i, p in enumerate(combined_df[PLAT_COL].unique()):
    sub = combined_df.filter(pl.col(PLAT_COL) == p)[TOX_COL]
    plt.hist(sub.to_list(), bins=10, alpha=0.5, label=p, color=colors[i%4])
plt.legend(); plt.title("Platform Toxicity Comparison"); plt.savefig("2_toxicity_by_platform_overlapping.png")

# 3. Toxicity by Platform (Stacked)
plt.figure(figsize=(10,6))
plats = combined_df[PLAT_COL].unique().to_list()
plt.hist([combined_df.filter(pl.col(PLAT_COL)==p)[TOX_COL].to_list() for p in plats], 
         bins=10, stacked=True, label=plats); plt.legend(); plt.savefig("3_toxicity_by_platform_stacked.png")

# 4. Toxicity Box Plot
plt.figure(figsize=(10,6))
data = [combined_df.filter(pl.col(PLAT_COL)==p)[TOX_COL].to_list() for p in plats]
plt.boxplot(data, labels=plats); plt.title("Toxicity Spread by Platform"); plt.savefig("4_toxicity_boxplot.png")

# 5. Length 0-1000
plt.figure(figsize=(10,6))
for name, df in datasets.items():
    plt.hist(df["content_length"], bins=50, range=(0,1000), alpha=0.5, label=name)
plt.legend(); plt.title("Content Length (0-1000)"); plt.savefig("5_length_0_1000_by_split.png")

# 6. Length Full Range (Log)
plt.figure(figsize=(10,6))
for name, df in datasets.items():
    plt.hist(df["content_length"], bins=100, alpha=0.5, label=name)
plt.yscale('log'); plt.legend(); plt.title("Length Full Range (Log Scale)"); plt.savefig("6_length_full_range_by_split.png")

# 7. Platform Sample Counts
plt.figure(figsize=(10,6))
counts = combined_df[PLAT_COL].value_counts()
plt.bar(counts[PLAT_COL], counts["count"], color='teal'); plt.title("Samples per Platform"); plt.savefig("7_platform_counts.png")

# 8. Bin Balance check
plt.figure(figsize=(10,6))
for name, df in datasets.items():
    b_counts = df[TOX_COL].value_counts().sort(TOX_COL)
    plt.plot(b_counts[TOX_COL], b_counts["count"], marker='o', label=name)
plt.legend(); plt.ylim(0, max(b_counts["count"])*1.2); plt.title("Bin Balance Verification"); plt.savefig("8_bin_balance_check.png")

print("Done! Generated 8 files in current directory.")
