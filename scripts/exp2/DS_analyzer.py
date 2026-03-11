import polars as pl
import matplotlib
matplotlib.use('Agg')  # For server without display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -------------------------------------------------------
# CONFIG - UPDATE THESE PATHS
# -------------------------------------------------------
DATA_DIR = Path("/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/project/data")
OUTPUT_DIR = Path("/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/project/analysis/datasets_graphs")

# Column names
PLATFORM_COLUMN = "platform"
CONTENT_COLUMN = "content"

print("=" * 70)
print("DATASET ANALYSIS - LENGTH BALANCED DATASETS")
print("Toxicity Bins + Length + Platform Distributions")
print("=" * 70)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Load datasets
# -------------------------------------------------------
print(f"\n📂 Loading datasets from '{DATA_DIR}/'...")
available_files = {}

for filename in ["train_2x_length_balanced.parquet", "val_2x_length_balanced.parquet", "test_2x_length_balanced.parquet"]:
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"  ✅ Found: {filename}")
        available_files[filename] = filepath
    else:
        print(f"  ❌ Missing: {filename}")

if not available_files:
    print(f"\n❌ No parquet files found!")
    exit(1)

# Load individual datasets
print("\n📂 Loading datasets...")
datasets = {}

for filename, filepath in available_files.items():
    name = filename.replace("_2x_length_balanced.parquet", "").capitalize()
    datasets[name] = pl.read_parquet(filepath)
    print(f"  {name}: {datasets[name].height:,} samples")

# Also load combined dataset for overall stats
lazy_frames = [pl.scan_parquet(f) for f in available_files.values()]
combined_df = pl.concat(lazy_frames, how="diagonal").collect()
print(f"\n  Combined: {combined_df.height:,} total samples")

# Add content length to all datasets
for name, df in datasets.items():
    if CONTENT_COLUMN in df.columns:
        datasets[name] = df.with_columns([
            pl.col(CONTENT_COLUMN).str.len_chars().alias("content_length")
        ])

if CONTENT_COLUMN in combined_df.columns:
    combined_df = combined_df.with_columns([
        pl.col(CONTENT_COLUMN).str.len_chars().alias("content_length")
    ])

# -------------------------------------------------------
# STATISTICS
# -------------------------------------------------------
print("\n" + "=" * 70)
print("TOXICITY BIN STATISTICS")
print("=" * 70)

if "tox_bin" in combined_df.columns:
    bin_counts = combined_df.group_by("tox_bin").agg([
        pl.len().alias("count")
    ]).sort("tox_bin")
    print("\nOverall Toxicity Bin Distribution:")
    print(bin_counts)
    
    if PLATFORM_COLUMN in combined_df.columns:
        platform_bin_counts = combined_df.group_by([PLATFORM_COLUMN, "tox_bin"]).agg([
            pl.len().alias("count")
        ]).sort([PLATFORM_COLUMN, "tox_bin"])
        print("\nBy Platform and Toxicity Bin:")
        print(platform_bin_counts)

print("\n" + "=" * 70)
print("LENGTH STATISTICS")
print("=" * 70)

for name, df in datasets.items():
    if "content_length" in df.columns:
        stats = df.select([
            pl.col("content_length").min().alias("min"),
            pl.col("content_length").quantile(0.25).alias("q25"),
            pl.col("content_length").median().alias("median"),
            pl.col("content_length").mean().alias("mean"),
            pl.col("content_length").quantile(0.75).alias("q75"),
            pl.col("content_length").quantile(0.95).alias("q95"),
            pl.col("content_length").max().alias("max"),
        ]).to_dicts()[0]
        
        print(f"\n{name} Set:")
        print(f"  Min: {stats['min']:.0f} | Q25: {stats['q25']:.0f} | Median: {stats['median']:.0f} | Mean: {stats['mean']:.1f}")
        print(f"  Q75: {stats['q75']:.0f} | Q95: {stats['q95']:.0f} | Max: {stats['max']:.0f}")

# -------------------------------------------------------
# PLOT 1: Overall Toxicity Bin Distribution
# -------------------------------------------------------
print("\n📊 Generating plots...")

if "tox_bin" in combined_df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = combined_df["tox_bin"].drop_nulls().to_numpy()
    ax.hist(bins, bins=np.arange(11)-0.5, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title("Overall Toxicity Bin Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Toxicity Bin", fontsize=12)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_toxicity_overall.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/1_toxicity_overall.png")
    plt.close()

# -------------------------------------------------------
# PLOT 2: Toxicity by Platform (Overlapping)
# -------------------------------------------------------
if PLATFORM_COLUMN in combined_df.columns and "tox_bin" in combined_df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    platforms = combined_df[PLATFORM_COLUMN].unique().sort().to_list()
    colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']
    
    for i, p in enumerate(platforms):
        sub = combined_df.filter(pl.col(PLATFORM_COLUMN) == p)["tox_bin"].drop_nulls().to_numpy()
        ax.hist(sub, bins=np.arange(11)-0.5, alpha=0.6, label=p, 
                color=colors[i % len(colors)], edgecolor='black')
    
    ax.set_title("Platform Toxicity Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Toxicity Bin", fontsize=12)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_xticks(range(10))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_toxicity_by_platform_overlapping.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/2_toxicity_by_platform_overlapping.png")
    plt.close()

# -------------------------------------------------------
# PLOT 3: Toxicity by Platform (Stacked)
# -------------------------------------------------------
if PLATFORM_COLUMN in combined_df.columns and "tox_bin" in combined_df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_platform = []
    
    for p in platforms:
        sub = combined_df.filter(pl.col(PLATFORM_COLUMN) == p)["tox_bin"].drop_nulls().to_numpy()
        data_by_platform.append(sub)
    
    ax.hist(data_by_platform, bins=np.arange(11)-0.5, label=platforms, 
            color=colors[:len(platforms)], stacked=True, edgecolor='black', alpha=0.8)
    
    ax.set_title("Toxicity Distribution by Platform (Stacked)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Toxicity Bin", fontsize=12)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_xticks(range(10))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_toxicity_by_platform_stacked.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/3_toxicity_by_platform_stacked.png")
    plt.close()

# -------------------------------------------------------
# PLOT 4: Toxicity Box Plot by Platform
# -------------------------------------------------------
if PLATFORM_COLUMN in combined_df.columns and "tox_bin" in combined_df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_platform = []
    
    for p in platforms:
        sub = combined_df.filter(pl.col(PLATFORM_COLUMN) == p)["tox_bin"].drop_nulls().to_numpy()
        data_by_platform.append(sub)
    
    bp = ax.boxplot(data_by_platform, labels=platforms, patch_artist=True)
    
    box_colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], box_colors[:len(platforms)]):
        patch.set_facecolor(color)
    
    ax.set_title("Toxicity Spread by Platform", fontsize=14, fontweight='bold')
    ax.set_ylabel("Toxicity Bin", fontsize=12)
    ax.set_xlabel("Platform", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_toxicity_boxplot.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/4_toxicity_boxplot.png")
    plt.close()

# -------------------------------------------------------
# PLOT 5: Content Length (0-1000) by Split
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
colors_splits = ['steelblue', 'coral', 'mediumseagreen']

for i, (name, df) in enumerate(datasets.items()):
    if "content_length" in df.columns:
        lengths = df["content_length"].to_numpy()
        ax.hist(lengths, bins=50, alpha=0.5, label=name, range=(0, 1000),
                color=colors_splits[i], edgecolor='black', linewidth=0.5)

ax.set_xlabel("Content Length (characters)", fontsize=12)
ax.set_ylabel("Number of Posts", fontsize=12)
ax.set_title("Content Length (0-1000)", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "5_length_0_1000_by_split.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/5_length_0_1000_by_split.png")
plt.close()

# -------------------------------------------------------
# PLOT 6: Content Length Full Range (Log Scale)
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for i, (name, df) in enumerate(datasets.items()):
    if "content_length" in df.columns:
        lengths = df["content_length"].to_numpy()
        ax.hist(lengths, bins=100, alpha=0.5, label=name,
                color=colors_splits[i], edgecolor='black', linewidth=0.5)

ax.set_xlabel("Content Length (characters)", fontsize=12)
ax.set_ylabel("Number of Posts (log scale)", fontsize=12)
ax.set_title("Length Full Range (Log Scale)", fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "6_length_full_range_by_split.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/6_length_full_range_by_split.png")
plt.close()

# -------------------------------------------------------
# PLOT 7: Samples per Platform
# -------------------------------------------------------
if PLATFORM_COLUMN in combined_df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    platform_counts = combined_df.group_by(PLATFORM_COLUMN).agg([
        pl.len().alias("count")
    ]).sort(PLATFORM_COLUMN)
    
    platforms_list = platform_counts[PLATFORM_COLUMN].to_list()
    counts = platform_counts["count"].to_list()
    
    ax.bar(platforms_list, counts, color='teal', edgecolor='black', alpha=0.7)
    ax.set_title("Samples per Platform", fontsize=14, fontweight='bold')
    ax.set_xlabel("Platform", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "7_platform_counts.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/7_platform_counts.png")
    plt.close()

# -------------------------------------------------------
# PLOT 8: Bin Balance Verification
# -------------------------------------------------------
if "tox_bin" in combined_df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, df) in enumerate(datasets.items()):
        if "tox_bin" in df.columns:
            bin_counts = df.group_by("tox_bin").agg([
                pl.len().alias("count")
            ]).sort("tox_bin")
            
            bins_list = bin_counts["tox_bin"].to_list()
            counts = bin_counts["count"].to_list()
            
            ax.plot(bins_list, counts, marker='o', label=name, linewidth=2, markersize=8)
    
    ax.set_title("Bin Balance Verification", fontsize=14, fontweight='bold')
    ax.set_xlabel("Toxicity Bin", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xticks(range(10))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "8_bin_balance_check.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR}/8_bin_balance_check.png")
    plt.close()

# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\n📁 Generated visualization files in: {OUTPUT_DIR}")
print("  1_toxicity_overall.png - Overall toxicity bin distribution")
print("  2_toxicity_by_platform_overlapping.png - Platform comparison")
print("  3_toxicity_by_platform_stacked.png - Platform contributions")
print("  4_toxicity_boxplot.png - Platform medians/quartiles")
print("  5_length_0_1000_by_split.png - Train/Val/Test length (0-1K)")
print("  6_length_full_range_by_split.png - Train/Val/Test full range")
print("  7_platform_counts.png - Samples per platform")
print("  8_bin_balance_check.png - Toxicity bin balance verification")
print("\n" + "=" * 70)
