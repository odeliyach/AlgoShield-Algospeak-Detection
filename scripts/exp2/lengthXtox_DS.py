import polars as pl
import os
import glob

# Length bins configuration
LENGTH_BINS = [100, 200, 300, 500]  # Creates 5 bins: 0-100, 100-200, 200-300, 300-500, 500+
LENGTH_LABELS = ["0-100", "100-200", "200-300", "300-500", "500+"]

def sample_with_length_balance(df, target_total, seed):
    """
    Sample from df with balanced representation across length bins.
    
    Strategy:
    1. Count how many length bins have data
    2. Target samples_per_length_bin = target_total / num_length_bins
    3. Sample proportionally from each length bin
    4. If shortage, randomly sample more from entire df
    """
    len_counts = (
        df.group_by("len_bin")
        .agg(pl.len().alias("count"))
    )
    
    n_len_bins = len(len_counts)
    if n_len_bins == 0:
        return df.head(0)  # Return empty dataframe
    
    target_per_len_bin = target_total // n_len_bins
    
    samples = []
    total_sampled = 0
    
    # Sample from each length bin
    for row in len_counts.iter_rows(named=True):
        len_bin = row["len_bin"]
        available = row["count"]
        want = min(target_per_len_bin, available)
        
        if want > 0:
            sample = (
                df.filter(pl.col("len_bin") == len_bin)
                .sample(n=want, seed=seed + hash(len_bin) % 100)
            )
            samples.append(sample)
            total_sampled += want
    
    # If we didn't reach target, randomly sample more from the entire df
    if total_sampled < target_total and samples:
        shortage = target_total - total_sampled
        combined = pl.concat(samples)
        remaining = df.join(combined, on=df.columns, how="anti")
        
        if len(remaining) > 0:
            extra = remaining.sample(
                n=min(shortage, len(remaining)),
                seed=seed + 1000
            )
            samples.append(extra)
            total_sampled += len(extra)
    
    # Combine all samples
    if samples:
        return pl.concat(samples)
    else:
        # Fallback: if no length-balanced sampling worked, just sample randomly
        return df.sample(n=min(target_total, len(df)), seed=seed)


def safe_load(path, platform_name):
    """Load and prepare a parquet file with toxicity and length bins."""
    df = pl.read_parquet(path)
    
    # 1. Fallback for missing tox_bin
    if "tox_bin" not in df.columns:
        df = df.with_columns(
            (pl.col("toxicity_toxigen") * 10).floor().cast(pl.Int32).alias("tox_bin")
        )
    
    # 2. Add length bin
    df = df.with_columns([
        pl.col("content").str.len_chars().alias("content_length")
    ]).with_columns([
        pl.col("content_length").cut(
            breaks=LENGTH_BINS,
            labels=LENGTH_LABELS
        ).alias("len_bin")
    ])
    
    # 3. Select needed columns
    return (df.select(["content", "tox_bin", "len_bin"])
            .with_columns([
                pl.lit(platform_name).alias("platform"),
                pl.col("content").cast(pl.String),
                pl.col("tox_bin").cast(pl.Int32).clip(0, 9)
            ]))


def create_all_balanced_datasets(madoc_dir, output_dir):
    """
    Create train, val, and test datasets with dual balancing (toxicity + length).
    
    Train/Val: Reddit + Koo (in-domain)
    Test: Bluesky + Voat (out-of-domain)
    """
    print("="*70)
    print("CREATING BALANCED DATASETS WITH DUAL BALANCING")
    print("Balancing factors: Toxicity bins (0-9) + Length bins (5 categories)")
    print("="*70)
    
    # ========================================================================
    # PART 1: TRAIN & VAL (Reddit + Koo)
    # ========================================================================
    print("\n[1/2] Creating TRAIN & VAL sets (Reddit + Koo)...")
    
    koo_path = os.path.join(madoc_dir, "koo_madoc.parquet")
    reddit_paths = glob.glob(os.path.join(madoc_dir, "reddit_*_madoc.parquet"))
    
    print(f"   Loading {len(reddit_paths) + 1} in-domain files...")
    
    all_dfs = [safe_load(koo_path, "koo")]
    for path in reddit_paths:
        all_dfs.append(safe_load(path, "reddit"))
    
    # Concatenate all in-domain data
    df = pl.concat(all_dfs)
    print(f"   Total in-domain samples: {len(df):,}")
    
    platforms = ["koo", "reddit"]
    tox_bins = range(10)
    
    # Target samples per (platform, tox_bin) combination
    # Total: 5,000 per group, split 500 val + 4,500 train
    val_target_per_tox_bin = 500
    train_target_per_tox_bin = 4500
    
    val_frames, train_frames = [], []
    
    print("\n   Sampling with DUAL BALANCING (toxicity + length)...")
    
    for platform in platforms:
        for tox_bin in tox_bins:
            # Get all samples for this (platform, tox_bin) combination
            group = df.filter(
                (pl.col("platform") == platform) & 
                (pl.col("tox_bin") == tox_bin)
            )
            
            if len(group) == 0:
                print(f"     ⚠️ No data for {platform} tox_bin={tox_bin}")
                continue
            
            # Shuffle the group
            group = group.sample(fraction=1.0, shuffle=True, seed=42)
            
            # Sample VAL set with length balancing
            val_sample = sample_with_length_balance(
                group, 
                val_target_per_tox_bin, 
                seed=42 + tox_bin
            )
            val_frames.append(val_sample)
            
            # Remove val samples from group for train sampling
            remaining = group.join(val_sample, on=group.columns, how="anti")
            
            # Sample TRAIN set with length balancing from remaining data
            train_sample = sample_with_length_balance(
                remaining, 
                train_target_per_tox_bin, 
                seed=42 + tox_bin + 100
            )
            train_frames.append(train_sample)
            
            print(f"     ✓ {platform:7s} tox_bin={tox_bin}: val={len(val_sample):4d}, train={len(train_sample):5d}")
    
    # Combine and shuffle final datasets
    val_df = pl.concat(val_frames).sample(fraction=1.0, shuffle=True, seed=42)
    train_df = pl.concat(train_frames).sample(fraction=1.0, shuffle=True, seed=42)
    
    # Remove the len_bin and content_length columns before saving (if they exist)
    cols_to_drop = [c for c in ["len_bin", "content_length"] if c in val_df.columns]
    if cols_to_drop:
        val_df = val_df.drop(cols_to_drop)
    
    cols_to_drop = [c for c in ["len_bin", "content_length"] if c in train_df.columns]
    if cols_to_drop:
        train_df = train_df.drop(cols_to_drop)
    
    print(f"\n   ✅ Train: {len(train_df):,} samples")
    print(f"   ✅ Val:   {len(val_df):,} samples")
    
    # ========================================================================
    # PART 2: TEST (Bluesky + Voat)
    # ========================================================================
    print("\n[2/2] Creating TEST set (Bluesky + Voat)...")
    
    # Load out-of-domain platforms
    bluesky_path = os.path.join(madoc_dir, "bluesky_madoc.parquet")
    voat_paths = glob.glob(os.path.join(madoc_dir, "voat_*_madoc.parquet"))
    
    print(f"   Loading {1 + len(voat_paths)} out-of-domain files...")
    
    bluesky_df = safe_load(bluesky_path, "bluesky")
    voat_df = pl.concat([safe_load(p, "voat") for p in voat_paths])
    
    print(f"   Bluesky samples: {len(bluesky_df):,}")
    print(f"   Voat samples:    {len(voat_df):,}")
    
    test_frames = []
    
    print("\n   Sampling with DUAL BALANCING (toxicity + length)...")
    
    # Sample 5,000 per (platform, tox_bin) with length balancing
    for df_data, platform in [(bluesky_df, "bluesky"), (voat_df, "voat")]:
        for tox_bin in range(10):
            bin_subset = df_data.filter(pl.col("tox_bin") == tox_bin)
            
            if len(bin_subset) == 0:
                print(f"     ⚠️ No data for {platform} tox_bin={tox_bin}")
                continue
            
            # Sample with length balancing
            sample = sample_with_length_balance(
                bin_subset, 
                target_total=5000, 
                seed=1337 + tox_bin  # Different seed for test set
            )
            test_frames.append(sample)
            print(f"     ✓ {platform:7s} tox_bin={tox_bin}: {len(sample):5d} samples")
    
    # Combine and shuffle
    test_df = pl.concat(test_frames).sample(fraction=1.0, shuffle=True, seed=1337)
    
    # Remove length bin columns before saving (if they exist)
    cols_to_drop = [c for c in ["len_bin", "content_length"] if c in test_df.columns]
    if cols_to_drop:
        test_df = test_df.drop(cols_to_drop)
    
    print(f"\n   ✅ Test: {len(test_df):,} samples")
    
    # ========================================================================
    # SAVE ALL DATASETS
    # ========================================================================
    print("\n" + "="*70)
    print("SAVING DATASETS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_2x_length_balanced.parquet")
    val_path = os.path.join(output_dir, "val_2x_length_balanced.parquet")
    test_path = os.path.join(output_dir, "test_2x_length_balanced.parquet")
    
    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)
    test_df.write_parquet(test_path)
    
    print(f"✅ Train saved: {train_path}")
    print(f"   {len(train_df):,} samples")
    
    print(f"\n✅ Val saved:   {val_path}")
    print(f"   {len(val_df):,} samples")
    
    print(f"\n✅ Test saved:  {test_path}")
    print(f"   {len(test_df):,} samples")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Train (in-domain):     {len(train_df):,} samples (Reddit + Koo)")
    print(f"Val (in-domain):       {len(val_df):,} samples (Reddit + Koo)")
    print(f"Test (out-of-domain):  {len(test_df):,} samples (Bluesky + Voat)")
    print(f"\nBalancing: Toxicity bins (0-9) × Length bins ({len(LENGTH_LABELS)})")
    print(f"Length categories: {', '.join(LENGTH_LABELS)}")
    print("\n🎉 All datasets created successfully!")


if __name__ == "__main__":
    MADOC_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/madoc_data"
    OUTPUT_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/project/data"
    
    create_all_balanced_datasets(MADOC_DIR, OUTPUT_DIR)
