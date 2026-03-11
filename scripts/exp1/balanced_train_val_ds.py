import polars as pl
import os
import glob

def create_balanced_splits(madoc_dir, output_dir):
    koo_path = os.path.join(madoc_dir, "koo_madoc.parquet")
    reddit_paths = glob.glob(os.path.join(madoc_dir, "reddit_*_madoc.parquet"))
    
    print(f"Reading {len(reddit_paths) + 1} in-domain files...")
    
    def safe_load(path, platform_name):
        df = pl.read_parquet(path)
        # 1. Fallback for missing tox_bin
        if "tox_bin" not in df.columns:
            df = df.with_columns(
                (pl.col("toxicity_toxigen") * 10).floor().cast(pl.Int32).alias("tox_bin")
            )
        # 2. Select ONLY what we need to solve ShapeError (width mismatch)
        return (df.select(["content", "tox_bin"])
                .with_columns([
                    pl.lit(platform_name).alias("platform"),
                    pl.col("content").cast(pl.String),
                    pl.col("tox_bin").cast(pl.Int32).clip(0, 9)
                ]))

    all_dfs = [safe_load(koo_path, "koo")]  # ✅ FIXED: Changed from load_and_prep to safe_load
    for path in reddit_paths:
        all_dfs.append(safe_load(path, "reddit"))  # ✅ FIXED: Changed from load_and_prep to safe_load
    
    # Standardized 3-column frames will concat perfectly now
    df = pl.concat(all_dfs)

    platforms = ["koo", "reddit"]
    bins = range(10)
    val_samples_per_group = 500
    train_samples_per_group = 4500  # 4500 train + 500 val = 5000 per (platform, bin)
    
    val_frames, train_frames = [], []
    for platform in platforms:
        for b in bins:
            group = df.filter((pl.col("platform") == platform) & (pl.col("tox_bin") == b))
            if len(group) == 0: continue
            
            group = group.sample(fraction=1.0, shuffle=True, seed=42)
            val_frames.append(group.head(val_samples_per_group))
            train_frames.append(group.slice(val_samples_per_group, train_samples_per_group))

    val_df = pl.concat(val_frames).sample(fraction=1.0, shuffle=True, seed=42)
    train_df = pl.concat(train_frames).sample(fraction=1.0, shuffle=True, seed=42)

    os.makedirs(output_dir, exist_ok=True)
    val_df.write_parquet(os.path.join(output_dir, "val_2x.parquet"))
    train_df.write_parquet(os.path.join(output_dir, "train_2x.parquet"))
    print(f"✅ Balanced splits saved to {output_dir}")

if __name__ == "__main__":
    MADOC_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/madoc_data"
    OUTPUT_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/data"
    create_balanced_splits(MADOC_DIR, OUTPUT_DIR)