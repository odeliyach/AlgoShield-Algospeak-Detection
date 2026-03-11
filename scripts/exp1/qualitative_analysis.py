"""
Qualitative Error Analysis — Algospeak Detection Project
=========================================================

PURPOSE:
  Run both baseline + fine-tuned models on the test set,
  find False Negatives that fine-tuning FIXED, and print
  real examples with Algospeak patterns for the Discussion section.

DOES NOT RETRAIN — only inference on the saved checkpoint.

HOW TO RUN:
  python qualitative_analysis.py \
      --checkpoint "/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/models/algospeak-finetuned/%j_run/checkpoint-22500" \
      --test_path  "/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/data/test_2x.parquet" \
      --n 500

  (set --n 0 to run on ALL 98455 test samples — takes ~30 min)

OUTPUTS:
  fn_fixed.csv             posts baseline missed, our model caught
  fn_still.csv             posts both models missed
  fp_new.csv               benign posts our model over-flagged
  qualitative_report.txt   summary + top examples ready for the paper

REQUIREMENTS:
  pip install transformers torch pandas polars
"""
import os
import argparse, re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASELINE      = "martin-ha/toxic-comment-model"
TOX_THRESHOLD = 5    # tox_bin >= 5 → toxic  (same as finetune.py)
TAU           = 0.5  # decision threshold

# Algospeak patterns from Fillies & Paschke (2024)
PATTERNS = [
    (r'\br[3e@]t[4a@]w?d[3e@]d?\b',       "leet:retarded"),
    (r'\bk[!i1|]ll\b',                     "leet:kill"),
    (r'\bkys\b',                           "abbrev:kill-yourself"),
    (r'le\s+dollar\s+bean',               "coded:le-dollar-bean"),
    (r'\bf[4a@][g9]{1,2}[o0]?[t+]?\b',    "leet:f*ggot"),
    (r'\bn[i1][g9]{1,2}[e3]?r+\b',        "leet:n-word"),
    (r'\bs[u*]ic[i1]d[e3]\b',             "leet:suicide"),
    (r'\b[a@]ssh[0o][l1][e3]\b',          "leet:a**hole"),
    (r'\bwh[0o]r[e3]\b',                  "leet:wh*re"),
    (r'\bst[u*]p[i1]d\b',                 "leet:stupid"),
]

def find_algospeak(text):
    t = text.lower()
    return [lbl for pat, lbl in PATTERNS if re.search(pat, t)]

def load_test(path, n):
    import polars as pl
    df = pl.read_parquet(path).select(["content", "tox_bin"]).to_pandas().dropna(subset=["content"])
    df["label"] = (df["tox_bin"] >= TOX_THRESHOLD).astype(int)
    if n > 0:
        tox = df[df["label"]==1].sample(min(n//2, (df["label"]==1).sum()), random_state=42)
        non = df[df["label"]==0].sample(min(n//2, (df["label"]==0).sum()), random_state=42)
        df  = pd.concat([tox, non]).reset_index(drop=True)
    return df

def predict(texts, tokenizer, model, device, batch_size=64):
    model.to(device).eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
        probs.extend(p)
        if i % (batch_size*20) == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    return [1 if p >= TAU else 0 for p in probs], [round(p,3) for p in probs]

def show(group, title, n=8):
    print(f"\n{'='*65}\n  {title}  (total={len(group)})\n{'='*65}")
    for _, r in group.sort_values("ft_prob", ascending=False).head(n).iterrows():
        txt = str(r["content"])[:200].replace('\n', ' ')
        print(f"  tox_bin={r['tox_bin']}  base={r['base_prob']:.2f} → ft={r['ft_prob']:.2f}"
              f"  len={r['char_len']}  [{r['algospeak']}]")
        print(f"  \"{txt}\"\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_path",  required=True)
    p.add_argument("--n", type=int, default=500,
                   help="samples to analyse (0 = all 98455)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df    = load_test(args.test_path, args.n)
    texts = df["content"].tolist()
    print(f"Loaded {len(df)} samples (toxic={df['label'].sum()})")

    # --- Baseline ---
    print("\n[1/2] Baseline inference...")
    bt, bm    = AutoTokenizer.from_pretrained(BASELINE), AutoModelForSequenceClassification.from_pretrained(BASELINE)
    bp, bprob = predict(texts, bt, bm, device)
    del bm

    # --- Fine-tuned ---
    print("\n[2/2] Fine-tuned inference...")
    import os

    model_path = args.checkpoint
    tokenizer_path = os.path.dirname(args.checkpoint)

    ft = AutoTokenizer.from_pretrained(tokenizer_path)
    fm = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    fp, fprob = predict(texts, ft, fm, device)
    del fm

    df["base_pred"] = bp;  df["base_prob"] = bprob
    df["ft_pred"]   = fp;  df["ft_prob"]   = fprob
    df["char_len"]  = df["content"].str.len()
    df["algospeak"] = df["content"].apply(lambda t: "; ".join(find_algospeak(str(t))) or "—")

    toxic = df["label"] == 1
    fn_fixed = df[toxic & (df["base_pred"]==0) & (df["ft_pred"]==1)].copy()
    fn_still = df[toxic & (df["base_pred"]==0) & (df["ft_pred"]==0)].copy()
    fp_new   = df[(df["label"]==0) & (df["base_pred"]==0) & (df["ft_pred"]==1)].copy()

    show(fn_fixed, "FN FIXED  (baseline missed → fine-tuned caught) ← USE THESE IN PAPER")
    show(fn_still, "FN STILL MISSED (both models failed)")
    show(fp_new,   "NEW FALSE POSITIVES (fine-tuned over-flagged)")

    n_tox = toxic.sum()
    alg   = fn_fixed[fn_fixed["algospeak"]!="—"]["algospeak"].str.split("; ").explode().value_counts()

    report = "\n".join([
        "QUALITATIVE ANALYSIS REPORT",
        "=" * 40,
        f"Samples: {len(df)}   Toxic: {n_tox}   Non-toxic: {(df['label']==0).sum()}",
        f"FN fixed  : {len(fn_fixed)} ({100*len(fn_fixed)/max(1,n_tox):.1f}% of toxic)",
        f"FN still  : {len(fn_still)} ({100*len(fn_still)/max(1,n_tox):.1f}% of toxic)",
        f"New FP    : {len(fp_new)}",
        "",
        "Algospeak patterns in FN_fixed:",
        *[f"  {k:30s} {v}" for k,v in alg.items()],
        "",
        "Length: FN_fixed mean/median char:",
        f"  fixed: {fn_fixed['char_len'].mean():.0f} / {fn_fixed['char_len'].median():.0f}",
        f"  still: {fn_still['char_len'].mean():.0f} / {fn_still['char_len'].median():.0f}",
        "",
        "TOP EXAMPLES FOR PAPER (fn_fixed, highest ft_prob):",
        *[f"  [{r['algospeak']}] base={r['base_prob']:.2f} ft={r['ft_prob']:.2f}\n  >> {str(r['content'])[:150].replace(chr(10),' ')}"
          for _, r in fn_fixed.sort_values("ft_prob", ascending=False).head(8).iterrows()],
    ])
    print("\n" + report)

    
    OUT_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/analysis/qualitative_analysis"

    fn_fixed.to_csv(f"{OUT_DIR}/fn_fixed.csv", index=False)
    fn_still.to_csv(f"{OUT_DIR}/fn_still.csv", index=False)
    fp_new.to_csv(f"{OUT_DIR}/fp_new.csv", index=False)

    with open(f"{OUT_DIR}/qualitative_report.txt", "w") as f:
        f.write(report)
    print("\n✅ Done. Files: fn_fixed.csv  fn_still.csv  fp_new.csv  qualitative_report.txt")
    print("   Open fn_fixed.csv — sort by ft_prob desc — pick best examples for Discussion section.")

if __name__ == "__main__":
    main()
