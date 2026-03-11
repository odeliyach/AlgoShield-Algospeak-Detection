import argparse, json, os, torch, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Hardcoded hyperparameters
# ---------------------------------------------------------------------------
SEED                        = 42
LEARNING_RATE               = 2e-5
EPOCHS                      = 10     # EarlyStopping will cut this short if val F1 plateaus
BATCH_SIZE                  = 8      # per-GPU; effective batch = 8 * 2 = 16
GRADIENT_ACCUMULATION_STEPS = 2
MAX_LENGTH                  = 512
BASE_MODEL                  = "martin-ha/toxic-comment-model"
TOX_BIN_THRESHOLD           = 5      # tox_bin >= 5 -> toxic, < 5 -> non-toxic

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_parquet(path):
    """
    Load a MADOC parquet file and binarize tox_bin into a 0/1 label.
    tox_bin >= TOX_BIN_THRESHOLD  ->  label = 1 (toxic)
    tox_bin <  TOX_BIN_THRESHOLD  ->  label = 0 (non-toxic)
    """
    print(f"  Loading {path} ...")
    df = pd.read_parquet(path, columns=["content", "tox_bin"])
    df = df.dropna(subset=["content", "tox_bin"])
    df = df[df["content"].str.strip().ne("")]
    df["labels"] = (df["tox_bin"] >= TOX_BIN_THRESHOLD).astype(int)
    n, n_tox = len(df), df["labels"].sum()
    print(f"    -> {n:,} rows | toxic={n_tox:,} ({n_tox/n*100:.1f}%) | non-toxic={n-n_tox:,} ({(n-n_tox)/n*100:.1f}%)")
    return df.reset_index(drop=True)

def to_hf_dataset(df):
    return Dataset.from_dict({"text": df["content"].tolist(), "labels": df["labels"].tolist()})

def tokenize(ds, tokenizer):
    def _tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """F1 (binary, toxic class) is primary metric due to class imbalance."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "f1_macro":  f1_score(labels, preds, average="macro", zero_division=0),
    }

def print_report(labels, preds, title):
    report = classification_report(labels, preds, target_names=["non-toxic", "toxic"], output_dict=True)
    sep = "=" * 55
    print(f"\n{sep}\n  {title}\n{sep}")
    print(classification_report(labels, preds, target_names=["non-toxic", "toxic"]))
    return report

def extract_metrics(report):
    return {
        "f1":        report["toxic"]["f1-score"],
        "precision": report["toxic"]["precision"],
        "recall":    report["toxic"]["recall"],
        "accuracy":  report["accuracy"],
        "f1_macro":  report["macro avg"]["f1-score"],
    }

# ---------------------------------------------------------------------------
# Paper figure helpers
# ---------------------------------------------------------------------------

def save_confusion_matrix(labels, preds, title, save_path):
    """
    Save confusion matrix PNG.
    False Negatives (bottom-left) are key for this project:
    they represent Algospeak that slipped through undetected.
    """
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ["non-toxic", "toxic"]
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")

def save_training_curves(log_history, save_path):
    """Save training loss + val F1 per epoch for the paper."""
    eval_entries  = [e for e in log_history if "eval_f1" in e]
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    if not eval_entries:
        print("  No eval entries - skipping training curves.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([e["step"] for e in train_entries],
                 [e["loss"] for e in train_entries], color="steelblue", linewidth=1.5)
    axes[0].set_xlabel("Training step"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss"); axes[0].grid(True, alpha=0.3)
    epochs = [e["epoch"] for e in eval_entries]
    axes[1].plot(epochs, [e["eval_f1"] for e in eval_entries],
                 marker="o", color="darkorange", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation F1 (toxic)")
    axes[1].set_title("Validation F1 per epoch")
    axes[1].set_xticks(epochs); axes[1].grid(True, alpha=0.3)
    plt.suptitle("Training dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    src = args.reading_params_path or BASE_MODEL
    out = args.writing_params_path
    os.makedirs(out, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Load data
    # train + val : Reddit + Koo (in-domain)
    # test        : Bluesky + Voat (out-of-domain, never seen in training)
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data ...")
    train_df = load_parquet(args.train_path)
    val_df   = load_parquet(args.val_path)
    test_df  = load_parquet(args.test_path)

    # ------------------------------------------------------------------
    # STEP 2: Load tokenizer + model
    # ------------------------------------------------------------------
    print(f"\n[2/5] Loading model from: {src}")
    tokenizer = AutoTokenizer.from_pretrained(src)
    model     = AutoModelForSequenceClassification.from_pretrained(src)
    print(f"  Label mapping: {model.config.id2label}")

    # ------------------------------------------------------------------
    # STEP 3: Tokenize all splits
    # ------------------------------------------------------------------
    print("\n[3/5] Tokenizing ...")
    train_ds = tokenize(to_hf_dataset(train_df), tokenizer)
    val_ds   = tokenize(to_hf_dataset(val_df),   tokenizer)
    test_ds  = tokenize(to_hf_dataset(test_df),  tokenizer)

    # ------------------------------------------------------------------
    # STEP 4: Baseline evaluation (pretrained model, no fine-tuning)
    # These numbers go in the report as the baseline to beat.
    # ------------------------------------------------------------------
    print("\n[4/5] Baseline evaluation ...")
    eval_args = TrainingArguments(
        output_dir=os.path.join(out, "_baseline_eval"),
        per_device_eval_batch_size=16,
        report_to="none", seed=SEED,
    )
    bt = Trainer(model=model, args=eval_args, compute_metrics=compute_metrics)
    bl_val_preds  = np.argmax(bt.predict(val_ds).predictions,  axis=-1)
    bl_test_preds = np.argmax(bt.predict(test_ds).predictions, axis=-1)
    bl_val_report  = print_report(val_df["labels"].tolist(),  bl_val_preds,  "BASELINE -- val (in-domain)")
    bl_test_report = print_report(test_df["labels"].tolist(), bl_test_preds, "BASELINE -- test (out-of-domain)")

    print("\n  Saving baseline confusion matrices ...")
    save_confusion_matrix(val_df["labels"].tolist(),  bl_val_preds,
                          "Baseline -- val",  os.path.join(out, "confusion_baseline_val.png"))
    save_confusion_matrix(test_df["labels"].tolist(), bl_test_preds,
                          "Baseline -- test", os.path.join(out, "confusion_baseline_test.png"))

    results = {
        "hyperparameters": {
            "seed": SEED, "lr": LEARNING_RATE, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "grad_accum": GRADIENT_ACCUMULATION_STEPS,
            "tox_bin_threshold": TOX_BIN_THRESHOLD,
        },
        "baseline": {
            "val":  extract_metrics(bl_val_report),
            "test": extract_metrics(bl_test_report),
        },
    }

    # ------------------------------------------------------------------
    # STEP 5: Fine-tuning
    # ------------------------------------------------------------------
    if not args.eval_only:
        print("\n[5/5] Fine-tuning ...")
        ft_args = TrainingArguments(
            output_dir=out,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # effective batch = 16
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.06,
            weight_decay=0.01,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,           # only keep best checkpoint, saves disk space
            fp16=True,                    # mixed precision: faster + less GPU memory
            dataloader_num_workers=2,
            report_to="none",
            seed=SEED,
        )
        trainer = Trainer(
            model=model, args=ft_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()

        print(f"\nBest checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Best val F1:     {trainer.state.best_metric:.4f}")

        print("\n  Saving training curves ...")
        save_training_curves(trainer.state.log_history, os.path.join(out, "training_curves.png"))

        model.save_pretrained(out)
        tokenizer.save_pretrained(out)

        # Post-finetune evaluation
        ft_val_preds  = np.argmax(trainer.predict(val_ds).predictions,  axis=-1)
        ft_test_preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
        ft_val_report  = print_report(val_df["labels"].tolist(),  ft_val_preds,  "FINE-TUNED -- val (in-domain)")
        ft_test_report = print_report(test_df["labels"].tolist(), ft_test_preds, "FINE-TUNED -- test (out-of-domain)")

        print("\n  Saving fine-tuned confusion matrices ...")
        save_confusion_matrix(val_df["labels"].tolist(),  ft_val_preds,
                              "Fine-tuned -- val",  os.path.join(out, "confusion_finetuned_val.png"))
        save_confusion_matrix(test_df["labels"].tolist(), ft_test_preds,
                              "Fine-tuned -- test", os.path.join(out, "confusion_finetuned_test.png"))

        results["finetuned"] = {
            "val":             extract_metrics(ft_val_report),
            "test":            extract_metrics(ft_test_report),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_val_f1":     trainer.state.best_metric,
        }

        print("\n[Delta: fine-tuned vs baseline]")
        for split in ["val", "test"]:
            bl = results["baseline"][split]["f1"]
            ft = results["finetuned"][split]["f1"]
            print(f"  {split:4s} F1: {bl:.4f} -> {ft:.4f}  ({ft-bl:+.4f})")

    # Save results.json — copy numbers from here into your report table
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    sep = "=" * 55
    print(f"\n{sep}\n  All outputs saved to: {out}\n{sep}")
    print("  results.json")
    print("  confusion_baseline_val.png / confusion_baseline_test.png")
    print("  confusion_finetuned_val.png / confusion_finetuned_test.png")
    print("  training_curves.png")
    print("Done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_path",          required=True,  help="train parquet (Reddit+Koo)")
    p.add_argument("--val_path",            required=True,  help="val parquet   (Reddit+Koo)")
    p.add_argument("--test_path",           required=True,  help="test parquet  (Bluesky+Voat)")
    p.add_argument("--writing_params_path", required=True,  help="output directory")
    p.add_argument("--reading_params_path", default=None,   help="load from local checkpoint instead of HF")
    p.add_argument("--eval_only",           action="store_true", help="skip training, only run baseline eval")
    main(p.parse_args())
