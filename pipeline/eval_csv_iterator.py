# -*- coding: utf-8 -*-
"""
Evaluate a CSV iteratively against a HF classifier and write a metrics report.

Usage (PowerShell):
  python pipeline/eval_csv_iterator.py `
    --model_dir results/final_model_v3 `
    --csv dataset/kaggle_clean.csv `
    --threshold 0.5 `
    --chunksize 20000 `
    --bsz 128 `
    --max_len 128 `
    --out_prefix results/eval_kaggle_clean_v3

CSV requirements:
- Must contain a text column named "text".
- Must contain a label column named "label" or "class".
- Label values accepted (case-insensitive):
    positive -> 1:  "suicidal", "suicide", 1, "1", "yes", "true"
    negative -> 0:  "non_suicidal", "non-suicide", 0, "0", "no", "false"
"""

import argparse
import json
import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out_prefix", default="results/eval_iter")
    return ap.parse_args()


# ---------- helpers ----------
POS_SET = {"suicidal", "suicide", "1", 1, "yes", "true", True}
NEG_SET = {"non_suicidal", "non-suicide", "0", 0, "no", "false", False}


def normalize_label(x):
    """Map diverse label values to {0,1}; return np.nan if unknown."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer)):
        if x == 1:
            return 1
        if x == 0:
            return 0
        # other integers → unknown
        return np.nan
    s = str(x).strip().lower()
    if s in POS_SET:
        return 1
    if s in NEG_SET:
        return 0
    # try numeric strings like "1.0"/"0.0"
    try:
        fv = float(s)
        if fv == 1.0:
            return 1
        if fv == 0.0:
            return 0
    except Exception:
        pass
    return np.nan


def find_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (text_col, label_col) with simple heuristics and clear error if missing."""
    cols_lower = {c.lower(): c for c in df.columns}
    text_col = cols_lower.get("text")
    label_col = cols_lower.get("label") or cols_lower.get("class")
    if not text_col:
        raise ValueError(
            f"CSV must contain a 'text' column. Found: {list(df.columns)}"
        )
    if not label_col:
        raise ValueError(
            f"CSV must contain a 'label' or 'class' column. Found: {list(df.columns)}"
        )
    return text_col, label_col


def chunk_iterator(csv_path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    # Read a small sample to detect columns, then re-open with usecols for efficiency.
    sample = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", nrows=1000)
    text_col, label_col = find_columns(sample)

    it = pd.read_csv(
        csv_path,
        engine="python",
        on_bad_lines="skip",
        usecols=[text_col, label_col],
        chunksize=chunksize,
        iterator=True,
    )
    for df in it:
        yield df.rename(columns={text_col: "text", label_col: "label"})


def build_encoder(tokenizer, max_len: int):
    def encode(batch_texts: Iterable[str]):
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    return encode


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Support both binary heads:
    # - num_labels == 2 -> softmax and take prob of class 1
    # - num_labels == 1 -> sigmoid(logit)
    num_labels = getattr(getattr(model, "config", None), "num_labels", 2)

    def predict_proba(batch_enc):
        with torch.no_grad():
            logits = model(**batch_enc).logits
            if num_labels == 1:
                # shape [N,1]
                p1 = torch.sigmoid(logits.view(-1))
            else:
                # shape [N,2]
                p1 = torch.softmax(logits, dim=-1)[:, 1]
            return p1.detach().cpu().numpy()

    encode = build_encoder(tokenizer, args.max_len)

    y_true_all, y_prob_all = [], []
    n_samples = 0
    n_kept_text, n_kept_label = 0, 0

    try:
        iterator = chunk_iterator(args.csv, args.chunksize)
    except Exception as e:
        print(f"[ERROR] CSV column detection failed: {e}")
        return

    for i, df in enumerate(iterator, 1):
        # Drop rows missing text or label; normalize labels
        df = df.dropna(subset=["text", "label"]).copy()
        n_kept_text += len(df)

        df["label"] = df["label"].map(normalize_label)
        df = df.dropna(subset=["label"])
        n_kept_label += len(df)
        if df.empty:
            print(f"[DEBUG] chunk={i} skipped (no valid rows)")
            continue

        texts = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).values

        # Batched inference
        probs = []
        for j in range(0, len(texts), args.bsz):
            batch_texts = texts[j : j + args.bsz]
            enc = encode(batch_texts)
            enc = {k: v.to(device) for k, v in enc.items()}
            probs.append(predict_proba(enc))
        probs = np.concatenate(probs)

        y_true_all.append(labels)
        y_prob_all.append(probs)
        n_samples += len(labels)
        print(f"[DEBUG] chunk={i} size={len(labels)} total={n_samples}")

    if n_samples == 0:
        print("[WARN] no samples parsed from CSV after cleaning.")
        return

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = (y_prob >= args.threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    report = {
        "n_samples": int(n_samples),
        "threshold": float(args.threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "confusion_matrix_labels": ["non-suicide", "suicide"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "model_dir": args.model_dir,
        "csv": args.csv,
        "kept_after_dropna_text": int(n_kept_text),
        "kept_after_label_normalize": int(n_kept_label),
        "model_num_labels": int(num_labels),
        "device": str(device),
    }

    with open(f"{args.out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[REPORT]", json.dumps(report, indent=2, ensure_ascii=False))

    # Save head scores (cap to 200k rows to keep file size reasonable)
    df_scores = pd.DataFrame(
        {
            "label": np.where(y_true == 1, "suicide", "non-suicide"),
            "prob_suicide": y_prob,
        }
    )
    df_scores.head(200_000).to_csv(
        f"{args.out_prefix}_scores_head200k.csv", index=False
    )


if __name__ == "__main__":
    main()
