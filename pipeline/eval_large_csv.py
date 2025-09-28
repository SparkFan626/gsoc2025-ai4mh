# pipeline/eval_large_csv.py
import argparse, html, json, numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             average_precision_score)
from tqdm import tqdm

def clean_text_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).map(html.unescape)
    s = s.str.replace(r"[\u2018\u2019\u201B\u2032]", "'", regex=True)
    s = s.str.replace(r"[\u201C\u201D\u2033]", '"', regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def load_model(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, mdl

@torch.no_grad()
def infer_batch(texts, tok, mdl, max_len=256, bsz=64):
    device = next(mdl.parameters()).device
    probs = []
    for i in range(0, len(texts), bsz):
        enc = tok(texts[i:i+bsz], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = mdl(**enc).logits
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)

def make_reader(path, chunksize, text_col, label_col, encoding_try=("utf-8","utf-8-sig","latin-1")):
    last_err = None
    for enc in encoding_try:
        try:
            return pd.read_csv(
                path,
                chunksize=chunksize,
                dtype={text_col: str, label_col: str},
                engine="python",
                encoding=enc,
                on_bad_lines="skip",
                quoting=3  # QUOTE_NONE
            )
        except Exception as e:
            last_err = e
    raise last_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="results/final_model_v2")
    ap.add_argument("--csv", default="dataset/Suicide_Detection.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="class")
    ap.add_argument("--pos_label", default="suicide")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--limit_rows", type=int, default=0, help="仅评估前N行，0为全量")
    ap.add_argument("--out_prefix", default="results/eval_kaggle_final_model_v2")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(exist_ok=True)

    tok, mdl = load_model(args.model_dir)

    all_probs, all_labels = [], []
    fp_rows, fn_rows = [], []

    total = 0
    reader = make_reader(args.csv, args.chunksize, args.text_col, args.label_col)

    pbar = tqdm(desc="Eval chunks", unit="chunk")
    for chunk in reader:
        pbar.update(1)
        if args.limit_rows and total >= args.limit_rows:
            break
        if args.limit_rows:
            remain = args.limit_rows - total
            if len(chunk) > remain:
                chunk = chunk.iloc[:remain]

        print(f"[DEBUG] chunk_size={len(chunk)} total_done={total}", flush=True)

        texts = clean_text_series(chunk[args.text_col].fillna(""))
        y = (chunk[args.label_col].astype(str).str.strip().str.lower() == args.pos_label).astype(int).values

        probs = infer_batch(texts.tolist(), tok, mdl, args.max_len, args.bsz)
        preds = (probs >= args.threshold).astype(int)

        all_probs.append(probs)
        all_labels.append(y)

        mis_fp = (preds == 1) & (y == 0)
        mis_fn = (preds == 0) & (y == 1)
        if mis_fp.any():
            tmp = pd.DataFrame({"text": texts[mis_fp], "label": y[mis_fp], "prob": probs[mis_fp]})
            fp_rows.append(tmp.sample(min(len(tmp), 100), random_state=42))
        if mis_fn.any():
            tmp = pd.DataFrame({"text": texts[mis_fn], "label": y[mis_fn], "prob": probs[mis_fn]})
            fn_rows.append(tmp.sample(min(len(tmp), 100), random_state=42))

        total += len(chunk)

    pbar.close()

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = (y_prob >= args.threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc = None
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = None

    report = {
        "n_samples": int(len(y_true)),
        "threshold": float(args.threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "confusion_matrix_labels": ["non-suicide", "suicide"],
        "confusion_matrix": cm,
        "model_dir": args.model_dir,
        "csv": args.csv,
    }
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[REPORT]", json.dumps(report, indent=2))

    if fp_rows:
        pd.concat(fp_rows, ignore_index=True).to_csv(f"{out_prefix}_FP_samples.csv", index=False)
    if fn_rows:
        pd.concat(fn_rows, ignore_index=True).to_csv(f"{out_prefix}_FN_samples.csv", index=False)

    pd.DataFrame({"prob": y_prob, "label": y_true}).to_csv(f"{out_prefix}_scores.csv", index=False)
    print(f"[OK] saved -> {out_prefix}.json / _scores.csv / _FP_samples.csv / _FN_samples.csv")
    
if __name__ == "__main__":
    main()
