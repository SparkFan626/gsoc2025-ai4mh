# -*- coding: utf-8 -*-
import argparse, html, re
import pandas as pd
from pathlib import Path

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = html.unescape(s)
    s = URL_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="dataset/kaggle_clean.csv")
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=2000)
    args = ap.parse_args()

    usecols = ["text","class"]
    it = pd.read_csv(args.input, engine="python", on_bad_lines="skip",
                     usecols=usecols, chunksize=100000, iterator=True)

    chunks = []
    total_raw = 0
    for ch in it:
        total_raw += len(ch)
        ch = ch.dropna(subset=["text","class"])
        ch["text_norm"] = ch["text"].map(norm_text)
        ch = ch[(ch["text_norm"].str.len() >= args.min_len) & (ch["text_norm"].str.len() <= args.max_len)]

        ch["label"] = ch["class"].astype(str).str.strip().str.lower().map({
            "suicide": "suicidal",
            "non-suicide": "non_suicidal",
            "non_suicide": "non_suicidal",
        })
        ch = ch.dropna(subset=["label"])
        chunks.append(ch[["text","text_norm","label"]])

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["text","text_norm","label"])
    before = len(df)
    df = df.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    after = len(df)

    Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
    df[["text","label"]].to_csv(args.output, index=False, encoding="utf-8")
    print(f"[INFO] raw_rows={total_raw}  kept_after_len={before}  deduped={after}")
    print(f"[OK] saved -> {args.output}")

if __name__ == "__main__":
    main()
