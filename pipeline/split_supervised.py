import argparse, html, re, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def norm_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = html.unescape(s)
    s = URL_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/unified_supervised_v2.csv")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--val_ratio",  type=float, default=0.10)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=4000)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    df = pd.read_csv(args.input, engine="python", on_bad_lines="skip")
    cols = set(df.columns)

    text_final = pd.Series([""] * len(df))
    if {"Title","Post"}.issubset(cols):
        text_final = (df["Title"].astype(str).fillna("").str.strip() + " " +
                      df["Post"].astype(str).fillna("").str.strip()).str.strip()
    if "text" in cols:
        m = text_final.eq("") | text_final.isna()
        text_final[m] = df.loc[m, "text"].astype(str).fillna("").str.strip()
    if "post" in cols:
        m = text_final.eq("") | text_final.isna()
        text_final[m] = df.loc[m, "post"].astype(str).fillna("").str.strip()

    if (text_final.eq("") | text_final.isna()).all():
        for c in df.columns:
            if df[c].dtype == "object":
                text_final = df[c].astype(str).fillna("").str.strip()
                break

    label_series = None
    for cand in ["label","class","Label"]:
        if cand in cols:
            label_series = df[cand]
            break
    if label_series is None:
        raise ValueError(f"Cannot find label column in: {list(df.columns)}")

    raw = label_series.astype(str).str.strip().str.lower()
    label_map = {
        "suicidal":"suicidal",
        "suicide":"suicidal",
        "non-suicidal":"non_suicidal",
        "non suicide":"non_suicidal",
        "non-suicide":"non_suicidal",
        "non_suicidal":"non_suicidal",
        "normal":"non_suicidal",
        "control":"non_suicidal",
    }
    mapped = raw.map(label_map)

    mask_lbl = mapped.isin(["suicidal","non_suicidal"])
    df2 = pd.DataFrame({"text": text_final, "label": mapped})
    df2 = df2.loc[mask_lbl].copy()

    df2["text_norm"] = df2["text"].map(norm_text)
    before = len(df2)
    df2 = df2[(df2["text_norm"].str.len() >= args.min_len) & (df2["text_norm"].str.len() <= args.max_len)]
    df2 = df2.drop_duplicates(subset=["text_norm"]).drop(columns=["text_norm"]).reset_index(drop=True)

    print(f"[INFO] cleaned & dedup: {before} -> {len(df2)}")
    print("[INFO] label dist (after map):", df2["label"].value_counts().to_dict())

    trainval, test = train_test_split(df2, test_size=args.test_ratio, stratify=df2["label"], random_state=SEED)
    val_ratio = args.val_ratio / (1 - args.test_ratio)
    train, val = train_test_split(trainval, test_size=val_ratio, stratify=trainval["label"], random_state=SEED)

    Path(args.out_dir).mkdir(exist_ok=True)
    train.to_csv(Path(args.out_dir) / "train.csv", index=False)
    val.to_csv(Path(args.out_dir) / "val.csv", index=False)
    test.to_csv(Path(args.out_dir) / "test.csv", index=False)

    print(f"[OK] saved -> {args.out_dir}/train.csv ({len(train)})  val.csv ({len(val)})  test.csv ({len(test)})")
    print("[INFO] dist train:", train["label"].value_counts().to_dict())
    print("[INFO] dist val  :", val["label"].value_counts().to_dict())
    print("[INFO] dist test :", test["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
