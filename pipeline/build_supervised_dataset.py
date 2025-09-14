# pipeline/build_supervised_dataset.py
import argparse, hashlib, html, re, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUT_DIR = Path("output")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def norm_text(s: str) -> str:
    """用于去重的归一化：html反转义 -> 去URL -> 小写 -> 折叠空白"""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = html.unescape(s)
    s = URL_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mk_id(text: str, source: str, idx: int) -> str:
    h = hashlib.md5(f"{source}::{idx}::{text}".encode("utf-8", "ignore")).hexdigest()
    return h

def pick_col(df: pd.DataFrame, candidates):
    """大小写无关选列名：返回实际存在的第一项"""
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def load_reddit_v2(path: Path, merge_title_post=True, min_len=5, max_len=2000):
    if not path.exists():
        print(f"[WARN] Reddit V2 not found: {path}")
        return pd.DataFrame(columns=["id","source","text","label","created_at"])

    df = pd.read_csv(path, engine="python")
    title_col = pick_col(df, ["Title"])
    post_col  = pick_col(df, ["Post"])
    label_col = pick_col(df, ["Label"])

    # 文本
    if merge_title_post and title_col and post_col:
        text = (
            df[title_col].fillna("").astype(str).str.strip() + " " +
            df[post_col].fillna("").astype(str).str.strip()
        ).str.strip()
    else:
        # 退化：只有 Post
        base_col = post_col or title_col
        if not base_col:
            print("[WARN] Reddit V2 missing Title/Post, skip.")
            return pd.DataFrame(columns=["id","source","text","label","created_at"])
        text = df[base_col].fillna("").astype(str).str.strip()

    # 标签
    label_map = {
        "suicidal": "suicidal",
        "non-suicidal": "non_suicidal",
        "non suicidal": "non_suicidal",
    }
    labels_raw = df[label_col].astype(str).str.strip().str.lower()
    labels = labels_raw.map(label_map)

    out = pd.DataFrame({
        "text": text,
        "label": labels,
        "source": "reddit_v2",
        "created_at": pd.NaT,
    })
    out = out.dropna(subset=["text","label"])
    out["text_norm"] = out["text"].map(norm_text)
    out = out[(out["text_norm"].str.len() >= min_len) & (out["text_norm"].str.len() <= max_len)]
    out = out.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    out["id"] = [mk_id(t, "reddit_v2", i) for i, t in enumerate(out["text_norm"])]
    return out[["id","source","text","label","created_at"]]

def load_twitter(path: Path, min_len=3, max_len=2800):
    if not path.exists():
        print(f"[WARN] Twitter dataset not found: {path}")
        return pd.DataFrame(columns=["id","source","text","label","created_at"])

    df = pd.read_csv(path, engine="python")
    tweet_col  = pick_col(df, ["Tweet"])
    suicide_col = pick_col(df, ["Suicide"])
    if not tweet_col or not suicide_col:
        print("[WARN] Twitter missing Tweet/Suicide columns, skip.")
        return pd.DataFrame(columns=["id","source","text","label","created_at"])

    text = df[tweet_col].astype(str).map(html.unescape).str.strip()
    label_map = {
        "potential suicide post": "suicidal",
        "not suicide post": "non_suicidal",
    }
    labels_raw = df[suicide_col].astype(str).str.strip().str.lower()
    labels = labels_raw.map(label_map)

    out = pd.DataFrame({
        "text": text,
        "label": labels,
        "source": "twitter_ds",
        "created_at": pd.NaT,
    }).dropna(subset=["text","label"])

    out["text_norm"] = out["text"].map(norm_text)
    out = out[(out["text_norm"].str.len() >= min_len) & (out["text_norm"].str.len() <= max_len)]
    out = out.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    out["id"] = [mk_id(t, "twitter_ds", i) for i, t in enumerate(out["text_norm"])]
    return out[["id","source","text","label","created_at"]]

def stratified_split(df: pd.DataFrame, test_size=0.10, val_size=0.10, seed=SEED):
    # 先切 test，再从剩余里切 val（保持分层）
    trainval, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(trainval, test_size=val_ratio, stratify=trainval["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reddit",  default="dataset/Suicidal Ideation Detection Reddit Dataset-Version 2.csv")
    parser.add_argument("--twitter", default="dataset/Suicide_Ideation_Dataset(Twitter-based).csv")
    parser.add_argument("--merge_title_post", action="store_true", help="Reddit: 合并 Title+Post")
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--val_ratio",  type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    reddit = load_reddit_v2(Path(args.reddit), merge_title_post=args.merge_title_post,
                            min_len=args.min_len, max_len=args.max_len)
    tw     = load_twitter(Path(args.twitter), min_len=3, max_len=2800)

    sup = pd.concat([reddit, tw], ignore_index=True)
    # 统一再清洗一遍文本（保持原 text 不动，仅用于筛选/去重）
    tmp = sup.copy()
    tmp["text_norm"] = tmp["text"].map(norm_text)
    before = len(tmp)
    tmp = tmp[(tmp["text_norm"].str.len() >= args.min_len) & (tmp["text_norm"].str.len() <= args.max_len)]
    tmp = tmp.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    print(f"[INFO] global dedup/length: {before} -> {len(tmp)}")

    # 删掉辅助列
    sup = tmp.drop(columns=["text_norm"])

    # 保存全集
    all_path = OUT_DIR / "unified_supervised.csv"
    sup.to_csv(all_path, index=False)
    print(f"[OK] saved {all_path} rows={len(sup)}")

    # 类分布
    dist = sup["label"].value_counts().to_dict()
    print("[INFO] class distribution:", dist)

    # 分层切分
    train, val, test = stratified_split(sup, test_size=args.test_ratio, val_size=args.val_ratio, seed=args.seed)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"[OK] saved splits: train={len(train)}  val={len(val)}  test={len(test)}")

    # 保存报告
    report = {
        "rows_total": len(sup),
        "class_distribution": dist,
        "paths": {
            "unified": str(all_path),
            "train": str(OUT_DIR / "train.csv"),
            "val": str(OUT_DIR / "val.csv"),
            "test": str(OUT_DIR / "test.csv"),
        },
        "params": vars(args),
    }
    with open(OUT_DIR / "unified_supervised.report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] report -> {OUT_DIR / 'unified_supervised.report.json'}")

if __name__ == "__main__":
    main()
