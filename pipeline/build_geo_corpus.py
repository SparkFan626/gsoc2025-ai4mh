# pipeline/build_geo_corpus.py
import argparse, html, json, re, hashlib
from pathlib import Path
import pandas as pd

OUT_DIR = Path("output")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = html.unescape(s)
    s = URL_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mk_id(text: str, source: str, idx: int) -> str:
    return hashlib.md5(f"{source}::{idx}::{text}".encode("utf-8","ignore")).hexdigest()

def pick_col(df: pd.DataFrame, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def load_reddit_v2(path: Path, merge_title_post=True, min_len=5, max_len=5000):
    if not path.exists():
        print(f"[WARN] Reddit V2 missing: {path}")
        return pd.DataFrame(columns=["id","source","text","created_at","url"])
    df = pd.read_csv(path, engine="python")
    title_col = pick_col(df, ["Title"])
    post_col  = pick_col(df, ["Post"])
    date_col  = pick_col(df, ["Date","Created","created_utc","timestamp"])
    url_col   = pick_col(df, ["URL","Permalink","url","link"])
    if merge_title_post and title_col and post_col:
        text = (df[title_col].fillna("") + " " + df[post_col].fillna("")).astype(str).str.strip()
    else:
        base = post_col or title_col
        if not base:
            return pd.DataFrame(columns=["id","source","text","created_at","url"])
        text = df[base].astype(str).str.strip()
    out = pd.DataFrame({
        "text": text,
        "source": "reddit_v2",
        "created_at": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT,
        "url": df[url_col] if url_col else None
    })
    out["text_norm"] = out["text"].map(norm_text)
    out = out[(out["text_norm"].str.len() >= min_len) & (out["text_norm"].str.len() <= max_len)]
    out = out.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    out["id"] = [mk_id(t, "reddit_v2", i) for i,t in enumerate(out["text_norm"])]
    return out[["id","source","text","created_at","url"]]

def load_twitter(path: Path, min_len=3, max_len=5000):
    if not path.exists():
        print(f"[WARN] Twitter missing: {path}")
        return pd.DataFrame(columns=["id","source","text","created_at","url"])
    df = pd.read_csv(path, engine="python")
    tw_col = pick_col(df, ["Tweet"])
    dt_col = pick_col(df, ["Created","created_at","timestamp"])
    if not tw_col:
        return pd.DataFrame(columns=["id","source","text","created_at","url"])
    text = df[tw_col].astype(str).map(html.unescape).str.strip()
    out = pd.DataFrame({
        "text": text,
        "source": "twitter_ds",
        "created_at": pd.to_datetime(df[dt_col], errors="coerce") if dt_col else pd.NaT,
        "url": None
    })
    out["text_norm"] = out["text"].map(norm_text)
    out = out[(out["text_norm"].str.len() >= min_len) & (out["text_norm"].str.len() <= max_len)]
    out = out.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    out["id"] = [mk_id(t, "twitter_ds", i) for i,t in enumerate(out["text_norm"])]
    return out[["id","source","text","created_at","url"]]

def load_sentiment_pool(path: Path, text_candidates=("cleaned_text","text","body","content"),
                        ts_candidates=("timestamp","created_at"), min_len=5, max_len=5000):
    if not path.exists():
        print(f"[WARN] sentiment pool missing: {path}")
        return pd.DataFrame(columns=["id","source","text","created_at","url"])
    df = pd.read_csv(path, engine="python")
    # 选文本列
    text_col = None
    for c in text_candidates:
        if c in df.columns:
            text_col = c; break
    if not text_col:
        # 兜底：第一列是文本
        text_col = df.columns[0]
    # 选时间
    ts_col = None
    for c in ts_candidates:
        if c in df.columns:
            ts_col = c; break
    # 选 URL（如有）
    url_col = "url" if "url" in df.columns else None

    text = df[text_col].astype(str).map(html.unescape).str.strip()
    created = None
    if ts_col:
        # 如果是 UNIX 秒
        try:
            created = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
        except Exception:
            created = pd.to_datetime(df[ts_col], errors="coerce")
    out = pd.DataFrame({
        "text": text,
        "source": "reddit_sentiment",
        "created_at": created if ts_col else pd.NaT,
        "url": df[url_col] if url_col else None
    })
    out["text_norm"] = out["text"].map(norm_text)
    out = out[(out["text_norm"].str.len() >= min_len) & (out["text_norm"].str.len() <= max_len)]
    out = out.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    out["id"] = [mk_id(t, "reddit_sentiment", i) for i,t in enumerate(out["text_norm"])]
    return out[["id","source","text","created_at","url"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reddit",  default="dataset/Suicidal Ideation Detection Reddit Dataset-Version 2.csv")
    ap.add_argument("--twitter", default="dataset/Suicide_Ideation_Dataset(Twitter-based).csv")
    ap.add_argument("--sent",    default="output/bert_sentiment_result.csv")
    ap.add_argument("--merge_title_post", action="store_true")
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=5000)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    r = load_reddit_v2(Path(args.reddit), merge_title_post=args.merge_title_post,
                       min_len=args.min_len, max_len=args.max_len)
    t = load_twitter(Path(args.twitter), min_len=3, max_len=args.max_len)
    s = load_sentiment_pool(Path(args.sent), min_len=args.min_len, max_len=args.max_len)

    geo = pd.concat([r,t,s], ignore_index=True)
    # 再全局去重一次（按归一化）
    tmp = geo.copy()
    tmp["text_norm"] = tmp["text"].map(norm_text)
    before = len(tmp)
    tmp = tmp.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)
    geo = tmp.drop(columns=["text_norm"])
    print(f"[INFO] merged & dedup: {before} -> {len(geo)}")

    out_path = OUT_DIR / "geo_corpus.csv"
    geo.to_csv(out_path, index=False)
    print(f"[OK] saved {out_path} rows={len(geo)}")

    with open(OUT_DIR / "geo_corpus.report.json","w",encoding="utf-8") as f:
        json.dump({
            "rows": len(geo),
            "sources_counts": geo["source"].value_counts().to_dict(),
            "path": str(out_path),
            "params": vars(args)
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] report -> {OUT_DIR/'geo_corpus.report.json'}")

if __name__ == "__main__":
    main()
