# pipeline/extract_locations.py
import os, time, argparse, pandas as pd
from tqdm import tqdm

KEEP_LABELS = {"GPE","LOC","FAC","ORG"} 

def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("spaCy model not found. Run: python -m spacy download en_core_web_sm")

def build_geocoder(use_geocoder: bool, user_email: str):
    if not use_geocoder:
        return None
    from geopy.geocoders import Nominatim
    return Nominatim(user_agent=f"ai4mh_loc_extractor_{user_email}")

def geocode_safe(geocoder, name: str, sleep_sec: float = 1.1):
    if geocoder is None or not name:
        return None, None
    try:
        loc = geocoder.geocode(name, addressdetails=False, timeout=10)
        time.sleep(sleep_sec)  # respect rate limit
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        return None, None
    return None, None

def pick_text_column(df: pd.DataFrame):
    for c in ["text","Post","Tweet","body","content"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            return c
    raise ValueError(f"No text-like column found. Columns: {df.columns.tolist()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="output/geo_corpus.csv")
    ap.add_argument("--hits_csv",  default="output/locations_candidates.csv")
    ap.add_argument("--counts_csv",default="output/location_counts.csv")
    ap.add_argument("--use_geocoder", action="store_true")
    ap.add_argument("--email", default="you@example.com")  
    ap.add_argument("--sleep", type=float, default=1.1, help="geocoder rate limit seconds")
    args = ap.parse_args()

    os.makedirs("output", exist_ok=True)
    df = pd.read_csv(args.input_csv)
    text_col = pick_text_column(df)
    print(f"[INFO] input={args.input_csv}, text_col={text_col}, rows={len(df)}")

    nlp = load_spacy()
    geocoder = build_geocoder(args.use_geocoder, args.email)

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="NER"):
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        if not text.strip():
            continue
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in KEEP_LABELS:
                lat, lon = (None, None)
                if geocoder:
                    lat, lon = geocode_safe(geocoder, ent.text, sleep_sec=args.sleep)
                rows.append({
                    "row_id": idx,
                    "id": row.get("id", None),
                    "source": row.get("source", None),
                    "entity": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "text_preview": text[:300].replace("\n"," "),
                    "created_at": row.get("created_at", None),
                    "url": row.get("url", None),
                    "lat": lat,
                    "lon": lon,
                })

    hits_df = pd.DataFrame(rows)
    hits_df.to_csv(args.hits_csv, index=False)
    print(f"[OK] candidates -> {args.hits_csv}, rows={len(hits_df)}")

    if hits_df.empty:
        print("[WARN] no location entities found.")
        pd.DataFrame(columns=["entity","label","count"]).to_csv(args.counts_csv, index=False)
        return

    counts = (
        hits_df.groupby(["entity","label"], as_index=False)
               .size()
               .sort_values("size", ascending=False)
               .rename(columns={"size":"count"})
    )
    counts.to_csv(args.counts_csv, index=False)
    print(f"[OK] counts -> {args.counts_csv}, uniques={len(counts)}")

if __name__ == "__main__":
    main()
