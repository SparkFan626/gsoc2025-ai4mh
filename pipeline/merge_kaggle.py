import pandas as pd

df_old = pd.read_csv("output/unified_supervised.csv")

df_kaggle = pd.read_csv("dataset/Suicide_Detection.csv")

df_kaggle = df_kaggle.rename(columns={"class": "label", "text": "post"})

df_kaggle = df_kaggle.sample(n=80000, random_state=42)

df_all = pd.concat([df_old, df_kaggle], ignore_index=True)

df_all.to_csv("output/unified_supervised_v2.csv", index=False)

print(f"✅ merged dataset saved: {len(df_all)} rows")
