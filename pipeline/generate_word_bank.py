import pandas as pd
from collections import Counter
import re
import os


input_path = os.path.join("dataset", "Suicidal Ideation Detection Reddit Dataset-Version 2.csv")
output_path = os.path.join("output", "word_bank.csv")

df = pd.read_csv(input_path)


label_col = df.columns[-1]
df[label_col] = df[label_col].astype(str).str.lower()


suicidal_df = df[df[label_col] == "suicidal"]


text_col = df.columns[0]
texts = suicidal_df[text_col].dropna().tolist()


words = []
for text in texts:
    tokens = re.findall(r'\b\w+\b', text.lower())  
    words.extend(tokens)


counter = Counter(words)
most_common = counter.most_common(200)  


pd.DataFrame(most_common, columns=["word", "frequency"]).to_csv(output_path, index=False)

print(f"Word bank saved to {output_path}")
