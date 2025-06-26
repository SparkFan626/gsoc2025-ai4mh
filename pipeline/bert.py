import pandas as pd
from transformers import pipeline
import torch
import os

# Create the output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Choose device: use GPU if available, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# Load the CSV file – make sure the file is in the same folder or update the path
df = pd.read_csv("output/filtered_reddit_posts.csv")

# Get the 'cleaned_text' column, drop any blank or missing values, and convert to string
texts = df["cleaned_text"].dropna()
texts = texts[texts.str.strip() != ""].astype(str).tolist()[:200]  # Limit to first 200 for speed

# Load the pre-trained sentiment analysis model from Hugging Face (fine-tuned on SST-2)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# Apply the sentiment model to each text
results = classifier(texts, truncation=True)

# Add the predictions back into a copy of the DataFrame
df_result = df.iloc[:len(results)].copy()
df_result["sentiment_label"] = [r["label"] for r in results]     # 'POSITIVE' or 'NEGATIVE'
df_result["sentiment_score"] = [r["score"] for r in results]     # Confidence score

# Save the result to a new CSV file
df_result.to_csv("output/bert_sentiment_result.csv", index=False)
print("Saved to output/bert_sentiment_result.csv ✅")
