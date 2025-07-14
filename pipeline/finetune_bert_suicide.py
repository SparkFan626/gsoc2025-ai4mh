import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch

# 1. load data
df = pd.read_csv("dataset/Suicidal Ideation Detection Reddit Dataset-Version 2.csv")
df = df.dropna(subset=["Post"])
df = df[["Post", "Label"]]
df["Label"] = df["Label"].apply(lambda x: 1 if x.lower() == "suicidal" else 0)

# 2. split data into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Post"].tolist(), df["Label"].tolist(), test_size=0.1, random_state=42
)

# 3. load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 4. convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 5. load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6. set training parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # 改为每个epoch保存
    save_total_limit=2,     # 最多保存2个checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    max_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# 7. use Trainer API to start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 7. use Trainer API to start training
trainer.train()

# 8. save the final model
trainer.save_model("./results/final_model")

# 9. evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# 10. save evaluation results to file
import json
with open("./results/evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print("Model saved to ./results/final_model/")
print("Evaluation results saved to ./results/evaluation_results.json")
