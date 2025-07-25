import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Load model
MODEL_PATH = "results/final_model"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()  

def predict_text(text: str) -> str:
    """Predict whether the input text contains suicidal ideation"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Suicidal" if prediction == 1 else "Non-Suicidal"

if __name__ == "__main__":
    print("Enter text to analyze (type 'exit' to quit):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        result = predict_text(user_input)
        print(f"Prediction: {result}")
