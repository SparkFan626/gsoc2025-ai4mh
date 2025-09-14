# pipeline/train_bert_from_unified.py
import argparse, json, os, numpy as np, pandas as pd, torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from collections import Counter

def load_csv(path):
    df = pd.read_csv(path)
    assert {"text","label"}.issubset(df.columns), f"CSV must have columns: text,label; got {df.columns.tolist()}"
    return df[["text","label"]].copy()

def build_datasets(train_csv, val_csv, max_length, model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    def to_ds(df):
        label2id = {"non_suicidal":0, "suicidal":1}
        df = df.copy()
        df["label"] = df["label"].map(label2id)
        return Dataset.from_pandas(df, preserve_index=False)

    train_df = load_csv(train_csv)
    val_df   = load_csv(val_csv)

    train_ds = to_ds(train_df).map(tok, batched=True, remove_columns=["text"])
    val_ds   = to_ds(val_df).map(tok,   batched=True, remove_columns=["text"])

    return tokenizer, train_ds, val_ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:,1]
    try:    roc_auc = roc_auc_score(labels, probs)
    except: roc_auc = float("nan")
    try:    pr_auc  = average_precision_score(labels, probs)
    except: pr_auc  = float("nan")
    return {"accuracy":acc, "precision":p, "recall":r, "f1":f1, "roc_auc":roc_auc, "pr_auc":pr_auc}

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # 这里加上 num_items_in_batch=None，并保留 return_outputs
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    # ===== 新增：实验名，用于不覆盖旧模型 =====
    ap.add_argument("--exp_name", default="final_model", help="model will be saved to results/<exp_name>/")
    # ==========================================
    ap.add_argument("--train_csv", default="output/train.csv")
    ap.add_argument("--val_csv",   default="output/val.csv")
    ap.add_argument("--test_csv",  default="output/test.csv")
    ap.add_argument("--model",     default="bert-base-uncased")
    ap.add_argument("--epochs",    type=int, default=3)
    ap.add_argument("--bsz",       type=int, default=16)
    ap.add_argument("--lr",        type=float, default=2e-5)
    ap.add_argument("--max_length",type=int, default=256)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--use_class_weight", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--resume_from", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tokenizer, train_ds, val_ds = build_datasets(args.train_csv, args.val_csv, args.max_length, args.model)

    label2id = {"non_suicidal":0, "suicidal":1}
    id2label = {v:k for k,v in label2id.items()}

    model = BertForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=id2label, label2id=label2id
    )

    class_weights = None
    if args.use_class_weight:
        train_df = pd.read_csv(args.train_csv)
        cnt = Counter(train_df["label"].map(label2id))
        total = sum(cnt.values())
        class_weights = torch.tensor([total/cnt[i] for i in range(2)], dtype=torch.float)
        print("[INFO] class weights:", class_weights.tolist())

    data_collator = DataCollatorWithPadding(tokenizer)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    # ===== 新：按实验名保存，不覆盖旧模型 =====
    final_dir = out_dir / args.exp_name
    # ========================================

    training_args = TrainingArguments(
        output_dir=str(out_dir/"checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=str(Path("logs")),
        logging_steps=50,
        seed=args.seed,
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights
    )

    trainer.train(resume_from_checkpoint=args.resume_from)

    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(final_dir/"label_map.json","w") as f:
        json.dump({"label2id":label2id, "id2label":id2label}, f, indent=2)

    eval_val = trainer.evaluate()
    with open(out_dir/f"evaluation_val_{args.exp_name}.json","w") as f:
        json.dump(eval_val, f, indent=2)
    print("[VAL] ", eval_val)

    test_csv = Path(args.test_csv)
    if test_csv.exists():
        test_df = load_csv(test_csv)
        test_df["label"] = test_df["label"].map(label2id)
        test_ds = Dataset.from_pandas(test_df, preserve_index=False)
        def tok(b): return tokenizer(b["text"], truncation=True, max_length=args.max_length)
        test_ds = test_ds.map(tok, batched=True, remove_columns=["text"])
        test_metrics = trainer.evaluate(test_ds)
        with open(out_dir/f"evaluation_test_{args.exp_name}.json","w") as f:
            json.dump(test_metrics, f, indent=2)
        print("[TEST]", test_metrics)

    print(f"[OK] Saved best model to: {final_dir}")

if __name__ == "__main__":
    main()
