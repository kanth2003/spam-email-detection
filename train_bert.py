"""Fine-tune DistilBERT for spam detection (use a GPU, e.g. free Google Colab T4).
Usage: python train_bert.py data.csv
"""
import sys, numpy as np, pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

NAME = "distilbert-base-uncased"
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/spam.csv").dropna(subset=["text", "label"])
df["label"] = df["label"].astype(str).str.lower().isin(["spam", "1"]).astype(int)
tr, te = train_test_split(df[["text", "label"]], test_size=0.2, stratify=df["label"], random_state=42)

tok = AutoTokenizer.from_pretrained(NAME)
enc = lambda b: tok(b["text"], truncation=True, max_length=256)
ds_tr = Dataset.from_pandas(tr.reset_index(drop=True)).map(enc, batched=True)
ds_te = Dataset.from_pandas(te.reset_index(drop=True)).map(enc, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    NAME, num_labels=2, id2label={0: "ham", 1: "spam"}, label2id={"ham": 0, "spam": 1})

def metrics(p):
    pred = np.argmax(p.predictions, axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(p.label_ids, pred, average="binary")
    return {"accuracy": accuracy_score(p.label_ids, pred), "precision": pr, "recall": rc, "f1": f1}

args = TrainingArguments(output_dir="bert_out", num_train_epochs=3, per_device_train_batch_size=16,
                         per_device_eval_batch_size=32, learning_rate=2e-5, weight_decay=0.01,
                         eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                         metric_for_best_model="f1", report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_te,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()
print(trainer.evaluate())
trainer.save_model("models/bert"); tok.save_pretrained("models/bert")
