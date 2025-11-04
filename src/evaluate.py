# src/evaluate.py
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.route import predict

USE_FULL = False        
SAMPLE_N = 200          
SEED = 42


df = pd.read_csv("data.csv", dtype=str).rename(columns={"Text": "text", "Label": "label"})
df = df[["text", "label"]]
bad = {"", "nan", "none", "null"}
df["text"]  = df["text"].fillna("").astype(str).str.strip()
df["label"] = df["label"].fillna("").astype(str).str.strip()
df = df[~df["text"].str.lower().isin(bad)]
df = df[~df["label"].str.lower().isin(bad)]

if len(df) == 0:
    raise SystemExit("⚠️ Нет валидных строк (проверь колонки text/label).")

if USE_FULL or len(df) <= SAMPLE_N:
    test_df = df
else:
    _, test_df = train_test_split(
        df, test_size=SAMPLE_N, random_state=SEED, stratify=df["label"]
    )

preds, trues = [], []
for _, row in test_df.iterrows():
    res = predict(row["text"])
    preds.append(res["label"])
    trues.append(row["label"])

print(classification_report(trues, preds, digits=3, zero_division=0))
print(f"\nSamples: {len(test_df)} | Classes: {df['label'].nunique()} (in full dataset)")
