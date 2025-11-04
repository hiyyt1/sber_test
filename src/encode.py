# src/encode.py
import os
import pickle
import faiss
import numpy as np
import pandas as pd
import yaml
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs("models", exist_ok=True)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает датафрейм от пустых и некорректных значений.

    Приводит имена колонок к стандартным ('text', 'label'),
    удаляет строки с пустыми или NaN значениями.

    Args:
        df (pd.DataFrame): Исходный датафрейм с колонками текстов и меток.

    Returns:
        pd.DataFrame: Очищенный датафрейм с двумя колонками — 'text' и 'label'.
    """
    df = df.rename(columns={"Text": "text", "Label": "label"})
    df = df[["text", "label"]]
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    bad_tokens = {"", "nan", "none", "null"}
    df = df[~df["text"].str.lower().isin(bad_tokens)]
    df = df[~df["label"].str.lower().isin(bad_tokens)]
    return df


df_raw = pd.read_csv("data.csv", dtype=str)
df = _clean_df(df_raw)
texts = df["text"].tolist()
labels = df["label"].tolist()
unique_labels = sorted(set(labels))

cfg_device = str(cfg.get("device", "cpu")).lower()
device = "cuda" if (cfg_device == "cuda" and torch.cuda.is_available()) else "cpu"

encoder = SentenceTransformer(cfg["embedder"], device=device)
print(f"[encode] using device={device}, model={cfg['embedder']}")
embeddings = encoder.encode(
    texts,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True
)
embeddings = np.asarray(embeddings, dtype="float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, "models/embeddings.faiss")

tokenized_corpus = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized_corpus)
with open("models/bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

prototypes: dict[str, np.ndarray] = {}
labels_np = np.array(labels)
for lbl in unique_labels:
    mask = (labels_np == lbl)
    cls_vecs = embeddings[mask]
    if len(cls_vecs) == 0:
        continue
    proto = cls_vecs.mean(axis=0)
    norm = np.linalg.norm(proto)
    if norm > 0:
        proto = proto / norm
    prototypes[lbl] = proto.astype("float32")

with open("models/prototypes.pkl", "wb") as f:
    pickle.dump(prototypes, f)

with open("models/texts.pkl", "wb") as f:
    pickle.dump(texts, f)
with open("models/labels.pkl", "wb") as f:
    pickle.dump(labels, f)

