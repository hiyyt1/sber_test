# src/route.py
import pickle
import faiss
import numpy as np
import yaml
import torch
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer

from src.retrieve import dense_search, bm25_search, reciprocal_rank_fusion
from src.rerank import Reranker
from src.llm import validate_label_with_llm

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg_device = str(cfg.get("device", "cpu")).lower()
device = "cuda" if (cfg_device == "cuda" and torch.cuda.is_available()) else "cpu"

encoder = SentenceTransformer(cfg["embedder"], device=device)
index = faiss.read_index("models/embeddings.faiss")
with open("models/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)
with open("models/prototypes.pkl", "rb") as f:
    protos = pickle.load(f)
with open("models/texts.pkl", "rb") as f:
    texts = pickle.load(f)
with open("models/labels.pkl", "rb") as f:
    labels = pickle.load(f)

PROTO_THR = float(cfg["confidence_threshold"])
RRF_K = int(cfg["rrf_k"])
TOPK_DENSE = int(cfg["dense_topk"])
TOPK_BM25 = int(cfg["bm25_topk"])

USE_RERANKER = bool(cfg.get("use_reranker", False))
RERANKER_MODEL = cfg.get("reranker_model", "BAAI/bge-reranker-base")
RERANK_TOPK = int(cfg.get("reranker_topk_after", 5))
reranker = Reranker(RERANKER_MODEL, device=device) if USE_RERANKER else None

USE_LLM = bool(cfg.get("use_llm_validation", True))
LLM_THR = float(cfg.get("llm_validation_threshold", 0.6))


def _norm_text(x) -> str:
    """
    Приводит входной текст к безопасному виду.

    Убирает None, NaN, пустые строки и приводит к str.
    """
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _majority_confidence(top_labels: list[str]) -> tuple[str, float]:
    """
    Определяет наиболее частую метку среди кандидатов и вычисляет её "уверенность".

    Args:
        top_labels (list[str]): Метки найденных документов.

    Returns:
        tuple[str, float]: (наиболее частая метка, доля её встречаемости).
    """
    if not top_labels:
        return "OTHER", 0.0
    cnt = Counter(top_labels)
    lbl, v = cnt.most_common(1)[0]
    return lbl, v / max(1, len(top_labels))


def predict(query: str) -> dict:
    """
    Основная функция маршрутизации (классификации) запроса.

    Последовательно выполняет несколько этапов:
      1. **Прототипы классов** — быстрая проверка, есть ли близкий класс.
      2. **Гибридный поиск (dense + BM25)** — поиск похожих примеров по эмбеддингам и словам.
      3. **Reranker (опционально)** — уточнение порядка кандидатов кросс-энкодером.
      4. **LLM-валидация (опционально)** — финальная проверка при низкой уверенности.

    Args:
        query (str): Пользовательский запрос.

    Returns:
        dict: Словарь с результатом классификации.
            {
                "label": str — предсказанный класс,
                "confidence": float — уверенность модели (0..1),
                "stage": str — этап, на котором принято решение,
                "examples": list[str] — несколько похожих запросов из корпуса,
                "debug": dict — отладочная информация (опционально)
            }
    """
    query = _norm_text(query)
    if not query:
        return {"label": "OTHER", "confidence": 0.0, "stage": "invalid", "examples": []}

    q_vec = encoder.encode([query], normalize_embeddings=True)[0].astype("float32")
    proto_scores = {lbl: float(np.dot(q_vec, vec)) for lbl, vec in protos.items()}
    best_label, best_proto_score = ("OTHER", 0.0)
    if proto_scores:
        best_label, best_proto_score = max(proto_scores.items(), key=lambda x: x[1])

    if best_proto_score >= PROTO_THR:
        return {
            "label": best_label,
            "confidence": best_proto_score,
            "stage": "prototype",
            "examples": []
        }

    dense_ids, _ = dense_search(q_vec, TOPK_DENSE)
    try:
        bm25_ids, _ = bm25_search(query, TOPK_BM25)
    except Exception:
        bm25_ids = []

    fused_ids = reciprocal_rank_fusion(dense_ids, bm25_ids, k=RRF_K)
    cand_texts = [texts[i] for i in fused_ids]
    cand_labels = [labels[i] for i in fused_ids]


    base_label, base_conf = _majority_confidence(cand_labels)

    stage = "hybrid"
    final_label, final_conf = base_label, base_conf
    final_examples = cand_texts[:3]

    if USE_RERANKER and cand_texts:
        ranked = reranker.rerank(query, cand_texts, topk=RERANK_TOPK)
        label_scores = defaultdict(float)
        label_counts = defaultdict(int)
        top_texts_reranked = []

        for text_i, score in ranked:
            idx = cand_texts.index(text_i)
            lbl = cand_labels[idx]
            label_scores[lbl] += float(score)
            label_counts[lbl] += 1
            if len(top_texts_reranked) < 3:
                top_texts_reranked.append(text_i)

        if label_scores:
            final_label = max(label_scores.items(), key=lambda x: x[1])[0]
            scores_arr = np.array(list(label_scores.values()), dtype="float32")
            exps = np.exp(scores_arr - scores_arr.max())
            conf_soft = float(exps.max() / exps.sum())
            final_conf = max(base_conf, conf_soft, best_proto_score)
            final_examples = top_texts_reranked
            stage = "hybrid+rerank"

    result = {
        "label": final_label,
        "confidence": final_conf,
        "stage": stage,
        "examples": final_examples,
        "debug": {
            "proto_best": best_proto_score,
            "hybrid_conf": base_conf
        }
    }

    if USE_LLM and final_conf < LLM_THR:
        cnt = Counter(cand_labels)
        candidate_labels = [l for l, _ in cnt.most_common(3)]
        label_to_examples = defaultdict(list)
        for i, lbl in zip(fused_ids, cand_labels):
            if len(label_to_examples[lbl]) < 3:
                label_to_examples[lbl].append(texts[i])

        llm_res = validate_label_with_llm(query, candidate_labels, label_to_examples)
        result["stage"] += "+llm"
        result["debug"]["llm_reason"] = llm_res.get("reason", "")

        if llm_res.get("label") and llm_res["label"] != "OTHER":
            result["label"] = llm_res["label"]
            result["confidence"] = max(result["confidence"], 0.66)

    return result
