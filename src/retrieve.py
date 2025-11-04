import faiss
import pickle
import numpy as np
from functools import lru_cache
from typing import Tuple, List


@lru_cache(maxsize=1)
def _load_faiss_index() -> faiss.Index:
    """
    Загрузка FAISS-индекса из файла.

    Кэширует объект в памяти при первом обращении, чтобы
    не читать файл при каждом вызове dense_search.

    Returns:
        faiss.Index: Загруженный FAISS-индекс.
    """
    return faiss.read_index("models/embeddings.faiss")


@lru_cache(maxsize=1)
def _load_bm25_and_texts() -> Tuple[object, list]:
    """
    Загрузка BM25-индекса и списка текстов из файлов.

    Используется для быстрого поиска по ключевым словам.

    Returns:
        Tuple:
            bm25 (BM25Okapi): Предобученный BM25-индекс.
            texts (list[str]): Список текстов корпуса.
    """
    bm25 = pickle.load(open("models/bm25_index.pkl", "rb"))
    texts = pickle.load(open("models/texts.pkl", "rb"))
    return bm25, texts


def dense_search(query_vec: np.ndarray, topk: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Поиск ближайших документов по эмбеддингам (dense-поиск).

    Args:
        query_vec (np.ndarray): Нормализованный L2-вектор запроса.
        topk (int): Количество ближайших кандидатов для возврата.

    Returns:
        Tuple:
            indices (np.ndarray[int]): Индексы найденных документов.
            scores (np.ndarray[float]): Косинусные сходства с запросом.
    """
    index = _load_faiss_index()
    D, I = index.search(query_vec.reshape(1, -1), topk)
    return I[0], D[0]


def bm25_search(query: str, topk: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Поиск релевантных документов по ключевым словам (BM25-поиск).

    Args:
        query (str): Пользовательский запрос в текстовом виде.
        topk (int): Количество кандидатов для возврата.

    Returns:
        Tuple:
            indices (np.ndarray[int]): Индексы документов.
            scores (np.ndarray[float]): BM25-оценки релевантности.

    Note:
        При пустом запросе возвращаются нулевые оценки.
    """
    bm25, texts = _load_bm25_and_texts()
    q = (query or "").lower().strip()
    tokens = q.split() if q else []
    scores = bm25.get_scores(tokens) if tokens else np.zeros(len(texts), dtype=float)
    top_ids = np.argsort(scores)[::-1][:topk]
    return top_ids, scores[top_ids]


def reciprocal_rank_fusion(rank_a: List[int], rank_b: List[int], k: int = 60) -> List[int]:
    """
    Слияние двух списков кандидатов методом Reciprocal Rank Fusion (RRF).

    Алгоритм комбинирует результаты из dense- и sparse-поиска, 
    назначая каждому документу вес 1 / (k + rank), где rank — позиция документа в каждом списке.

    Args:
        rank_a (List[int]): Список индексов документов из dense-поиска.
        rank_b (List[int]): Список индексов документов из BM25-поиска.
        k (int): Параметр сглаживания (обычно 30–120).

    Returns:
        List[int]: Итоговый список индексов документов, отсортированных по убыванию интегрального веса (топ-10).
    """
    all_ids = list(dict.fromkeys(list(rank_a) + list(rank_b)))
    pos_a = {doc: i + 1 for i, doc in enumerate(rank_a)}
    pos_b = {doc: i + 1 for i, doc in enumerate(rank_b)}
    len_a, len_b = len(rank_a) + 1, len(rank_b) + 1

    scores = {}
    for doc in all_ids:
        ra = pos_a.get(doc, len_a)
        rb = pos_b.get(doc, len_b)
        scores[doc] = 1.0 / (k + ra) + 1.0 / (k + rb)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:10]]
