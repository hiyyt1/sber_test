# src/rerank.py
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Класс для повторного ранжирования (reranking) кандидатов с помощью кросс-энкодера.

    Используется после этапа гибридного поиска (dense + BM25) для уточнения порядка
    кандидатов. Кросс-энкодер оценивает степень смыслового соответствия пары 
    (запрос, документ) и сортирует результаты по убыванию вероятности релевантности.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        """
        Инициализация модели кросс-энкодера.

        Загружает предобученный reranker-модель из библиотеки SentenceTransformers
        и готовит её к предсказанию на выбранном устройстве.

        Args:
            model_name (str): Имя модели в Hugging Face (по умолчанию "BAAI/bge-reranker-base").
            device (str): Устройство для вычислений ("cpu" или "cuda").
        """
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[str], topk: int = 5) -> list[tuple[str, float]]:
        """
        Перерангировывает кандидатов по релевантности с помощью кросс-энкодера.
        Args:
            query (str): Текст запроса пользователя.
            candidates (list[str]): Список кандидатов (корпусных текстов) для оценки.
            topk (int): Количество лучших кандидатов, которые нужно вернуть.

        Returns:
            list[tuple[str, float]]:
                Отсортированный список пар (текст, score), где score — оценка релевантности.
                Чем выше значение, тем сильнее связь между запросом и документом.
        """
        if not candidates:
            return []

        pairs = [(query, c) for c in candidates]
        scores = self.model.predict(pairs) 
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:topk]  
