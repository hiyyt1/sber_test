# src/llm.py
import os, json, yaml
from typing import List, Dict, Any
from openai import OpenAI, APIConnectionError, RateLimitError, BadRequestError

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

_API_KEY = _cfg["llm"].get("api_key") or os.getenv("OPENROUTER_API_KEY")
_BASE_URL = _cfg["llm"]["base_url"]
_MODEL = _cfg["llm"]["model"]
_TIMEOUT = int(_cfg["llm"].get("request_timeout", 12))

_client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL) if _API_KEY else None

SYSTEM_PROMPT = """Ты — строгий валидатор классификации. 
Тебе дан пользовательский запрос, список допустимых меток и свидетельства (похожие примеры).
Правила:
- Выбери РОВНО одну метку из предложенного списка labels, либо "OTHER", если уверенности нет.
- Отдай краткую причину в одно-два предложения.
Верни ТОЛЬКО JSON вида: {"label":"...", "reason":"..."} без лишнего текста.
"""

def validate_label_with_llm(
    query: str,
    candidate_labels: List[str],
    label_to_examples: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Возвращает {"label": str, "reason": str, "raw": str}
    Если LLM недоступен — вернёт {"label": "OTHER", "reason": "llm_unavailable", "raw": ""}
    """
    if _client is None:
        return {"label": "OTHER", "reason": "llm_unavailable", "raw": ""}

    user_payload = {
        "query": query,
        "labels": candidate_labels,
        "evidence": {lbl: label_to_examples.get(lbl, [])[:3] for lbl in candidate_labels},
        "instruction": "Choose best label from `labels` or OTHER. Return JSON only."
    }

    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=0.0,
            timeout=_TIMEOUT,
        )
        text = resp.choices[0].message.content.strip()
        text_clean = text.strip("` \n")
        if text_clean.startswith("{") and text_clean.endswith("}"):
            parsed = json.loads(text_clean)
        else:
            start = text.find("{")
            end = text.rfind("}")
            parsed = json.loads(text[start:end+1]) if start != -1 and end != -1 else {"label": "OTHER", "reason": "parse_failed"}

        label = parsed.get("label", "OTHER")
        reason = parsed.get("reason", "")
        if label not in candidate_labels and label != "OTHER":
            label = "OTHER"
        return {"label": label, "reason": reason, "raw": text}
    except (APIConnectionError, RateLimitError, BadRequestError, TimeoutError) as e:
        return {"label": "OTHER", "reason": f"llm_error:{type(e).__name__}", "raw": ""}
    except Exception as e:
        return {"label": "OTHER", "reason": f"llm_error:{str(e)[:80]}", "raw": ""}
