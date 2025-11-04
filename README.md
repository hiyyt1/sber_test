
# гибридный поиск с Reranker и LLM

Классификатор текстовых запросов, объединяющий:

* ⚡ **Прототипы** (векторные центры классов)
* 🔍 **Гибридный поиск** (Dense + BM25 + RRF)
* 🎯 **Reranker (CrossEncoder)**
* 🤖 **LLM-валидация** (OpenRouter / DeepSeek)
* 💬 **Telegram-бот** для интерфейса

---

## 🚀 Быстрый старт

```bash
# 1. Установка окружения
python -m venv venv
source venv/bin/activate

# 2. Зависимости
pip install --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Настройка
cp config.example.yaml config.yaml
# добавь свой TELEGRAM_BOT_TOKEN и OPENROUTER_API_KEY

# 4. Генерация артефактов
python src/encode.py

# 5. Проверка и запуск
python -m src.evaluate
python app.py
```

---

## ⚙️ Конфигурация

`config.yaml`:

```yaml
embedder: "intfloat/multilingual-e5-base"
device: "cpu"
use_reranker: true
use_llm_validation: true
```

---

## 📁 Структура

```
├─ app.py
├─ config.example.yaml
├─ requirements.txt
├─ src/
│  ├─ encode.py
│  ├─ retrieve.py
│  ├─ rerank.py
│  ├─ route.py
│  ├─ llm.py
│  └─ evaluate.py
```

---

## 🧠 Архитектура

<p align="center">
  <img src="https://github.com/hiyyt1/sber_test/blob/main/dbc39f6c-8a49-44f6-9dc4-a8a3c72588e7.png" width="700">
</p>

---

## 🔑 Переменные окружения

```bash
export OPENROUTER_API_KEY="sk-or-..."
export TELEGRAM_BOT_TOKEN="..."
```


