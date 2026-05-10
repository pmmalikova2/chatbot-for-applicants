# Чат-бот для абитуриентов НИУ ВШЭ

Чат-бот на основе Retrieval-Augmented Generation (RAG), отвечающий на вопросы абитуриентов
о поступлении в НИУ ВШЭ (Москва, бакалавриат). База знаний формируется из официальных
документов с ba.hse.ru.

## Структура

- data/raw_pdfs/ — исходные PDF
- data/processed/ — извлеченный текст
- data/qa.csv — золотой набор тестовых вопросов
- src/ — модули пайплайна
- scripts/ — точки входа
- configs/ — YAML-конфиги экспериментов
- results/ — результаты экспериментов
- legacy/ — старая версия кода

## Запуск (черновик)

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Запустить Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 3. Запустить Ollama
ollama serve

# 4. Подготовить данные
python scripts/01_ingest.py

# 5. Построить индекс под конкретный конфиг
python scripts/02_index.py --config configs/baseline.yaml

# 6. Запустить эксперимент
python scripts/03_run_experiment.py --config configs/baseline.yaml
```

## Стек

- Qdrant — векторная база
- Sentence-transformers / Ollama (bge-m3) — эмбеддеры
- Ollama (qwen2.5:7b-instruct) — LLM-генератор
- BAAI/bge-reranker-v2-m3 — реранкер
