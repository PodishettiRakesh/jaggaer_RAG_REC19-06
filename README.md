# Single-Document Ingestion and Chroma Embedding

This repository contains a minimal Python script to load a single PDF document, chunk its text, compute sentence embeddings, and store them in a local Chroma database.

## Files

- `load_and_embed.py`: loads the PDF, creates overlapping chunks, and persists a Chroma collection.
- `requirements.txt`: Python dependencies required to run the script.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the script on a single PDF:

```bash
python main.py ingest "2022 Q3 NVDA.pdf"
```

4. The embeddings are stored in `./chroma_db` by default.

## Hybrid search

After ingesting the document, run hybrid retrieval over the same collection:

```bash
python main.py query "What was NVIDIA's net cash from operating activities in Q3 2022?"
```

## Custom options

```bash
python main.py ingest "2022 Q3 NVDA.pdf" --collection-name nvda_q3 --chunk-size 300 --chunk-overlap 50
```

```bash
python main.py query "What was NVIDIA's net cash from operating activities in Q3 2022?" --top-k 5 --dense-weight 0.6 --sparse-weight 0.4
```

## Notes

- The script currently supports a single PDF document.
- Text is chunked by word count with overlap for better retrieval context.
- Embeddings are created using `sentence-transformers` and stored locally in Chroma.
