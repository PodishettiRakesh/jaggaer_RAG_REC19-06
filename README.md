# KG-RAG: Hybrid Retrieval Pipeline

Python RAG pipeline for ingesting PDF documents, chunking text, embedding chunks into ChromaDB, and retrieving relevant chunks with hybrid search (dense vectors + BM25).

See `problem.md` for the full system design and evaluation plan.

## Project structure

| File | Purpose |
| --- | --- |
| `main.py` | CLI entry point for `ingest` and `query` |
| `load_and_embed.py` | PDF loading, chunking, metadata enrichment, Chroma persistence |
| `retrieval.py` | Hybrid retrieval with dense + BM25 score fusion |
| `requirements.txt` | Python dependencies |
| `problem.md` | Architecture, components, and evaluation framework |

## Prerequisites

- Python 3.10+
- A local PDF to ingest (for example `2022 Q3 NVDA.pdf`)

## Setup

```bash
pip install -r requirements.txt
```

On first run, the sentence-transformers model (`all-MiniLM-L6-v2`) is downloaded automatically.

## Usage

### 1. Ingest a PDF

```bash
python main.py ingest "2022 Q3 NVDA.pdf"
```

This will:

1. Extract text per page with `pdfplumber`
2. Split text with a recursive character splitter (default: 1000 chars, 200 overlap)
3. Attach metadata (`document`, `company`, `quarter`, `page`, `chunk_id`)
4. Embed chunks and store them in `./chroma_db`

### 2. Query with hybrid search

```bash
python main.py query "What was NVIDIA's net cash from operating activities in Q3 2022?"
```

Hybrid retrieval:

1. Retrieves top-20 candidates from dense vector search
2. Retrieves top-20 candidates from BM25
3. Min-max normalizes both score sets
4. Fuses scores per chunk and returns top-5 results

**Final Score = 0.5 × Dense Score + 0.5 × BM25 Score**

Query output includes the fused score, dense/BM25 breakdown, and citation metadata (document, company, quarter, page, chunk id).

## CLI options

**Ingest**

```bash
python main.py ingest "2022 Q3 NVDA.pdf" \
  --collection-name kg_rag_single_doc \
  --persist-dir ./chroma_db \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --model-name all-MiniLM-L6-v2
```

**Query**

```bash
python main.py query "share repurchase operating cash flow" \
  --collection-name kg_rag_single_doc \
  --persist-dir ./chroma_db \
  --top-k 5 \
  --retrieve-k 20 \
  --dense-weight 0.5 \
  --sparse-weight 0.5 \
  --model-name all-MiniLM-L6-v2
```

| Flag | Default | Description |
| --- | --- | --- |
| `--collection-name` | `kg_rag_single_doc` | Chroma collection name |
| `--persist-dir` | `./chroma_db` | Local Chroma persistence directory |
| `--chunk-size` | `1000` | Chunk size in characters (ingest) |
| `--chunk-overlap` | `200` | Chunk overlap in characters (ingest) |
| `--top-k` | `5` | Final number of chunks returned (query) |
| `--retrieve-k` | `20` | Candidates per retrieval method before fusion (query) |
| `--dense-weight` | `0.5` | Weight for dense vector scores (query) |
| `--sparse-weight` | `0.5` | Weight for BM25 scores (query) |
| `--model-name` | `all-MiniLM-L6-v2` | Sentence Transformers embedding model |

## Notes

- Currently supports ingesting one PDF at a time per collection (re-ingesting replaces the collection).
- `./chroma_db` and local PDFs are gitignored; regenerate embeddings after clone with `ingest`.
- Uses ChromaDB `PersistentClient` (compatible with Chroma 1.x).

## Roadmap (from `problem.md`)

- Cross-encoder reranking (top-20 → top-5)
- LLM answer generation with citations
- Multi-document ingestion and evaluation metrics
