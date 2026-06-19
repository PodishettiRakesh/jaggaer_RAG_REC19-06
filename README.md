# KG-RAG — Question Answering over a Multi-Document Corpus

A hybrid retrieval-augmented generation (RAG) system built for the Docugami KG-RAG datasets (SEC 10-Q filings and NTSB Aviation Incident/Accident Reports). The system ingests long-form PDF documents, retrieves the most relevant chunks using a dense + BM25 hybrid search, builds a grounded context, and generates a cited answer using Gemini.

---

## How the system works

The pipeline has four stages, each in its own module:

```
PDF → [load_and_embed] → ChromaDB
Query → [retrieval] → top-k chunks → [context_builder] → context + citations → [generation] → answer
```

### 1. Ingestion (`load_and_embed.py`)

- Reads pages from a PDF using `pdfplumber`, preserving page numbers.
- Parses company and quarter from the filename convention (e.g. `2022 Q3 NVDA.pdf` → company: `NVDA`, quarter: `2022_Q3`). This metadata bootstraps source attribution for every chunk.
- Splits page text with a **recursive character splitter** (LangChain-style separator hierarchy: `\n\n → \n → ". " → " " → ""`), defaulting to 1,000-character chunks with 200-character overlap.
- Embeds all chunks with `all-MiniLM-L6-v2` (Sentence Transformers) and persists them in a local ChromaDB `PersistentClient`.
- Each chunk carries structured metadata: `document`, `company`, `quarter`, `page`, `chunk_id`.

### 2. Hybrid retrieval (`retrieval.py`)

- Retrieves the top-`retrieve_k` (default 20) candidates from **dense vector search** via ChromaDB cosine similarity.
- Retrieves the top-`retrieve_k` candidates from **BM25 sparse search** over the full collection, computed in-memory using `rank_bm25`.
- Min-max normalises both score sets independently, then fuses them:

  ```
  final_score = dense_weight × dense_score + sparse_weight × bm25_score
  ```

  Default weights are 0.5 / 0.5. Both are tunable via CLI flags.
- Returns the top-`top_k` (default 5) chunks by fused score, with per-chunk `dense_score` and `bm25_score` exposed for inspection.

Hybrid search is more robust than either method alone: dense search captures semantic meaning while BM25 catches exact financial terms, ticker symbols, and numeric figures that embeddings compress.

### 3. Context building (`context_builder.py`)

- Sorts retrieved chunks by `(company, quarter, page, score)` to present evidence in document order.
- Formats each chunk as a labelled source block:
  ```
  [Source 1: 2022_Q3_NVDA.pdf | Company: NVDA | Quarter: 2022_Q3 | Page: 8 | Chunk: chunk_3]
  <chunk text>
  ```
- Collects unique `document + page` citations for the sources section.

### 4. Answer generation (`generation.py`)

- Sends the context and question to Gemini via the `google-genai` SDK.
- The system prompt instructs the model to answer **only from the provided context** and to explicitly state when the answer cannot be determined from the corpus — satisfying the unanswerable-question requirement.
- Returns the answer text and a formatted sources list.

---

## Handling the three question difficulty tiers

| Tier | What it requires | How this system handles it |
|---|---|---|
| **Single-Doc, Single-Chunk** | Retrieve one correct chunk from one document | Dense + BM25 fusion reliably surfaces the single best chunk; Gemini cites its page. |
| **Single-Doc, Multi-Chunk** | Retrieve multiple non-contiguous regions from one document | `retrieve_k=20` candidates before fusion gives broad coverage; context builder preserves all top-k chunks ordered by page. |
| **Multi-Doc** | Retrieve correct chunks across several documents | Multiple PDFs can be ingested into the same named collection; hybrid search ranks across all of them jointly. |

**Known limitation on Multi-Doc:** the current CLI ingests one PDF at a time and overwrites the collection. To support multi-document queries, run `ingest` sequentially with the same `--collection-name` — but note that `create_chroma_collection` currently deletes and recreates the collection on each call. A `--append` mode is the first planned extension (see Roadmap).

**Unanswerable questions:** The generation prompt explicitly instructs Gemini: *"If the context does not contain enough information to answer the question, explicitly state that the answer cannot be determined from the corpus."* This is enforced in the prompt, not via a retrieval threshold, which is the current tradeoff (see Evaluation Notes).

**Table-heavy documents:** SEC 10-Q financial statements contain dense tables. The current approach relies on `pdfplumber`'s text extraction, which preserves table rows as text lines. Retrieval of numeric figures works reasonably well for single-value lookups (e.g. net cash from operations) but degrades for cross-row aggregations. A dedicated table extraction path is the second planned extension.

---

## Evaluation notes

### Metrics proposed

Three metrics are appropriate for this system, in order of implementation priority:

**1. Answer correctness (implemented manually)**
Compare generated answers against the KG-RAG human-reviewed Q&A (`qna_data`) for the SEC 10-Q collection. Scoring: exact-match for numeric answers, substring-match for named entities.

**2. Citation precision (implemented)**
Every answer includes a sources section listing `document + page`. This can be checked against the human Q&A ground-truth source annotations: does the cited page actually contain the answer span?

**3. Retrieval hit-rate (implemented via `query` subcommand)**
Use `python main.py query "<question>"` to inspect raw hybrid-search results. Check whether the ground-truth chunk appears in the top-5 results. This directly measures whether the retrieval stage, not the generator, is the failure point.

### Metric not yet implemented: Ragas

[Ragas](https://github.com/explodinggradients/ragas) provides automated faithfulness and answer-relevancy scoring without ground-truth answers. It is the planned evaluation extension — it would run `generate_answer` output through a second LLM pass that checks whether every claim in the answer is supported by the retrieved context.

### Key design decisions and tradeoffs

- **Hybrid over pure dense:** Financial documents contain exact numeric values and ticker symbols that semantic embeddings compress poorly. BM25 rescues these. The equal 0.5/0.5 split is a reasonable default; tuning toward BM25 may improve numeric-heavy queries.
- **`all-MiniLM-L6-v2` for embeddings:** Fast and lightweight. A domain-adapted financial embedding model (e.g. `FinBERT`) would likely improve retrieval quality on SEC filings at the cost of ingestion speed.
- **No cross-encoder reranking:** A cross-encoder reranker (e.g. `ms-marco-MiniLM`) over the top-20 candidates before passing top-5 to the LLM would improve precision, especially for Multi-Chunk and Multi-Doc tiers. This is the third planned extension.
- **Unanswerable detection via prompt, not threshold:** The current approach trusts the LLM to self-report. A retrieval-confidence threshold (e.g. abstain if max fused score < 0.3) would be more robust and is planned.

---

## Project structure

| File | Purpose |
|---|---|
| `main.py` | CLI entry point — `ingest`, `query`, `ask` subcommands |
| `load_and_embed.py` | PDF loading, recursive chunking, metadata enrichment, ChromaDB persistence |
| `retrieval.py` | Hybrid dense + BM25 retrieval with min-max normalisation and score fusion |
| `context_builder.py` | Sorts and formats retrieved chunks into an LLM-ready context block with citations |
| `generation.py` | Gemini answer generation; enforces corpus-grounded answering |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for API keys |

---

## Setup

**Prerequisites:** Python 3.10+, a [Gemini API key](https://aistudio.google.com/apikey).

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

`.env` format:
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
```

---

## Usage

### Ingest a PDF

```bash
python main.py ingest "2022 Q3 NVDA.pdf"
```

Options: `--chunk-size 1000`, `--chunk-overlap 200`, `--collection-name kg_rag_single_doc`, `--persist-dir ./chroma_db`, `--model-name all-MiniLM-L6-v2`

### Inspect retrieval (no generation)

```bash
python main.py query "What was NVIDIA's net cash from operating activities in Q3 2022?"
```

Prints each result with its fused score, dense score, BM25 score, document, page, and chunk preview. Use this to verify the retrieval stage before running the full pipeline.

### Full RAG answer

```bash
python main.py ask "What was NVIDIA's net cash from operating activities in Q3 2022?"
```

Output:
```
Question: What was NVIDIA's net cash from operating activities in Q3 2022?

Answer:
According to the Condensed Consolidated Statements of Cash Flows (2022_Q3_NVDA.pdf, Page 8),
NVIDIA's net cash provided by operating activities for the nine months ended October 30, 2022
was $3,393 million...

Sources:
- 2022_Q3_NVDA.pdf Page 8
```

All CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--collection-name` | `kg_rag_single_doc` | ChromaDB collection |
| `--persist-dir` | `./chroma_db` | ChromaDB local path |
| `--top-k` | `5` | Final chunks sent to the LLM |
| `--retrieve-k` | `20` | Candidates per retrieval method before fusion |
| `--dense-weight` | `0.5` | Dense score weight |
| `--sparse-weight` | `0.5` | BM25 score weight |
| `--model-name` | `all-MiniLM-L6-v2` | Sentence Transformers embedding model |
| `--gemini-model` | `GEMINI_MODEL` env var | Gemini model for generation |

---

## Roadmap

- **Multi-document ingestion** — `--append` mode so multiple PDFs share one collection without recreating it.
- **Table-aware extraction** — Separate `pdfplumber` table extraction path that preserves row/column structure as pipe-delimited text before chunking.
- **Cross-encoder reranking** — Rerank top-20 candidates with `ms-marco-MiniLM` before passing top-5 to Gemini.
- **Retrieval-confidence abstention** — Abstain automatically when max fused score falls below a threshold, rather than relying on the LLM prompt.
- **Ragas evaluation** — Automated faithfulness and answer-relevancy scoring against the KG-RAG `qna_data`.