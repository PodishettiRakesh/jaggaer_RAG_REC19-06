# Multi-Document Question Answering System over KG-RAG Dataset

## 1. Problem Understanding

### Objective

Build a Retrieval-Augmented Generation (RAG) system capable of answering questions over large collections of documents, specifically:

1. SEC 10-Q Financial Reports
2. NTSB Aviation Incident Reports

The system must:

* Answer natural language questions.
* Support single-document and multi-document reasoning.
* Handle table-heavy documents.
* Provide grounded answers with citations.
* Detect when the corpus does not contain enough information to answer a question.
* Include an evaluation framework to measure performance.

---

# 2. Key Challenges

## Challenge 1: Single-Chunk Questions

Example:

"What was Microsoft's net cash from operating activities in Q3 2022?"

The answer exists in one document and one chunk.

### Requirement

Retrieve the correct chunk and generate the answer.

---

## Challenge 2: Multi-Chunk Questions

Example:

"For Amazon's Q1 2023, how does share repurchase information relate to equity discussion?"

The answer exists across multiple sections of the same document.

### Requirement

Retrieve multiple relevant chunks and combine them.

---

## Challenge 3: Multi-Document Questions

Example:

"How has Apple's iPhone revenue changed across quarters?"

The answer requires information from multiple quarterly reports.

### Requirement

Retrieve information from multiple documents and synthesize a single answer.

---

## Challenge 4: Table-Heavy Documents

Financial reports contain:

* Balance Sheets
* Income Statements
* Cash Flow Statements

NTSB reports also contain structured tabular fields.

Traditional text chunking often breaks table structure and reduces retrieval quality.

---

# 3. Proposed Architecture

Question
↓
Query Classification
↓
Hybrid Retrieval
(BM25 + Vector Search)
↓
Reranking
↓
Context Construction
↓
LLM Answer Generation
↓
Grounded Answer + Citations

---

# 4. System Components

## Component 1: Document Ingestion

### Purpose

Load and preprocess PDF documents.

### Method

Use:

* pdfplumber
* PyMuPDF
* Unstructured

### Output

Document objects containing:

* text
* page number
* document name
* metadata

Example:

{
"document":"2022_Q1_AAPL.pdf",
"page":12,
"content":"..."
}

---

## Component 2: Chunking Strategy

### Purpose

Split documents into retrievable units.

### Method

Recursive Character Text Splitter

Parameters:

Chunk Size: 1000
Chunk Overlap: 200

### Why

Maintains context while keeping chunks retrieval-friendly.

---

## Component 3: Metadata Enrichment

Store metadata with every chunk.

Example:

{
"company":"AAPL",
"quarter":"2022_Q1",
"page":12,
"chunk_id":"chunk_45"
}

### Why

Allows metadata filtering before retrieval.

---

## Component 4: Hybrid Retrieval

### Method

Combine:

1. Dense Retrieval
2. Sparse Retrieval (BM25)

### Dense Retrieval

Captures semantic meaning.

Example:

"cash generated from operations"

matches

"net cash from operating activities"

---

### Sparse Retrieval (BM25)

Captures exact keywords.

Example:

"share repurchase"

"operating cash flow"

"deferred revenue"

---

### Final Score

Final Score =
0.5 × Dense Score
+
0.5 × BM25 Score

### Why

Financial datasets benefit significantly from exact keyword matching.

---

## Component 5: Reranking

### Method

Retrieve Top-20 chunks.

Use Cross-Encoder Reranker.

Examples:

* BGE Reranker
* Cohere Rerank

Select Top-5 chunks.

### Why

Improves precision by removing irrelevant chunks.

---

## Component 6: Context Builder

Build final context from retrieved chunks.

Example:

Question:
"Compare Apple's revenue across quarters."

Context:

Q1 Revenue Chunk
Q2 Revenue Chunk
Q3 Revenue Chunk
Q4 Revenue Chunk

### Why

Allows multi-document reasoning.

---

## Component 7: Answer Generation

Prompt structure:

You are a financial and aviation analyst.

Answer using ONLY the provided context.

If information is insufficient, explicitly state that the answer cannot be determined from the corpus.

Provide source citations.

### Why

Reduces hallucinations.

---

## Component 8: Citation Generation

Every answer includes:

Document Name
Page Number

Example:

Sources:

* 2022_Q1_AAPL.pdf Page 15
* 2022_Q2_AAPL.pdf Page 18

### Why

Supports verifiability and grounding.

---

# 5. Handling Table-Heavy Documents

## Problem

Naive chunking destroys table structure.

Example:

Revenue | 100
Profit | 50

may become split across chunks.

---

## Proposed Solution

Extract tables separately.

Store tables as independent retrieval units.

Example:

{
"type":"table",
"table_name":"Cash Flow Statement",
"content":"..."
}

---

## Future Enhancement

Create a dedicated table index.

Text Questions → Text Index

Financial Metric Questions → Table Index

---

# 6. Multi-Document Reasoning Strategy

## Detection

Query classifier identifies:

* compare
* trend
* across quarters
* across companies

as Multi-Document questions.

---

## Retrieval

Increase Top-K retrieval.

Retrieve chunks from multiple documents.

---

## Aggregation

Group retrieved chunks by:

* company
* quarter
* document

Construct chronological context.

---

## Benefit

Improves performance on Tier-3 questions where answers span multiple reports.

---

# 7. Unanswerable Question Detection

## Method

Confidence-based retrieval.

If:

Maximum Retrieval Score < Threshold

Return:

"Insufficient evidence found in the corpus."

### Why

Avoids hallucinated answers.

---

# 8. Evaluation Framework

The evaluation framework is the most important part of the system.

---

## Metric 1: Retrieval Recall@K

Measures whether relevant chunks were retrieved.

Formula:

Relevant Retrieved
/
Total Relevant Chunks

Metrics:

* Recall@5
* Recall@10

---

## Metric 2: Answer Correctness

Compare generated answer against human-reviewed answer.

Methods:

* Exact Match
* LLM-as-Judge

Score Range:

0-5

---

## Metric 3: Citation Accuracy

Measures whether cited documents actually contain supporting evidence.

Formula:

Correct Citations
/
Total Citations

---

## Metric 4: Grounding Score

Measures factual support.

Formula:

Supported Claims
/
Total Claims

Example:

Answer contains 5 claims.

Only 4 supported by retrieved evidence.

Grounding Score = 4/5

---

## Metric 5: Multi-Document Coverage

Used for cross-document questions.

Formula:

Documents Retrieved
/
Documents Required

Example:

Needed:

Q1
Q2
Q3
Q4

Retrieved:

Q1
Q2
Q3

Coverage = 3/4

---

## Metric 6: Unanswerable Detection Accuracy

Formula:

Correct Abstentions
/
Total Unanswerable Questions

Measures whether the system appropriately refuses unsupported questions.

---

# 9. Technology Stack

Document Processing

* PyMuPDF
* pdfplumber

Embedding Model

* OpenAI Embeddings
  or
* BGE Embeddings

Vector Database

* ChromaDB

Sparse Retrieval

* BM25

Framework

* LangChain

LLM

* GPT-4o-mini

---

# 10. Why No LangGraph or CrewAI?

This problem is fundamentally:

* Retrieval
* Search
* Ranking
* Grounding
* Evaluation

It is NOT an agentic workflow problem.

Using CrewAI or multi-agent systems would increase complexity without significantly improving evaluation metrics.

A simple, explainable RAG pipeline provides better ROI within the 75-minute constraint.

---

# 11. Future Improvements

1. Knowledge Graph Construction
2. Table-Aware Retrieval
3. Query Routing
4. Graph RAG
5. Cross-Document Reasoning Agent
6. Adaptive Retrieval Strategies
7. Automated Evaluation Dashboard

---

# Conclusion

The proposed system focuses on the evaluation criteria emphasized in the problem statement:

* Correctness
* Grounding
* Citation Quality
* Multi-Document Reasoning
* Retrieval Performance
* Handling Unanswerable Questions

The architecture intentionally prioritizes retrieval quality and evaluation rigor over agentic orchestration, making it suitable for both the coding round and the subsequent technical discussion.
