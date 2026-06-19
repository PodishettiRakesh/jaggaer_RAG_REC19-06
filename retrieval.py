from typing import Dict, List

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from load_and_embed import collection_to_documents


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum, maximum = min(values), max(values)
    if maximum == minimum:
        return {key: 1.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


def dense_search(collection, query: str, model_name: str, top_k: int = 20) -> Dict[str, object]:
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    return collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def bm25_scores(documents: List[Dict[str, object]], query: str) -> Dict[str, float]:
    if not documents:
        return {}

    texts = [doc["document"] for doc in documents]
    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize(query))

    return {doc["id"]: float(score) for doc, score in zip(documents, scores)}


def hybrid_search(
    collection,
    query: str,
    model_name: str,
    top_k: int = 5,
    retrieve_k: int = 20,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> List[Dict[str, object]]:
    """
    Retrieve chunks using weighted fusion of dense vector search and BM25.

    Final Score = dense_weight * normalized_dense_score + sparse_weight * normalized_bm25_score
    """
    documents = collection_to_documents(collection)
    if not documents:
        return []

    doc_by_id = {doc["id"]: doc for doc in documents}

    dense_results = dense_search(collection, query, model_name, top_k=retrieve_k)
    dense_raw_scores = {
        doc_id: 1.0 / (1.0 + distance)
        for doc_id, distance in zip(dense_results["ids"][0], dense_results["distances"][0])
    }

    sparse_raw_scores = bm25_scores(documents, query)
    sparse_ranked_ids = [
        doc_id
        for doc_id, _ in sorted(sparse_raw_scores.items(), key=lambda item: item[1], reverse=True)[:retrieve_k]
    ]

    candidate_ids = set(dense_raw_scores) | set(sparse_ranked_ids)
    dense_candidates = {doc_id: dense_raw_scores.get(doc_id, 0.0) for doc_id in candidate_ids}
    sparse_candidates = {doc_id: sparse_raw_scores.get(doc_id, 0.0) for doc_id in candidate_ids}

    dense_normalized = _min_max_normalize(dense_candidates)
    sparse_normalized = _min_max_normalize(sparse_candidates)

    results = []
    for doc_id in candidate_ids:
        doc = doc_by_id[doc_id]
        dense_score = dense_normalized[doc_id]
        sparse_score = sparse_normalized[doc_id]
        final_score = (dense_weight * dense_score) + (sparse_weight * sparse_score)
        results.append(
            {
                "id": doc_id,
                "document": doc["document"],
                "metadata": doc["metadata"],
                "score": final_score,
                "dense_score": dense_score,
                "bm25_score": sparse_score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]
