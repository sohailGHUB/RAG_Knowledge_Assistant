from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def create_reranker():
    return CrossEncoder(MODEL_NAME)


def rerank(
    query,
    results,
    model,
    top_k=5,
    rerank_threshold=None
):
    if not results:
        return []

    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = model.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):

        score = float(score)

        # Reject chunks that are not relevant enough
        if (
            rerank_threshold is not None
            and score < rerank_threshold
        ):
            continue

        result = result.copy()
        result["rerank_score"] = score

        reranked_results.append(result)

    # Highest rerank score first
    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]