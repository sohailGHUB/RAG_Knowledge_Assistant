from embeddings import create_embedding_model
from vector_store import load_index, search_index
from keyword_search import keyword_search


def retrieve(
    query,
    candidate_k=8,
    similarity_threshold=0.30,
    max_results=5,
    document_type=None
):
    """
    Hybrid retrieval using:

    1. Semantic search with FAISS
    2. Keyword search
    3. Candidate filtering
    """

    index, chunks = load_index()

    model = create_embedding_model()

    query_embedding = model.encode(query)

    # -----------------------------------------
    # Semantic search
    # -----------------------------------------

    semantic_similarities, semantic_indices = search_index(
        index,
        query_embedding,
        candidate_k
    )

    semantic_results = {}

    for similarity, index_position in zip(
        semantic_similarities,
        semantic_indices
    ):

        chunk = chunks[index_position]

        if (
            document_type is not None
            and chunk["document_type"] != document_type
        ):
            continue

        semantic_results[index_position] = {
            "similarity": float(similarity)
        }

    # -----------------------------------------
    # Keyword search
    # -----------------------------------------

    keyword_results = keyword_search(
        query,
        chunks,
        top_k=candidate_k
    )

    keyword_scores = {}

    for result in keyword_results:

        index_position = result["index"]

        chunk = chunks[index_position]

        if (
            document_type is not None
            and chunk["document_type"] != document_type
        ):
            continue

        keyword_scores[index_position] = (
            result["keyword_score"]
        )

    # -----------------------------------------
    # Combine candidates
    # -----------------------------------------

    combined_indices = (
        set(semantic_results.keys())
        | set(keyword_scores.keys())
    )

    results = []

    for index_position in combined_indices:

        chunk = chunks[index_position]

        semantic_score = semantic_results.get(
            index_position,
            {}
        ).get(
            "similarity",
            0.0
        )

        keyword_score = keyword_scores.get(
            index_position,
            0.0
        )

        # -------------------------------------
        # Reject weak keyword-only matches
        # -------------------------------------

        if (
            semantic_score < similarity_threshold
            and keyword_score < 0.75
        ):
            continue

        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "document_type": chunk["document_type"],
                "page": chunk["page"],
                "section": chunk["section"],
                "similarity": semantic_score,
                "keyword_score": keyword_score
            }
        )

    # -----------------------------------------
    # Sort candidates
    # -----------------------------------------

    results.sort(
        key=lambda x: (
            x["similarity"],
            x["keyword_score"]
        ),
        reverse=True
    )

    return results[:max_results]