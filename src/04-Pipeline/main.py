import numpy as np

from retriever import retrieve
from query_rewriter import rewrite_query
from confidence import has_sufficient_evidence
from generator import generate_answer
from reranker import create_reranker, rerank
from embeddings import create_embedding_model


def determine_candidate_k(question):

    question_lower = question.lower()

    comparison_words = [
        "compare",
        "difference",
        "differences",
        "versus",
        "vs",
        "advantages",
        "disadvantages"
    ]

    broad_question_words = [
        "explain",
        "how does",
        "architecture",
        "best practices",
        "overview",
        "describe"
    ]

    if any(
        word in question_lower
        for word in comparison_words
    ):
        return 12

    if any(
        word in question_lower
        for word in broad_question_words
    ):
        return 10

    return 8


def optimize_context(
    results,
    similarity_threshold=0.90
):
    """
    Remove semantically redundant chunks.

    If two chunks are extremely similar,
    keep the one with the better rerank score.
    """

    if not results:
        return []

    model = create_embedding_model()

    texts = [
        result["text"]
        for result in results
    ]

    embeddings = model.encode(texts)

    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    optimized = []

    for i, result in enumerate(results):

        duplicate = False

        for kept_index in optimized:

            similarity = np.dot(
                embeddings[i],
                embeddings[kept_index]
            )

            if similarity >= similarity_threshold:

                duplicate = True
                break

        if not duplicate:
            optimized.append(i)

    return [
        results[index]
        for index in optimized
    ]


def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"Source: {result['source']}\n"
            f"Document Type: {result['document_type']}\n"
            f"Page: {result['page']}\n"
            f"Section: {result['section']}\n"
            f"Text:\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_source_reference(result):

    return (
        f"[{result['source']}, "
        f"Page {result['page']}, "
        f"Section: {result['section']}]"
    )


if __name__ == "__main__":

    question = input("Ask a question: ")

    # -----------------------------------------
    # Query rewriting
    # -----------------------------------------

    print("\nPreparing query...")

    search_query = rewrite_query(question)

    print(
        f"Search query: {search_query}"
    )

    # -----------------------------------------
    # Retrieval
    # -----------------------------------------

    print("\nSearching documents...")

    candidate_k = determine_candidate_k(
        search_query
    )

    print(
        f"Retrieving {candidate_k} candidate chunks..."
    )

    results = retrieve(
        search_query,
        candidate_k=candidate_k,
        similarity_threshold=0.30,
        max_results=candidate_k
    )

    if not results:

        print(
            "\nI don't have enough information "
            "in the provided documents."
        )

        exit()

    # -----------------------------------------
    # Reranking
    # -----------------------------------------

    print("\nReranking results...")

    reranker = create_reranker()

    results = rerank(
        search_query,
        results,
        reranker,
        top_k=5,
        rerank_threshold=0.0
    )

    if not results:

        print(
            "\nI don't have enough information "
            "in the provided documents."
        )

        exit()

    # -----------------------------------------
    # Semantic deduplication
    # -----------------------------------------

    before_optimization = len(results)

    results = optimize_context(results)

    after_optimization = len(results)

    print(
        f"\nContext optimization: "
        f"{before_optimization} → "
        f"{after_optimization} chunks"
    )

    # -----------------------------------------
    # Evidence check
    # -----------------------------------------

    if not has_sufficient_evidence(
        results,
        minimum_results=1,
        minimum_rerank_score=0.0
    ):

        print(
            "\nI don't have enough information "
            "in the provided documents."
        )

        exit()

    # -----------------------------------------
    # Display results
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RETRIEVED CHUNKS")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n[{i}] {result['source']}"
        )

        print(
            f"Document Type: "
            f"{result['document_type']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Section: "
            f"{result['section']}"
        )

        print(
            f"FAISS Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Keyword Score: "
            f"{result['keyword_score']:.4f}"
        )

        print(
            f"Rerank Score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Source Reference: "
            f"{build_source_reference(result)}"
        )

        print("-" * 60)

        print(
            result["text"][:500]
        )

    # -----------------------------------------
    # Build context
    # -----------------------------------------

    context = build_context(results)

    print(
        f"\nContext contains "
        f"{len(results)} chunks."
    )

    # -----------------------------------------
    # Generation
    # -----------------------------------------

    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        context
    )

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(answer)

    # -----------------------------------------
    # Sources
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. {build_source_reference(result)}"
        )