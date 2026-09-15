import re


def tokenize(text):
    """
    Convert text into lowercase words.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


def keyword_search(
    query,
    chunks,
    top_k=5
):
    """
    Simple keyword-based retrieval.

    Scores chunks based on how many query terms
    appear in the chunk.
    """

    query_words = tokenize(query)

    if not query_words:
        return []

    scored_chunks = []

    for index, chunk in enumerate(chunks):

        chunk_words = tokenize(chunk["text"])

        common_words = query_words.intersection(
            chunk_words
        )

        if not common_words:
            continue

        score = len(common_words) / len(query_words)

        scored_chunks.append(
            {
                "index": index,
                "keyword_score": score
            }
        )

    scored_chunks.sort(
        key=lambda x: x["keyword_score"],
        reverse=True
    )

    return scored_chunks[:top_k]