def has_sufficient_evidence(
    results,
    minimum_results=1,
    minimum_rerank_score=0.0
):
    """
    Determine whether retrieved chunks contain
    sufficient evidence to answer the question.
    """

    if not results:
        return False

    if len(results) < minimum_results:
        return False

    best_score = max(
        result.get("rerank_score", float("-inf"))
        for result in results
    )

    if best_score < minimum_rerank_score:
        return False

    return True