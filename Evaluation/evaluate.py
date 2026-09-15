from retriever import retrieve
from reranker import create_reranker, rerank


TEST_CASES = [
    {
        "question": "What is Delta Lake?",
        "expected_source": "azureDatabricks.pdf"
    },
    {
        "question": "What is Azure Data Factory?",
        "expected_source": "azureDataFactory.pdf"
    },
    {
        "question": "What is Apache Spark?",
        "expected_source": "spark.pdf"
    },
    {
        "question": "What is Azure Synapse Analytics?",
        "expected_source": "azureSynapse.pdf"
    },
    {
        "question": "What is Azure SQL Database?",
        "expected_source": "sqlDatabase.pdf"
    },
    {
        "question": "What is data engineering?",
        "expected_source": "dataEngineering.pdf"
    },
    {
        "question": "What is Kubernetes?",
        "expected_source": None
    }
]


def evaluate():

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    reranker = create_reranker()

    total = len(TEST_CASES)
    correct = 0

    for number, test in enumerate(
        TEST_CASES,
        start=1
    ):

        question = test["question"]
        expected_source = test["expected_source"]

        print("\n" + "-" * 70)

        print(
            f"Test {number}/{total}: "
            f"{question}"
        )

        results = retrieve(
            question,
            candidate_k=10,
            similarity_threshold=0.30,
            max_results=10
        )

        results = rerank(
            question,
            results,
            reranker,
            top_k=5,
            rerank_threshold=0.0
        )

        if expected_source is None:

            if not results:

                print("Result: PASS")
                print(
                    "Correctly found no relevant "
                    "information."
                )

                correct += 1

            else:

                print("Result: FAIL")
                print(
                    "Unexpected results were retrieved."
                )

            continue

        if not results:

            print("Result: FAIL")
            print("No results retrieved.")

            continue

        top_result = results[0]

        actual_source = top_result["source"]

        print(
            f"Expected source: "
            f"{expected_source}"
        )

        print(
            f"Retrieved source: "
            f"{actual_source}"
        )

        print(
            f"Rerank score: "
            f"{top_result['rerank_score']:.4f}"
        )

        if actual_source == expected_source:

            print("Result: PASS")

            correct += 1

        else:

            print("Result: FAIL")

    accuracy = (
        correct / total
    ) * 100

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Passed: {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate()