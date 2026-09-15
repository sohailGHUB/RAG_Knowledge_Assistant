import streamlit as st

from retriever import retrieve
from query_rewriter import rewrite_query
from confidence import has_sufficient_evidence
from generator import generate_answer
from reranker import create_reranker, rerank


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 RAG Knowledge Assistant")

st.caption(
    "Ask questions about the documents in your knowledge base."
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# LOAD RERANKER
# --------------------------------------------------

@st.cache_resource
def load_reranker():
    return create_reranker()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show retrieval details only for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("results")
        ):

            with st.expander(
                "📚 View Sources & Retrieval Details"
            ):

                results = message["results"]

                st.write(
                    f"**Retrieved chunks:** {len(results)}"
                )

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.markdown(
                        f"### Chunk {i}"
                    )

                    st.write(
                        f"**Source:** {result['source']}"
                    )

                    st.write(
                        f"**Document Type:** "
                        f"{result['document_type']}"
                    )

                    st.write(
                        f"**Page:** {result['page']}"
                    )

                    st.write(
                        f"**Section:** {result['section']}"
                    )

                    st.write(
                        f"**FAISS Similarity:** "
                        f"{result['similarity']:.4f}"
                    )

                    st.write(
                        f"**Keyword Score:** "
                        f"{result['keyword_score']:.4f}"
                    )

                    st.write(
                        f"**Rerank Score:** "
                        f"{result['rerank_score']:.4f}"
                    )

                    st.markdown("**Retrieved Text:**")

                    st.info(
                        result["text"]
                    )

                    if i < len(results):
                        st.divider()


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask something about your documents..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    # ----------------------------------------------
    # USER MESSAGE
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # ----------------------------------------------
    # ASSISTANT RESPONSE
    # ----------------------------------------------

    with st.chat_message("assistant"):

        status = st.status(
            "🔎 Searching your knowledge base...",
            expanded=True
        )

        try:

            # --------------------------------------
            # QUERY REWRITING
            # --------------------------------------

            status.write(
                "🧠 Understanding your question..."
            )

            search_query = rewrite_query(
                question
            )


            # --------------------------------------
            # RETRIEVAL
            # --------------------------------------

            status.write(
                "🔎 Finding relevant information..."
            )

            results = retrieve(
                search_query,
                candidate_k=10,
                similarity_threshold=0.30,
                max_results=10
            )


            if not results:

                status.update(
                    label="❌ No relevant information found",
                    state="error"
                )

                answer = (
                    "I couldn't find enough information "
                    "in the knowledge base to answer "
                    "this question."
                )

                st.markdown(answer)

                saved_results = []


            else:

                # ----------------------------------
                # RERANKING
                # ----------------------------------

                status.write(
                    "🎯 Selecting the most relevant information..."
                )

                reranker_model = load_reranker()

                results = rerank(
                    search_query,
                    results,
                    reranker_model,
                    top_k=5,
                    rerank_threshold=0.0
                )


                # ----------------------------------
                # CONFIDENCE CHECK
                # ----------------------------------

                if not has_sufficient_evidence(
                    results,
                    minimum_results=1,
                    minimum_rerank_score=0.0
                ):

                    status.update(
                        label="⚠️ Not enough evidence",
                        state="error"
                    )

                    answer = (
                        "I couldn't find enough reliable "
                        "information in the knowledge base "
                        "to answer this question."
                    )

                    st.markdown(answer)

                    saved_results = results


                else:

                    # --------------------------------
                    # BUILD CONTEXT
                    # --------------------------------

                    context_parts = []

                    for result in results:

                        context_parts.append(
                            f"Source: {result['source']}\n"
                            f"Document Type: "
                            f"{result['document_type']}\n"
                            f"Page: {result['page']}\n"
                            f"Section: {result['section']}\n"
                            f"Text:\n{result['text']}"
                        )

                    context = "\n\n---\n\n".join(
                        context_parts
                    )


                    # --------------------------------
                    # GENERATE ANSWER
                    # --------------------------------

                    status.write(
                        "✍️ Generating your answer..."
                    )

                    answer = generate_answer(
                        question,
                        context
                    )


                    status.update(
                        label="✅ Answer ready",
                        state="complete",
                        expanded=False
                    )

                    st.markdown(answer)

                    saved_results = results


            # --------------------------------------
            # SAVE ASSISTANT MESSAGE
            # --------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "results": saved_results
                }
            )


            # --------------------------------------
            # CURRENT RESPONSE DETAILS
            # --------------------------------------

            if saved_results:

                with st.expander(
                    "📚 View Sources & Retrieval Details"
                ):

                    st.write(
                        f"**Retrieved chunks:** "
                        f"{len(saved_results)}"
                    )

                    for i, result in enumerate(
                        saved_results,
                        start=1
                    ):

                        st.markdown(
                            f"### Chunk {i}"
                        )

                        st.write(
                            f"**Source:** {result['source']}"
                        )

                        st.write(
                            f"**Document Type:** "
                            f"{result['document_type']}"
                        )

                        st.write(
                            f"**Page:** {result['page']}"
                        )

                        st.write(
                            f"**Section:** {result['section']}"
                        )

                        st.write(
                            f"**FAISS Similarity:** "
                            f"{result['similarity']:.4f}"
                        )

                        st.write(
                            f"**Keyword Score:** "
                            f"{result['keyword_score']:.4f}"
                        )

                        st.write(
                            f"**Rerank Score:** "
                            f"{result['rerank_score']:.4f}"
                        )

                        st.markdown(
                            "**Retrieved Text:**"
                        )

                        st.info(
                            result["text"]
                        )

                        if i < len(saved_results):
                            st.divider()


        except Exception:

            status.update(
                label="⚠️ Unable to generate answer",
                state="error"
            )

            answer = (
                "The AI model is temporarily unavailable. "
                "Please try again in a few moments."
            )

            st.warning(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "results": []
                }
            )