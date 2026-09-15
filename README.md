# RAG Knowledge Assistant

> A document-grounded Retrieval-Augmented Generation (RAG) system for answering technical questions from a curated PDF knowledge base.

The RAG Knowledge Assistant combines semantic retrieval, keyword search, cross-encoder reranking, evidence validation, and Gemini-based generation to produce grounded answers with document, page, and section references.

---

## 📌 Overview

This project is a learning-focused implementation of a Retrieval-Augmented Generation system built from individual components.

The knowledge base contains technical documents covering:

- Apache Spark
- Azure Databricks
- Data Engineering
- Azure Data Factory
- Azure Synapse Analytics
- Azure SQL Database

Instead of passing the entire document collection to an LLM, the system retrieves the most relevant chunks first and uses those chunks as evidence for answer generation.

Core idea:

    Documents → Retrieval → Evidence → Generation → Grounded Answer

---

## 🏗️ Architecture
<img width="1536" height="1024" alt="Architecture" src="https://github.com/user-attachments/assets/56aacf00-7cb7-4981-aa96-91caa7b9eb39" />


The system consists of two major pipelines.

### 1. Offline Indexing Pipeline

    PDF Documents
          │
          ▼
    Document Loader
          │
          ▼
    Section-Aware Chunking
          │
          ▼
    Sentence Transformer Embeddings
          │
          ▼
    FAISS Vector Index

This pipeline runs when the knowledge base is created or when documents change.

### 2. Online Query & Answering Pipeline

    User Question
          │
          ▼
    Query Rewriting
          │
          ▼
    Hybrid Retrieval
      ┌───┴────┐
      ▼        ▼
    Semantic  Keyword
    Search    Search
      │        │
      └───┬────┘
          ▼
    Candidate Results
          │
          ▼
    Cross-Encoder Reranking
          │
          ▼
    Evidence Check
          │
          ▼
    Context Construction
          │
          ▼
    Gemini Generation
          │
          ▼
    Answer + Sources

---

## ✨ Key Features

- 📄 PDF-based technical knowledge base
- 🧩 Section-aware document chunking
- 🧠 Sentence Transformer embeddings
- 🔎 FAISS semantic vector search
- 🔤 Keyword-based retrieval
- 🔀 Hybrid retrieval
- 📊 Cross-encoder reranking
- 🛡️ Evidence/confidence checking
- ✍️ Domain-aware query rewriting
- 🤖 Gemini-powered answer generation
- 📚 Source attribution
- 🚫 Out-of-scope query handling
- 💬 Streamlit chat interface
- 🔁 Locally reproducible vector indexing

---

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| PDF Processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Reranking | Cross-Encoder |
| LLM | Google Gemini |
| UI | Streamlit |
| Configuration | python-dotenv |

---

# 🔄 End-to-End RAG Flow

## 1. Document Ingestion

Technical PDF documents are placed inside:

    Data/Documents/

`document_loader.py` uses PyMuPDF to extract text page by page while retaining document and page information.

---

## 2. Section-Aware Chunking

`chunker.py` divides extracted text into smaller, section-aware chunks.

Each chunk retains metadata such as:

    text
    document
    document_type
    page
    section

This metadata is later used for source attribution.

---

## 3. Embedding Generation

`embeddings.py` converts each chunk into a numerical vector using:

    all-MiniLM-L6-v2

The embeddings are normalized before being stored in FAISS.

---

## 4. FAISS Vector Store

`vector_store.py` creates a FAISS index using:

    IndexFlatIP

With normalized embeddings, inner-product similarity corresponds to cosine similarity.

The generated files are stored locally:

    index/
    ├── faiss.index
    └── chunks.pkl

The `index/` directory is excluded from Git.

---

## 5. Query Rewriting

`query_rewriter.py` expands important domain terms with related terminology.

For example:

    Delta Lake

can be expanded with:

    transaction log
    ACID
    schema management
    time travel

This helps retrieval match terminology used in the knowledge base.

---

## 6. Hybrid Retrieval

`retriever.py` combines semantic and keyword retrieval.

Semantic Search:
FAISS retrieves chunks based on semantic similarity.

Keyword Search:
`keyword_search.py` identifies chunks containing important query terms.

    Query
      │
      ├───────────────┐
      ▼               ▼
    Semantic       Keyword
    Search         Search
      │               │
      └───────┬───────┘
              ▼
       Candidate Chunks

---

## 7. Cross-Encoder Reranking

`reranker.py` uses:

    cross-encoder/ms-marco-MiniLM-L-6-v2

Each candidate is evaluated using the relationship between:

    Question + Retrieved Chunk

The candidates are then reordered according to relevance.

---

## 8. Evidence Check

`confidence.py` checks whether the retrieved evidence is sufficient.

If relevant evidence cannot be found, the system avoids generating an unsupported answer.

Example:

    User:
    What is Kubernetes?

    Assistant:
    I couldn't find enough reliable information in the knowledge
    base to answer this question.

---

## 9. Context Construction

The highest-ranked chunks are combined into the context supplied to the LLM.

The context retains:

    Document
    Page
    Section
    Retrieved Text

---

## 10. Gemini Generation

`generator.py` sends the question and retrieved context to Gemini.

The generation process is instructed to:

- Use only the supplied context
- Avoid unsupported information
- Answer clearly and concisely
- Indicate when the knowledge base does not contain enough information

---

## 11. Streamlit Response

`app.py` provides the Streamlit chat interface.

The response can include:

- Generated answer
- Source document
- Page number
- Section
- Retrieval scores
- Relevant retrieved text

---



# 💡 Example Questions

    What is Delta Lake and how does it provide ACID transactions?

    What are narrow and wide transformations in Spark?

    What is Azure Data Factory?

    What is the difference between Azure Synapse and Databricks?

    What are the components of a data engineering pipeline?

    What is Azure SQL Database?

For questions outside the knowledge base, the system can return an evidence-based refusal instead of fabricating an answer.

---

# 🚀 Getting Started

## 1. Clone the Repository

    git clone https://github.com/sohailGHUB/RAG_Knowledge_Assistant.git
    cd RAG_Knowledge_Assistant

## 2. Create a Virtual Environment

### Windows

    python -m venv .venv
    .venv\Scripts\activate

## 3. Install Dependencies

    pip install -r requirements.txt

---

# 🔐 Configuration

Create a `.env` file in the project root:

    GEMINI_API_KEY=your_gemini_api_key_here

Never commit `.env` to GitHub.

The repository provides `.env.example` as a configuration template.

---

# 🗂️ Build the Vector Index

Place the PDF documents inside:

    Data/Documents/

Then run:

    python -m src.pipeline.indexer

The indexing process performs:

    PDF Loading
         ↓
    Section-Aware Chunking
         ↓
    Embedding Generation
         ↓
    FAISS Index Creation
         ↓
    Local Index Persistence

The generated `index/` directory is intentionally excluded from Git.

---

# ▶️ Run the Application

Start Streamlit:

    python -m streamlit run src/app.py

If Streamlit's file watcher causes dependency-related issues:

    python -m streamlit run src/app.py --server.fileWatcherType none

---

# 🔁 Rebuilding the Knowledge Base

When documents are added or modified:

    1. Update PDFs
           ↓
    2. Run the indexing pipeline
           ↓
    3. Generate new embeddings
           ↓
    4. Rebuild FAISS index
           ↓
    5. Start the Streamlit application

The vector index can always be recreated from the source documents.

---

# 🖥️ Screenshots

### Main Interface
<img width="1920" height="1080" alt="RAG_UI" src="https://github.com/user-attachments/assets/29cbb588-4e1e-46fc-bf1d-8da88aca8479" />


Additional examples are organized under:

    Screenshots/responses/
    Screenshots/retrieval_info_and_metadata/
    Screenshots/unsupported_queries/

---

# 🎯 Design Principles

### Grounded Generation

The LLM receives retrieved evidence instead of relying only on pretrained knowledge.

### Retrieval Before Generation

Relevant information is retrieved and evaluated before generation.

### Hybrid Search

Semantic and keyword retrieval provide complementary retrieval signals.

### Reranking

A Cross-Encoder improves the ordering of retrieved candidates.

### Evidence Validation

The system can reject questions when sufficient supporting evidence is unavailable.

### Source Attribution

Retrieved content retains document, page, and section metadata.

### Reproducibility

Generated vector artifacts are recreated locally instead of being stored in Git.

---

# ⚠️ Limitations

- The knowledge base is limited to the included documents.
- Retrieval quality depends on document quality and chunking.
- The initial evaluation dataset is small.
- Gemini availability and rate limits can affect generation.
- The FAISS index must be rebuilt when source documents change.
- The project currently uses a local FAISS vector store rather than a production vector database.
- This implementation is primarily a learning and portfolio project rather than a production deployment.

---

# 🔮 Future Improvements

Potential improvements for a future version include:

- LangChain integration
- LangGraph-based orchestration
- Multi-query retrieval
- Advanced query transformation
- Conversation-aware retrieval
- Larger evaluation datasets
- Automated RAG evaluation
- Production vector databases
- Observability and tracing
- Docker containerization
- Azure deployment
- Kubernetes deployment
- MLOps integration

---

# 📌 Version

## RAG V1

V1 intentionally implements the major RAG components individually rather than hiding the workflow behind an orchestration framework.

The core pipeline is:

    Documents
        ↓
    Chunking
        ↓
    Embeddings
        ↓
    Vector Search
        ↓
    Hybrid Retrieval
        ↓
    Reranking
        ↓
    Evidence Check
        ↓
    Context Construction
        ↓
    Gemini
        ↓
    Grounded Answer

A future RAG V2 can introduce frameworks such as LangChain and LangGraph while preserving the concepts learned through this implementation.

---

# 📄 License

This project is licensed under the terms specified in the `LICENSE` file.

---

# 👤 Author

**Sohail**

GitHub: https://github.com/sohailGHUB/RAG_Knowledge_Assistant

---

⭐ If you find this project useful, consider giving the repository a star.
