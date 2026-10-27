# 🗄️ Vector Stores: ChromaDB & FAISS

Vector Stores are specialized databases designed to store high-dimensional embedding vectors, index them efficiently, and execute low-latency nearest-neighbor similarity searches for Retrieval-Augmented Generation (RAG).

This directory demonstrates two industry-standard vector stores:
1. **ChromaDB:** An AI-native, developer-friendly open-source vector database supporting local persistence with SQLite and hnswlib.
2. **FAISS (Facebook AI Similarity Search):** A high-performance library for dense vector clustering and similarity search developed by Meta Research.

---

## 📁 Directory Structure

```plaintext
Vectore Store/
├── Chroma.ipynb        # ChromaDB setup, persistence, querying, and retriever usage
├── Faiss.ipynb         # FAISS vector indexing, similarity scores, vector lookups & saving/loading
├── speech.txt          # Sample text dataset (historic speech)
├── chroma_db/          # Persisted Chroma database files (SQLite3 + hnsw index)
├── faiss_index/        # Persisted FAISS index files (index.faiss + index.pkl)
└── README.md           # Documentation for this directory
```

---

## 📑 Detailed Notebook Breakdown

### 1. [`Chroma.ipynb`](Chroma.ipynb) — Chroma Vector Database
- **Provider:** `langchain_chroma.Chroma` / `langchain_community.vectorstores.Chroma`
- **Embedding Model:** Local Ollama embeddings (`nomic-embed-text`)
- **Key Capabilities Covered:**
  - **Document Loading & Chunking:** Ingests `speech.txt` and chunks with `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`.
  - **In-Memory Store:**
    ```python
    from langchain_chroma import Chroma
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
    ```
  - **Local Disk Persistence:** Saves indexed embeddings to disk so they survive process restarts:
    ```python
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    ```
  - **Reloading from Disk:**
    ```python
    db_loaded = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    ```
  - **Similarity Search with Distance Scores:** Computes relevance metrics:
    ```python
    results = vectordb.similarity_search_with_score("democracy and peace")
    for doc, score in results:
        print(f"Score: {score:.4f} | Content: {doc.page_content[:80]}...")
    ```
  - **Conversion to Retriever:**
    ```python
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    ```

---

### 2. [`Faiss.ipynb`](Faiss.ipynb) — Facebook AI Similarity Search
- **Provider:** `langchain_community.vectorstores.FAISS` (via `faiss-cpu`)
- **Embedding Model:** Local Ollama embeddings (`nomic-embed-text`)
- **Key Capabilities Covered:**
  - **Fast In-Memory Indexing:**
    ```python
    from langchain_community.vectorstores import FAISS
    db = FAISS.from_documents(docs, embeddings)
    ```
  - **Standard Similarity Search:**
    ```python
    results = db.similarity_search("What is the speech about?", k=2)
    ```
  - **Similarity Search with Distance Scores:**
    ```python
    docs_and_score = db.similarity_search_with_score("free peoples")
    # Returns (Document, float_distance) tuples (L2 Euclidean distance)
    ```
  - **Direct Vector Search:** Passing raw floating-point embedding vectors directly to the index:
    ```python
    query_vector = embeddings.embed_query("rights of mankind")
    db_score = db.similarity_search_by_vector(query_vector)
    ```
  - **Persisting & Reloading FAISS Index:**
    - Saving:
      ```python
      db.save_local("faiss_index")
      # Generates faiss_index/index.faiss and faiss_index/index.pkl
      ```
    - Safe Reloading:
      ```python
      new_db = FAISS.load_local(
          "faiss_index",
          embeddings,
          allow_dangerous_deserialization=True
      )
      ```

---

## ⚖️ ChromaDB vs. FAISS Comparison

| Metric | ChromaDB | FAISS |
| :--- | :--- | :--- |
| **Primary Focus** | Full-featured Vector DB (Metadata filtering, collections, persistence) | Low-level C++ similarity search library optimized for speed and scale |
| **Storage Backend** | SQLite3 + parquet + HNSW index | Flat binary files (`.faiss` index + `.pkl` metadata) |
| **Metadata Filtering** | Built-in rich SQL-like metadata filtering (`where={"source": "..."}`) | Basic dictionary mapping via pickle |
| **Persistence** | Native `persist_directory` parameter | `save_local()` / `load_local()` |
| **Best For** | Application prototypes, small-to-medium production apps, RAG pipelines | Massive datasets, latency-critical similarity search, GPU-accelerated clusters |

---

## 🛠️ Installation

```bash
pip install langchain-chroma chromadb faiss-cpu langchain-community
```
Make sure Ollama is running if using `nomic-embed-text`:
```bash
ollama run nomic-embed-text
```
