# 🧬 Embedding Techniques in LangChain

Embeddings are numerical vector representations of textual data where semantic similarity is mapped to geometric proximity in a high-dimensional vector space.

This directory explores three major embedding providers and paradigms:
1. **Commercial Cloud API:** OpenAI (`text-embedding-3-large` with custom dimensions)
2. **Open-Source Local Framework:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
3. **Self-Hosted Local Inference:** Ollama (`nomic-embed-text`)

---

## 📁 Directory Structure

```plaintext
Embeddings/
├── embedding.ipynb         # OpenAI Embeddings + Custom Dimensions + ChromaDB integration
├── huggingface.ipynb       # Hugging Face Sentence Transformers (Local & Offline)
├── ollamaembedding.ipynb   # Ollama Local Embeddings (nomic-embed-text)
├── speech.txt              # Sample historical text document for chunking tests
└── README.md               # Documentation for this directory
```

---

## 🔍 Detailed Notebook Breakdown

### 1. [`embedding.ipynb`](embedding.ipynb) — OpenAI Embeddings
- **Provider:** OpenAI API (`langchain_openai.OpenAIEmbeddings`)
- **Key Models:** `text-embedding-3-large` (3072 dimensions by default) and `text-embedding-3-small` (1536 dimensions).
- **Custom Dimension Scaling:** Demonstrates OpenAI's MRL (Matryoshka Representation Learning) capability to downscale embeddings (e.g., to 1024 or 512 dimensions) without significant accuracy degradation.
  ```python
  from langchain_openai import OpenAIEmbeddings

  # Embedding with truncated dimension
  embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
  query_vec = embeddings.embed_query("Sample text")
  print("Vector Dimension:", len(query_vec))  # 1024
  ```
- **Document Chunking & Vector Store Ingestion:** Loads `speech.txt`, splits with `RecursiveCharacterTextSplitter`, embeds chunks, and builds a searchable `Chroma` database.

### 2. [`huggingface.ipynb`](huggingface.ipynb) — Hugging Face Sentence Transformers
- **Provider:** Hugging Face via `langchain_huggingface.HuggingFaceEmbeddings`
- **Default Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Features:**
  - Zero API cost: Runs locally on CPU/GPU.
  - Generates dense vector representations using PyTorch and Sentence-Transformers.
  - Supports both single query embedding (`embed_query`) and bulk document embedding (`embed_documents`).
  ```python
  from langchain_huggingface import HuggingFaceEmbeddings

  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vector = embeddings.embed_query("GenAI vector embedding test")
  print("MiniLM Dimensions:", len(vector))  # 384
  ```

### 3. [`ollamaembedding.ipynb`](ollamaembedding.ipynb) — Self-Hosted Ollama Embeddings
- **Provider:** Ollama (`langchain_community.embeddings.OllamaEmbeddings` or `langchain_ollama.OllamaEmbeddings`)
- **Default Model:** `nomic-embed-text` (512 / 768 dimensions)
- **Features:**
  - Completely private and offline execution on local machines.
  - No external API keys or rate limits.
  - Compatible with various Ollama models (`nomic-embed-text`, `all-minilm`, `mxbai-embed-large`).
  ```python
  from langchain_community.embeddings import OllamaEmbeddings

  embeddings = OllamaEmbeddings(model="nomic-embed-text")
  vectors = embeddings.embed_documents([
      "Alpha is the first letter of the Greek alphabet.",
      "Beta is the second letter of the Greek alphabet."
  ])
  ```

---

## 📊 Comparison Matrix

| Feature | OpenAI (`text-embedding-3-large`) | Hugging Face (`all-MiniLM-L6-v2`) | Ollama (`nomic-embed-text`) |
| :--- | :--- | :--- | :--- |
| **Hosting** | Cloud API (OpenAI) | Local (PyTorch / Transformers) | Local Server (Ollama daemon) |
| **Cost** | Pay-per-token API | 100% Free / Open Source | 100% Free / Self-hosted |
| **Dimensions** | 3072 (configurable down to 256) | 384 | 768 (model dependent) |
| **Privacy** | Data sent to OpenAI | 100% On-device / Private | 100% On-device / Private |
| **Hardware Reqs** | Minimal (Network only) | Moderate (CPU/GPU) | Moderate (Local RAM/VRAM) |
| **Speed** | Network latency dependent | Extremely fast on CPU/GPU | Fast local inference |

---

## 🚀 Setup & Execution

### 1. Environment Keys (in `.env`)
```env
OPEN_AI_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 2. Ollama Setup (for `ollamaembedding.ipynb`)
Download and start Ollama, then pull the embedding model:
```bash
ollama run nomic-embed-text
```
