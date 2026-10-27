# 🦜️🔗 LangChain & Generative AI Masterclass

A comprehensive, production-ready repository containing tutorials, hands-on notebooks, scripts, and deployed applications covering the complete **LangChain (v0.2+)** and **Generative AI** landscape.

This repository walks you through the entire GenAI development lifecycle — from raw document ingestion and smart text chunking to dense vector embeddings, vector databases, LangChain Expression Language (LCEL), multi-turn stateful chatbots, Conversational RAG (Retrieval-Augmented Generation), and deployment with **Streamlit** and **FastAPI / LangServe**.

---

## 🏗️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph DataIngestion["1. Data Ingestion & Splitting"]
        D1[Plain Text / speech.txt] --> L1[TextLoader]
        D2[PDF / attention.pdf] --> L2[PyPDFLoader]
        D3[Web Articles / HTML] --> L3[WebBaseLoader / HTMLHeaderTextSplitter]
        D4[ArXiv / Wikipedia] --> L4[ArxivLoader / WikipediaLoader]
        D5[JSON / OpenAPI Specs] --> L5[RecursiveJsonSplitter]
        L1 & L2 & L3 & L4 & L5 --> Splitter[RecursiveCharacterTextSplitter]
    end

    subgraph VectorEcosystem["2. Embeddings & Vector Storage"]
        Splitter --> EmbedChoice{Choose Embeddings}
        EmbedChoice -->|Cloud API| E1[OpenAI text-embedding-3-large]
        EmbedChoice -->|Local Open-Source| E2[HuggingFace all-MiniLM-L6-v2]
        EmbedChoice -->|Self-Hosted Offline| E3[Ollama nomic-embed-text]
        E1 & E2 & E3 --> VStore{Vector Store}
        VStore -->|In-Memory / File Index| VS1[(FAISS Index)]
        VStore -->|Persisted SQLite + HNSW| VS2[(ChromaDB)]
    end

    subgraph RetrievalRAG["3. Advanced Retrieval & Context"]
        VS1 & VS2 --> Retriever[VectorStoreRetriever / MMR]
        Retriever --> HRetriever[create_history_aware_retriever]
        History[(ChatMessageHistory Store)] <-->|Session History| HRetriever
        HRetriever --> DocChain[create_stuff_documents_chain]
    end

    subgraph LLMOrchestration["4. LCEL & Model Execution"]
        DocChain --> LCEL[LCEL Pipeline: Prompt | LLM | StrOutputParser]
        LCEL --> Models{LLM Providers}
        Models --> M1[Groq Cloud: LLaMA3 / Qwen]
        Models --> M2[Google Gemini: 1.5 Flash / Pro]
        Models --> M3[Ollama Local: Gemma 2B / Mistral]
    end

    subgraph ObservabilityDeployment["5. Deployment & Observability"]
        LCEL --> StreamlitApp[Streamlit Web App: app.py]
        LCEL --> LangServeApp[FastAPI + LangServe: LCEL/serve.py]
        LCEL -.->|Tracing & Monitoring| LangSmith[[LangSmith Observability]]
    end
```

---

## 📂 Repository Structure

```plaintext
LANGCHAIN/
│
├── 📁 1.1-openai/                  # Foundational LangChain & Simple RAG App
│   ├── GettingStarted.ipynb        # Chat models, ChatPromptTemplate, StrOutputParser
│   ├── Simpleapp.ipynb             # Web scraping, FAISS indexing, Gemini RAG chain
│   └── README.md                   # Dedicated documentation for 1.1-openai
│
├── 📁 Embeddings/                  # Embedding Models & Vector Representation
│   ├── embedding.ipynb             # OpenAI Embeddings (text-embedding-3-large & MRL)
│   ├── huggingface.ipynb           # Hugging Face Sentence Transformers (all-MiniLM-L6-v2)
│   ├── ollamaembedding.ipynb       # Ollama local embeddings (nomic-embed-text)
│   ├── speech.txt                  # Sample historical text dataset
│   └── README.md                   # Dedicated documentation for Embeddings
│
├── 📁 LCEL/                        # LangChain Expression Language & API Serving
│   ├── serve.py                    # Production FastAPI server powered by LangServe & Groq
│   ├── simplellmLCEL.ipynb         # LCEL chaining syntax, stream/batch/invoke, multi-provider
│   └── README.md                   # Dedicated documentation for LCEL & LangServe
│
├── 📁 Vectore Store/               # Vector Databases & Nearest-Neighbor Search
│   ├── Chroma.ipynb                # ChromaDB collection management, persistence & querying
│   ├── Faiss.ipynb                 # FAISS vector indexing, L2 distance score, serialization
│   ├── speech.txt                  # Reference speech document
│   ├── chroma_db/                  # Local SQLite and HNSW vector index files
│   ├── faiss_index/                # Serialized FAISS index files (.faiss & .pkl)
│   └── README.md                   # Dedicated documentation for Vector Stores
│
├── 📄 app.py                       # Interactive Streamlit Web UI (Ollama + LangSmith)
├── 📄 attention.pdf                # Reference research paper: "Attention Is All You Need"
├── 📄 speech.txt                   # Reference text: Woodrow Wilson 1917 Address to Congress
├── 📄 requirements.txt             # Python project dependencies
├── 📄 .env                         # Environment variables configuration (API keys)
│
├── 📓 CharacterTextsplitter.ipynb          # Basic fixed-length character chunking
├── 📓 RecursiveCharactertextsplitter.ipynb # Semantic boundary-aware recursive chunking
├── 📓 HTMLTextsplitter.ipynb               # Structural HTML tag-based chunking
├── 📓 RecursiveJsonSplitter.ipynb          # Structured JSON data chunking preserving keys
├── 📓 DataIngestion.ipynb                  # Multi-source document loaders (PDF, Web, Arxiv, Wiki)
├── 📓 vectorretriever.ipynb                # Document abstractions, VectorStores, and Retrievers
├── 📓 chatbots.ipynb                       # Stateful conversational chatbot with session history & trimming
└── 📓 conversationqa.ipynb                 # Full Conversational RAG with history-aware query re-writing
```

---

## 🧭 Comprehensive Module Walkthrough

### 1. Data Ingestion & Document Loaders ([`DataIngestion.ipynb`](DataIngestion.ipynb))
LangChain standardizes diverse data sources into uniform `Document(page_content, metadata)` objects:
- **`TextLoader`:** Ingests local UTF-8 plain text files (`speech.txt`).
- **`PyPDFLoader`:** Reads binary PDF documents (`attention.pdf`), extracting page-by-page text content and pagination metadata.
- **`WebBaseLoader`:** Scrapes HTML web pages with BeautifulSoup integration (`bs4`).
- **`ArxivLoader`:** Directly queries the arXiv scientific preprint repository by query or paper ID, downloading paper metadata and abstracts.
- **`WikipediaLoader`:** Queries Wikipedia articles dynamically with localized language parameters and article limits.

---

### 2. Text Splitting Strategies

| Notebook | Splitter Class | Splitting Strategy | Primary Use Case |
| :--- | :--- | :--- | :--- |
| [`CharacterTextsplitter.ipynb`](CharacterTextsplitter.ipynb) | `CharacterTextSplitter` | Splits on raw character count with defined separator and overlap. | Simple flat text where layout structure is irrelevant. |
| [`RecursiveCharactertextsplitter.ipynb`](RecursiveCharactertextsplitter.ipynb) | `RecursiveCharacterTextSplitter` | Recursively checks list of separators: `["\n\n", "\n", " ", ""]`. Keeps paragraphs and sentences intact. | **Industry standard / Recommended** for general prose, articles, and research papers. |
| [`HTMLTextsplitter.ipynb`](HTMLTextsplitter.ipynb) | `HTMLHeaderTextSplitter` | Splits on structural HTML tags (`<h1>`, `<h2>`, `<h3>`), preserving hierarchy in chunk metadata. | Web scrapers, documentation sites, and structured HTML articles. |
| [`RecursiveJsonSplitter.ipynb`](RecursiveJsonSplitter.ipynb) | `RecursiveJsonSplitter` | Traverses nested JSON objects, chunking by node size while retaining parent key structures. | OpenAPI schemas, JSON configuration files, REST API payloads. |

---

### 3. Embeddings Ecosystem ([`Embeddings/`](Embeddings/))
Comprehensive exploration of vector representation algorithms:
- **OpenAI Embeddings ([`Embeddings/embedding.ipynb`](Embeddings/embedding.ipynb)):** Uses `text-embedding-3-large`. Demonstrates Matryoshka Representation Learning (MRL) for custom vector dimensions (e.g., downsizing from 3072 to 1024/512).
- **Hugging Face Sentence Transformers ([`Embeddings/huggingface.ipynb`](Embeddings/huggingface.ipynb)):** Local, open-source embeddings using `all-MiniLM-L6-v2` (384-dimensional). Fast CPU/GPU inference with zero API costs.
- **Ollama Offline Embeddings ([`Embeddings/ollamaembedding.ipynb`](Embeddings/ollamaembedding.ipynb)):** Runs completely locally on hardware via `nomic-embed-text` without external network dependency.

👉 *See [Embeddings README](Embeddings/README.md) for detailed code and comparison.*

---

### 4. Vector Stores & Retrievers ([`Vectore Store/`](Vectore%20Store/) & [`vectorretriever.ipynb`](vectorretriever.ipynb))
- **ChromaDB ([`Vectore Store/Chroma.ipynb`](Vectore%20Store/Chroma.ipynb)):** Stores vectors locally with SQLite backend and HNSW indexing. Demonstrates persistence across sessions using `persist_directory="./chroma_db"`.
- **FAISS ([`Vectore Store/Faiss.ipynb`](Vectore%20Store/Faiss.ipynb)):** High-speed vector similarity engine. Covers L2 Euclidean distance scoring, searching directly with vector floats (`similarity_search_by_vector`), and saving/loading indexes (`index.faiss` and `index.pkl`).
- **Retriever Abstractions ([`vectorretriever.ipynb`](vectorretriever.ipynb)):** Explains why Vector Stores are not directly Runnables and converts them via `.as_retriever(search_type="similarity", search_kwargs={"k": 2})` to integrate directly into LCEL chains.

👉 *See [Vector Store README](Vectore%20Store/README.md) for detailed setup and API details.*

---

### 5. LangChain Expression Language (LCEL) & Serving ([`LCEL/`](LCEL/))
- **Declarative Composition ([`LCEL/simplellmLCEL.ipynb`](LCEL/simplellmLCEL.ipynb)):** Composing prompts, models, and parsers using the pipe syntax:
  ```python
  chain = prompt | model | StrOutputParser()
  ```
- **Production API Server ([`LCEL/serve.py`](LCEL/serve.py)):** Deploys a translation pipeline powered by **Groq** (`qwen/qwen3.8-27b`) via **FastAPI** and **LangServe**:
  - Exposes interactive UI at `http://localhost:8000/chain/playground/`
  - Interactive Swagger documentation at `http://localhost:8000/docs`

👉 *See [LCEL README](LCEL/README.md) for endpoint documentation and client invocation.*

---

### 6. Conversational Chatbots with History ([`chatbots.ipynb`](chatbots.ipynb))
Builds stateful chatbots capable of multi-turn dialogues:
- **Session Management:** Stores conversational state in-memory keyed by `session_id` using `ChatMessageHistory`.
- **`RunnableWithMessageHistory`:** Wraps LCEL chains to automatically inject and append message histories.
- **`MessagesPlaceholder`:** Injects conversation history into system/user prompts.
- **Context Window Management (`trim_messages`):** Prevents token overflow by dynamically trimming older messages while keeping critical system instructions and recent conversation context:
  ```python
  from langchain_core.messages import trim_messages

  trimmer = trim_messages(
      max_tokens=70,
      strategy="last",
      token_counter=model,
      include_system=True,
      allow_partial=False
  )
  ```

---

### 7. Conversational RAG with Chat History ([`conversationqa.ipynb`](conversationqa.ipynb))
An enterprise-grade Retrieval-Augmented Generation pipeline handling conversational context:
1. **Contextualizing the Question:** When a user asks a follow-up (e.g., *"How do we achieve it?"* referring to *"Self-Reflection"*), `create_history_aware_retriever` re-writes the query into a standalone query using the chat history.
2. **Stuff Documents Chain:** `create_stuff_documents_chain` inserts retrieved documents into a context prompt.
3. **Retrieval Chain Integration:** Combines history-aware retrieval with the document answering chain via `create_retrieval_chain`.
4. **Session Persistence:** Manages multi-user chat sessions using `RunnableWithMessageHistory`.

---

### 8. Interactive Streamlit Web Application ([`app.py`](app.py))
A lightweight, reactive web application for querying local LLMs:
- **Model:** Local Ollama model (`gemma:2b`).
- **Chain:** `prompt | llm | StrOutputParser()`.
- **LangSmith Tracing:** Logs every query, latency, token count, and execution trace to LangSmith.

---

## ⚙️ Prerequisites & Environment Setup

### 1. Clone & Set Up Python Virtual Environment
```bash
# Clone the repository (if applicable)
cd LANGCHAIN

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory (or update the existing one) with the following keys:

```env
# Cloud Model Providers
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=AIzaSy_your_google_gemini_api_key_here
OPEN_AI_KEY=sk-proj-your_openai_api_key_here
HF_TOKEN=hf_your_huggingface_token_here

# LangSmith Observability & Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGCHAIN_PROJECT=Langchain-Tutorials
```

### 4. Install & Run Ollama (for Local Models)
If using offline Ollama models (`gemma:2b`, `nomic-embed-text`):
1. Download Ollama from [ollama.com](https://ollama.com/).
2. Pull the required models:
   ```bash
   ollama pull gemma:2b
   ollama pull nomic-embed-text
   ```

---

## 🚀 How to Run the Projects

### 1. Run the Streamlit Chatbot Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the FastAPI / LangServe REST API Server
```bash
python LCEL/serve.py
```
- API Docs: `http://localhost:8000/docs`
- LangServe Playground: `http://localhost:8000/chain/playground/`

### 3. Launch the Jupyter Notebooks
```bash
jupyter lab
# OR
jupyter notebook
```
Open any of the `.ipynb` files to experiment interactively with step-by-step execution.

---

## 📊 Summary of Dedicated Subfolder Documentation

| Subfolder | Focus Area | Detailed Guide Link |
| :--- | :--- | :--- |
| **`1.1-openai/`** | Google Gemini, LCEL Primitives, Web RAG Pipeline | [📖 1.1-openai README](1.1-openai/README.md) |
| **`Embeddings/`** | OpenAI vs. Hugging Face vs. Ollama Embeddings | [📖 Embeddings README](Embeddings/README.md) |
| **`LCEL/`** | LCEL Syntax, Streaming, Batching, FastAPI & LangServe | [📖 LCEL README](LCEL/README.md) |
| **`Vectore Store/`** | ChromaDB & FAISS Indexing, Persistence, Distance Metrics | [📖 Vector Store README](Vectore%20Store/README.md) |

---

## 🔍 Observability with LangSmith

This repository has built-in integration with **LangSmith**. Whenever `LANGCHAIN_TRACING="true"` or `LANGCHAIN_TRACING_V2="true"` is enabled:
- **Trace Latency:** Measure time taken per component (prompt formatting, embedding calculation, vector retrieval, LLM generation).
- **Inspect Inputs/Outputs:** View exact token inputs, system messages, retrieved documents, and raw model outputs.
- **Debug Chains:** Pinpoint failure points in complex chains and check token usage per call.

Log in to [smith.langchain.com](https://smith.langchain.com/) to view your traces under the project name specified in your `.env`.

---

## 🛡️ License & Acknowledgements

- **LangChain:** [LangChain Documentation](https://python.langchain.com/)
- **Ollama:** [Ollama Documentation](https://ollama.com/)
- **Meta FAISS:** [FAISS GitHub](https://github.com/facebookresearch/faiss)
- **ChromaDB:** [Chroma Documentation](https://docs.trychroma.com/)
- **Groq:** [Groq Fast Inference](https://groq.com/)
- **Google Gemini:** [Google AI Studio](https://ai.google.dev/)
