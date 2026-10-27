# 🚀 Getting Started with LangChain & Foundational Apps

This directory contains introductory notebooks demonstrating foundational concepts in **LangChain (v0.2+)**, from model initialization and chat templates to a complete end-to-end **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 📁 Directory Structure

```plaintext
1.1-openai/
├── GettingStarted.ipynb    # Model initialization, Prompt Templates & Output Parsers
├── Simpleapp.ipynb         # Full Web Scraping -> Chunking -> Vector Store -> RAG Pipeline
└── README.md               # Documentation for this directory
```

---

## 📑 Notebook Overviews

### 1. [`GettingStarted.ipynb`](GettingStarted.ipynb)
**Goal:** Master the core building blocks of LangChain with Google Gemini and Chat Models.

- **Key Concepts Covered:**
  - **Model Instantiation:** Initializing `ChatGoogleGenerativeAI` (`model="gemini-1.5-flash"` or `gemini-pro`).
  - **Invoking Models:** Synchronous invocation via `llm.invoke("What is GenAI?")`.
  - **Chat Prompt Templates:** Designing reusable prompts using `ChatPromptTemplate.from_messages` with `("system", ...)` and `("user", ...)` roles.
  - **Output Parsers:** Standardizing LLM outputs using `StrOutputParser()` to extract plain strings from `AIMessage` objects.
  - **LCEL Chaining:** Assembling chains using the pipe operator:
    ```python
    chain = prompt | llm | output_parser
    response = chain.invoke({"question": "What is GenAI?"})
    ```

### 2. [`Simpleapp.ipynb`](Simpleapp.ipynb)
**Goal:** Build a complete end-to-end Question-Answering application over live web documentation.

- **Workflow Architecture:**
  ```mermaid
  flowchart LR
      Web[Web URL: LangSmith Docs] --> Loader[WebBaseLoader]
      Loader --> Splitter[RecursiveCharacterTextSplitter]
      Splitter --> Embeddings[HuggingFaceEmbeddings all-MiniLM-L6-v2]
      Embeddings --> VectorStore[(FAISS Vector Index)]
      VectorStore --> Retriever[Retriever Engine]
      Retriever --> Chain[create_retrieval_chain]
      Chain --> LLM[ChatGoogleGenerativeAI]
      LLM --> Output[Structured Answer]
  ```

- **Detailed Step-by-Step Implementation:**
  1. **Data Ingestion:** Loads web documentation dynamically from `https://docs.smith.langchain.com/` using `WebBaseLoader`.
  2. **Text Chunking:** Splits long web pages into manageable segments using `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
  3. **Vector Embeddings:** Uses `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` to convert text chunks into dense 384-dimensional vectors.
  4. **Vector Storage:** Stores and indexes vectors in an in-memory **FAISS** vector store via `FAISS.from_documents(docs, embeddings)`.
  5. **Retrieval Chain:** Combines:
     - `create_stuff_documents_chain(llm, prompt)` for context stuffing into prompts.
     - `create_retrieval_chain(retriever, question_answer_chain)` to fetch relevant chunks and generate accurate grounded answers.

---

## 🛠️ Environment & Prerequisites

Ensure the following API keys are configured in your root `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING=true
LANGCHAIN_PROJECT=Langchain-Tutorials
```

### Required Packages
```bash
pip install langchain langchain-community langchain-google-genai langchain-huggingface sentence-transformers faiss-cpu beautifulsoup4 python-dotenv
```

---

## 💡 Quick Code Example: The Core LCEL Pattern

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

# 2. Define Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI assistant that explains concepts simply."),
    ("user", "Explain: {topic}")
])

# 3. Create LCEL Chain
chain = prompt | llm | StrOutputParser()

# 4. Invoke
print(chain.invoke({"topic": "How Vector Databases Work"}))
```
