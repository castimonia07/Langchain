# ⚡ LangChain Expression Language (LCEL) & API Serving

This directory is dedicated to **LangChain Expression Language (LCEL)** — the declarative syntax for orchestrating LangChain components into production-grade pipelines — and deploying those chains as production REST APIs using **FastAPI** and **LangServe**.

---

## 📁 Directory Structure

```plaintext
LCEL/
├── serve.py            # FastAPI + LangServe REST API microservice
├── simplellmLCEL.ipynb # LCEL primitives: Prompt | Model | Parser, Groq & Gemini integration
└── README.md           # Documentation for this directory
```

---

## 🧩 What is LCEL?

LangChain Expression Language (LCEL) provides a unified, declarative composition style using Python's bitwise OR operator (`|`). 

### Core Benefits of LCEL:
1. **First-Class Streaming Support:** Immediate time-to-first-token without rewriting logic.
2. **Async & Sync Equivalence:** Every LCEL chain natively supports `.invoke()`, `.ainvoke()`, `.stream()`, `.astream()`, `.batch()`, and `.abatch()`.
3. **Optimized Parallelism:** Independent branches execute concurrently automatically.
4. **Seamless Tracing:** Automatically integrated with LangSmith for full lifecycle observability.
5. **Turnkey Deployment:** Direct deployment to web servers using `langserve`.

---

## 📑 File Walkthroughs

### 1. [`simplellmLCEL.ipynb`](simplellmLCEL.ipynb) — LCEL Core Concepts
- **Multi-Model Orchestration:** Explores switching between cloud providers:
  - Google Gemini (`ChatGoogleGenerativeAI`)
  - Groq Fast Inference (`ChatGroq`)
- **Prompt Templating:** Uses `ChatPromptTemplate.from_messages` with message tuples:
  ```python
  prompt = ChatPromptTemplate.from_messages([
      ("system", "Translate the following into {language}:"),
      ("user", "{text}")
  ])
  ```
- **Output Parsing:** Uses `StrOutputParser()` to extract cleaned output text from raw `AIMessage` objects.
- **Pipeline Assembly:**
  ```python
  chain = prompt | model | StrOutputParser()
  response = chain.invoke({"language": "French", "text": "Hello, how are you?"})
  ```

---

### 2. [`serve.py`](serve.py) — Production LangServe Microservice
Exposes an LCEL chain as an asynchronous REST API server using **FastAPI** and **LangServe**.

#### Microservice Architecture:
```mermaid
flowchart TD
    Client[HTTP Client / Frontend] -->|POST /chain/invoke| FastAPI[FastAPI Server :8000]
    Client -->|GET /chain/playground| Playground[Interactive Web Playground]
    Client -->|GET /docs| Swagger[Swagger / OpenAPI Documentation]
    FastAPI --> LangServe[LangServe Route Handler]
    LangServe --> LCEL[LCEL Chain: Prompt | ChatGroq | StrOutputParser]
    LCEL --> GroqAPI[Groq Cloud LLM: qwen/qwen3.8-27b]
    GroqAPI --> LCEL
    LCEL --> FastAPI
    FastAPI --> Client
```

#### Server Code Highlights:
- **Model:** `ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=groq_api_key)`
- **Chain Definition:**
  ```python
  prompt_template = ChatPromptTemplate.from_messages([
      ("system", "Translate the following into {language}:"),
      ("user", "{text}")
  ])
  chain = prompt_template | model | StrOutputParser()
  ```
- **LangServe Route Mounting:**
  ```python
  add_routes(
      app,
      chain,
      path="/chain"
  )
  ```

---

## 🚀 Running the LangServe Server

### 1. Set Environment Variables
In your root `.env` file, ensure you have:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Launch the Server
Run the service with Uvicorn:
```bash
python LCEL/serve.py
```
Or directly via uvicorn CLI:
```bash
uvicorn LCEL.serve:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access Endpoints
Once running, open your browser:
- **Interactive LangServe Playground:** [http://localhost:8000/chain/playground/](http://localhost:8000/chain/playground/)
- **Swagger OpenAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 4. Consuming the API via cURL / Python Client

#### Via cURL:
```bash
curl -X POST "http://localhost:8000/chain/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": {"language": "German", "text": "Good morning, friend!"}}'
```

#### Via LangServe RemoteRunnable in Python:
```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
result = remote_chain.invoke({"language": "Spanish", "text": "Welcome to LangChain"})
print(result)  # "Bienvenido a LangChain"
```
