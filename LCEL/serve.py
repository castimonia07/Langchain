from fastapi import FastAPI
import os
from importlib import import_module

ChatPromptTemplate = import_module("langchain_core.prompts").ChatPromptTemplate
StrOutputParser = import_module("langchain_core.output_parsers").StrOutputParser
ChatGroq = import_module("langchain_groq").ChatGroq
add_routes = import_module("langserve").add_routes
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=groq_api_key)

## 1. Create prompt template
system_template="Translate the following into {language}:"

prompt_template=ChatPromptTemplate.from_messages(
    [("system",system_template),
    ("user","{text}")]
)

parser=StrOutputParser()

## create chain
chain=prompt_template|model|parser

## App definition
app= FastAPI(title="Langchain Server",
            version="1.0",
            description="A simple API server using Langchain runnable interfaces"
            )



## Adding chain routes

add_routes(
    app,
    chain,
    path="/chain"
)


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

