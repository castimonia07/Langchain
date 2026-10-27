import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama  # type: ignore[reportMissingImports]
import streamlit as st  # type: ignore[reportMissingImports]
from langchain_core.prompts import ChatPromptTemplate  # type: ignore[reportMissingImports]
from langchain_core.output_parsers import StrOutputParser  # type: ignore[reportMissingImports]

load_dotenv()

# os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")



## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title("Langchain Demo With Gemma Model")
input_text=st.text_input("What question you have in mind?")


## Ollama Llama2 model
llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))



