import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os

chat = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstore = Chroma(
    persist_directory="./intro.ai",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 1,
        "lambda_mult": 0.7
    }
)

PROMPT_S = """You will receive a question from a student taking the Intro to AI course.
Answer the question using only the provided context."""

PROMPT_TEMPLATE_H = """This is the question:
{question}

This is the context:
{context}"""

prompt_s = SystemMessage(PROMPT_S)

prompt_template_h = HumanMessagePromptTemplate.from_template(
    PROMPT_TEMPLATE_H
)

chat_prompt_template = ChatPromptTemplate(
    [prompt_s, prompt_template_h]
)

str_output_parser = StrOutputParser()

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | chat_prompt_template
    | chat
    | str_output_parser
)

st.header("365 Q&A Chatbot", divider = True)

question = st.text_input("Type your question:")

if st.button("Ask"):
    if question:
        response = chain.invoke(question)
        st.write(response)
    else:
        st.warning("Please type your question.", icon="⚠️")