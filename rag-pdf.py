import os
import faiss
import openai
import numpy as np
import gradio as gr
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain


#Extracting from PDF for the chat to refer to
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# 
def load_documents(pdf_path):
    return [extract_text_from_pdf(pdf_path)]

# We need gto create an FAISS Vector Store
def create_vector_store(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.create_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

def chatbot(pdf_path, input_text):
    docs = load_documents(pdf_path)
    vector_store = create_vector_store(docs)
    retriever = vector_store.as_retriever()
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.7, "max_length": 512})
    
    from langchain_core.output_parsers import StrOutputParser

    from langchain_core.prompts import ChatPromptTemplate


    system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)

    prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

    qa_chain = prompt_template.partial(context="{context}") | llm | StrOutputParser()
    retrieved_docs = retriever.get_relevant_documents(input_text)
    context = ''.join([doc.page_content for doc in retrieved_docs])
    response = qa_chain.invoke({"input": input_text, "context": context})
    return response

iface = gr.Interface(
    fn=chatbot,
    inputs=[gr.File(label="Upload your PDF"), gr.Textbox(label="Ask me anything about the document")],
    outputs=gr.Textbox(label="AI Response"),
    title="RAG-Powered PDF QA Bot",
    description="Upload a PDF and ask questions about its content."
)

if __name__ == "__main__":
    iface.launch()
