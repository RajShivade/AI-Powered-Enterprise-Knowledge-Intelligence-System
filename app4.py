# AI Customer Support Knowledge Assistant
import streamlit as st

import os 

from langchain_google_genai import ChatGoogleGenerativeAI 

#RAG Pipeline
from langchain_community.document_loaders import WebBaseLoader #Document_Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter #Text_splitter
from langchain_ollama import OllamaEmbeddings #Embeddings


from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
import time


from dotenv import load_dotenv
load_dotenv()

# Load the Google Gemini API key 
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini")

urls = [
    "https://docs.stripe.com/payments",
    "https://docs.stripe.com/refunds",
    "https://www.notion.so/help/create-a-page"
]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.loader = WebBaseLoader(urls)
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
    
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = Chroma.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    
    
st.title("AI Customer Support Knowledge Assistant")

llm_brain = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

support_prompt = ChatPromptTemplate.from_template(
"""
You are a professional Customer Support AI assistant.

Use the provided documentation to answer the customer's question.

Context:
{context}

Customer Question:
{input}

Instructions:
- Provide a helpful and clear answer.
- Use step-by-step instructions when explaining processes.
- Be polite and professional.
- If the answer cannot be found in the documentation, say:
  "I'm sorry, but I could not find this information in the available documentation."
- When possible, reference the source of the information.

Answer:
"""
)

document_chain = create_stuff_documents_chain(llm_brain,support_prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)

prompt = st.text_input("Input Your Prompt here")

if prompt: 
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print("Response_Time", time.process_time()-start)
    st.write(response['answer'])
    
    #With a streamlit expander 
    with st.expander("Document Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------------")