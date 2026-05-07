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
from langchain_chroma import Chroma
import time
from dotenv import load_dotenv

# Page Configuration
st.set_page_config(
    page_title="HelpHub | AI Support Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Professional UI Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
    }

    /* Header Section */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: white;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        border-radius: 0 0 2rem 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .main-title {
        color: #4f46e5;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.025em;
    }
    
    .sub-title {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Chat Bubbles */
    .chat-bubble {
        padding: 1.25rem;
        border-radius: 1.25rem;
        margin-bottom: 1rem;
        max-width: 85%;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    
    .user-bubble {
        background-color: #4f46e5;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0.25rem;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.2);
    }
    
    .ai-bubble {
        background-color: white;
        color: #1e293b;
        margin-right: auto;
        border-bottom-left-radius: 0.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .verified-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background-color: #eef2ff;
        color: #4338ca;
        padding: 0.25rem 0.625rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 1rem !important;
        padding: 0.75rem 1.25rem !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Environment
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini")

# RAG Configuration
URLS = [
    "https://docs.stripe.com/payments",
    "https://docs.stripe.com/refunds",
    "https://www.notion.so/help/create-a-page"
]

@st.cache_resource
def initialize_rag():
    loader = WebBaseLoader(URLS)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma.from_documents(splits, embeddings)

# App Header
st.markdown("""
    <div class="header-container">
        <div class="main-title">HelpHub</div>
        <div class="sub-title">Intelligent Support Assistant</div>
    </div>
    """, unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm HelpHub. How can I assist you with your documentation queries today?"}
    ]

if "vectors" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        st.session_state.vectors = initialize_rag()

# LLM Setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
prompt_template = ChatPromptTemplate.from_template("""
You are a professional Customer Support AI. Use the context to answer the question.
Context: {context}
Question: {input}
Answer:""")

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Display Chat History
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "ai-bubble"
    badge = '<div class="verified-badge">✓ HelpHub Verified</div>' if msg["role"] == "assistant" else ""
    st.markdown(f'<div class="chat-bubble {role_class}">{badge}{msg["content"]}</div>', unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about Stripe or Notion..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)

    with st.spinner("Analyzing documentation..."):
        start = time.time()
        response = retrieval_chain.invoke({"input": prompt})
        latency = time.time() - start
        
        answer = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        st.markdown(f"""
            <div class="chat-bubble ai-bubble">
                <div class="verified-badge">✓ HelpHub Verified</div>
                {answer}
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.5rem;">
                    Response time: {latency:.2f}s
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.sidebar:
            st.subheader("📖 Sources")
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Source {i+1}**")
                st.caption(doc.page_content[:200] + "...")
                st.divider()

st.markdown("<br><p style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>© 2026 HelpHub AI Support</p>", unsafe_allow_html=True)