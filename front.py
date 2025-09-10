

import streamlit as st
import os
import time
import pickle
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# -------------------- BACKEND SETUP --------------------

# Set API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
genai.configure(api_key="AIzaSyB4R8OwkEZ5QAL5J28olzLXL5D1xui1F9A")   # Replace with your Gemini API key

# Initialize model
model = genai.GenerativeModel("gemini-2.5-flash")

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create retriever
# -------------------- BACKEND SETUP --------------------

# ... (your other setup code like genai.configure and model init)

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS vector store
DB_FAISS_PATH = "index_faiss"

try:
    if os.path.exists(DB_FAISS_PATH):
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True  # Required for FAISS with pickle
        )
    else:
        st.error("Vector store not found. Please run ingest.py first.")
        vectorstore = None
except Exception as e:
    st.error(f"Error loading vector store: {e}")
    vectorstore = None

# ... (the rest of your code, like the generate_answer function)

def generate_answer(query):
    try:
        # First, check if the vectorstore exists
        if vectorstore:
            # Use a search method that returns documents AND their relevance scores
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=3)

            # Set a threshold for relevance. Scores are between 0 (dissimilar) and 1 (very similar).
            # You can adjust this value based on testing.
            RELEVANCE_THRESHOLD = 0.72 
            
            # Filter documents that are above the relevance threshold
            relevant_docs = [doc for doc, score in docs_with_scores if score > RELEVANCE_THRESHOLD]

            # If there are relevant documents, use the RAG context
            if relevant_docs:
                context = "\n".join([doc.page_content for doc in relevant_docs])
                prompt = f"""
                You are a helpful assistant providing information about Ruhani Gera.
                Use the following context to answer the question.

                Context:
                {context}

                Question:
                {query}
                """
            # Otherwise, use a general prompt and let the LLM answer from its own knowledge
            else:
                prompt = f"""
                You are a helpful and knowledgeable AI assistant. Answer the following question.

                Question:
                {query}
                """
        # Fallback if the vectorstore wasn't loaded
        else:
             prompt = f"""
                You are a helpful and knowledgeable AI assistant. Answer the following question.

                Question:
                {query}
                """

        # Generate the response using the dynamically created prompt
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text.strip()

    except Exception as e:
        return f"❌ Error: {e}"


# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Dark Theme CSS
st.markdown("""
    <style>
    .stApp {
        background: #0f1117;
        color: #e4e6eb;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #1c1f2e;
        color: white;
        padding: 20px;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #bb86fc;
    }
    .sidebar-sub {
        font-size: 14px;
        color: #bbb;
    }
    .dev-info {
        margin-top: 20px;
        font-size: 15px;
        color: #ffcc00;
    }
    .history-item {
        background: #2b2f44;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 14px;
        color: white;
        cursor: pointer;
        border: 1px solid #444;
    }
    .history-item:hover {
        background: #3c4260;
    }
    .user-msg {
        background-color: #6a1b9a; /* purple bubble */
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0px;
        font-size: 16px;
        max-width: 75%;
        float: right;
        clear: both;
        color: #fff;
    }
    .bot-msg {
        background-color: #1f2233;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0px;
        font-size: 16px;
        max-width: 75%;
        float: left;
        clear: both;
        color: #e4e6eb;
        border: 1px solid #333;
    }
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 300px; /* same as sidebar width */
        right: 0;
        background: #0f1117;
        padding: 12px;
        border-top: 1px solid #333;
    }
    .stChatInput input {
        background: #2b2f44 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px !important;
        border: 1px solid #555 !important;
    }
    .typing-indicator {
        font-size: 14px;
        color: #bbb;
        font-style: italic;
        margin: 6px 0;
    }
    .stButton > button {
        background: #9c27b0; /* Darker purple */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 18px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: #7b1fa2;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🤖 RAG Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Built to assist with general questions.</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='dev-info'>👩‍💻 Developer: Ruhani Gera</div>", unsafe_allow_html=True)
    st.markdown("<div class='dev-info'>💼 Role: Chatbot Developer</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Contact Button
    linkedin_url = "https://www.linkedin.com/in/ruhani-gera-851454300/"
    st.markdown(f"<a href='{linkedin_url}' target='_blank'><button style='background:#bb86fc;color:white;padding:10px 15px;border:none;border-radius:8px;cursor:pointer;'>📩 Contact with me</button></a>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<b style='color:#bb86fc;'>✨ Features:</b><br>- RAG-powered responses<br>- Context-aware conversations<br>- Simple, modern UI", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<b style='color:#bb86fc;'>🕘 History:</b>", unsafe_allow_html=True)

   
    # Show History Button
    if st.button("📜 Show History", use_container_width=True):
        st.session_state.show_history = not st.session_state.get("show_history", False)

    # Display history messages if toggled
    if st.session_state.get("show_history", False):
        if "history" in st.session_state and st.session_state.history:
            for speaker, msg in st.session_state.history:
                if speaker == "You":
                    st.markdown(f"<div class='history-item'>🙋 {msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='history-item'>🤖 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:gray;'>No history yet...</p>", unsafe_allow_html=True)
        

    # Clear History button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.chat_start_time = time.strftime("%H:%M:%S")
        st.rerun()


# -------------------- MAIN --------------------
# Animated header
st.markdown("<h1 style='text-align:center; color:#bb86fc; animation: fadeIn 2s;'>💬 Welcome to RAG Chatbot</h1>", unsafe_allow_html=True)

# Initialize history + start time
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.chat_start_time = time.strftime("%H:%M:%S")

st.markdown(f"<p style='text-align:center; color:gray;'>🕒 Session started at {st.session_state.chat_start_time}</p>", unsafe_allow_html=True)

# Display chat messages
chat_container = st.container()
with chat_container:
    for speaker, text in st.session_state.history:
        if speaker == "You":
            st.markdown(f"<div class='user-msg'>🙋 {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{text}</div>", unsafe_allow_html=True)

# -------------------- CHAT INPUT --------------------
st.markdown("<div class='chat-input-container'>", unsafe_allow_html=True)

# Placeholder inside box
placeholder_text = "💡 Ask me anything..."
user_input = st.chat_input(placeholder_text)

# If clicked history message exists
if "pending_input" in st.session_state and st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = ""

# If user submits
if user_input:
    st.session_state.history.append(("You", user_input))

    # Typing indicator
    with chat_container:
        st.markdown("<div class='typing-indicator'>🤖 Bot is typing...</div>", unsafe_allow_html=True)
        st.empty()

    answer = generate_answer(user_input)
    st.session_state.history.append(("Bot", answer))
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)




