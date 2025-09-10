import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables (e.g., GEMINI_API_KEY if needed later)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Load your .txt file
loader = TextLoader("mydata.txt", encoding="utf-8")  # use utf-8 to avoid encoding errors
documents = loader.load()

if not documents:
    raise ValueError("❌ mydata.txt is empty or not loaded correctly.")

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# ✅ Use HuggingFace embeddings with PyTorch backend (no TensorFlow)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vector store
db = FAISS.from_documents(docs, embeddings)

# Save FAISS index locally
db.save_local("index_faiss")

print("✅ Data ingestion complete. FAISS index saved to 'index_faiss'")
