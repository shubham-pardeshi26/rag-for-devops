from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load an embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Read DevOps documents
docs = []
for file in os.listdir("docs"):
    with open(f"docs/{file}", "r", encoding="utf-8") as f:
        docs.append(f.read())

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_text(" ".join(docs))

# Store in FAISS
vector_store = FAISS.from_texts(docs, embeddings)
vector_store.save_local("devops_faiss_index")

print("✅ Documents indexed successfully!")
