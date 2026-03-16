from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(base_dir, '..', '..', 'models')
file_dir = os.path.abspath(file_dir) 
def build_vector_store(documents, save_path=file_dir):
    print("[DEBUG] Initializing Embedding Model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("[DEBUG] Generating embeddings and building FAISS index...")
    vector_db = FAISS.from_documents(documents, embeddings)
        
    vector_db.save_local(save_path)
    print(f"[DEBUG] Vector store saved successfully at {save_path}")
    return vector_db