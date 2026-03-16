from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

class PneumoniaRetriever:
    def __init__(self, index_path="models"):
        print(f"[DEBUG] Initializing Retriever from: {index_path}")
        
        # 1. Load the same embedding model used for building
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Load the FAISS index
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            self.vector_db = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True # Required for loading local .pkl files
            )
            print("[DEBUG] Vector Database loaded successfully.")
        else:
            print(f"[DEBUG] ERROR: index.faiss not found in {index_path}")

    def get_advisory(self, severity_score, status):
        """
        Takes the results from your Vision model and queries the RAG.
        """
        # Create a clinical query based on the model results
        query = f"Clinical management and treatment protocol for {status} pneumonia with a severity of {severity_score:.1f}%"
        
        print(f"[DEBUG] Querying RAG: '{query}'")
        
        # Retrieve the top 2 most relevant chunks
        docs = self.vector_db.similarity_search(query, k=2)
        
        # Combine the chunks into a single reference text
        context = "\n\n".join([doc.page_content for doc in docs])
        
        print(f"[DEBUG] Retrieved {len(docs)} relevant context chunks.")
        return context

# For testing independently
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(base_dir, '..', 'models')
    file_dir = os.path.abspath(file_dir) 
    retriever = PneumoniaRetriever(index_path=file_dir)
    # Test with a mock severity score
    sample_context = retriever.get_advisory(88.5, "High Risk")
    print("\n--- RETRIEVED ADVISORY CONTEXT ---")
    print(sample_context)