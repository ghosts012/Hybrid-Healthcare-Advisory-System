from pdf_loader import load_medical_pdf
from chunker import create_chunks
from embedder import build_vector_store
import os

# CONFIGURATION
base_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(base_dir, '..', '..', 'data', 'medical_reference_who.pdf')
file_dir = os.path.abspath(file_dir) 
PDF_PATH = file_dir

PNEUMONIA_PAGES = [104, 105, 106, 107, 108, 110, 111, 112, 113, 114] # Example page range

def run_pipeline():
    # 1. Load
    raw_data = load_medical_pdf(PDF_PATH, PNEUMONIA_PAGES)
    
    # 2. Chunk
    if raw_data:
        docs = create_chunks(raw_data)
        
        # 3. Embed & Store
        build_vector_store(docs)
        print("[DEBUG] Pipeline execution finished.")
    else:
        print("[DEBUG] Pipeline failed: No data extracted.")

if __name__ == "__main__":
    run_pipeline()