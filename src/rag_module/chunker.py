# import sys
# sys.path.append(r"C:\Users\HP\Desktop\dev\Hybrid-Healthcare-Advisory-System\.venv\Lib\site-packages")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def create_chunks(raw_pages):
    print("[DEBUG] Initializing Chunker...")
    # Using overlap to ensure medical terms aren't cut in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    
    docs = []
    for content in raw_pages:
        # Create LangChain Document objects
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            docs.append(Document(page_content=chunk))
    
    print(f"[DEBUG] Chunking complete. Created {len(docs)} chunks.")
    return docs