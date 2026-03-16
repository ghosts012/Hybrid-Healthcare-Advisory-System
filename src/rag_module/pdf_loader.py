import pdfplumber
import os
import numpy as np

def load_medical_pdf(file_path, pneumonia_pages):
    print(f"[DEBUG] Loading PDF from: {file_path}")
    extracted_data = []
    
    if not os.path.exists(file_path):
        print(f"[DEBUG] ERROR: File not found at {file_path}")
        return []

    with pdfplumber.open(file_path) as pdf:
        for p_num in pneumonia_pages:
            # pdfplumber is 0-indexed
            page = pdf.pages[p_num - 1]
            text = page.extract_text()
            
            # Extract tables and format as Markdown
            tables = page.extract_tables()
            table_str = ""
            for table in tables:
                table_str += "\n" + "\n".join([" | ".join([str(c) if c else "" for c in row]) for row in table])
            
            combined_content = f"PAGE {p_num}\nTEXT:\n{text}\nTABLES:\n{table_str}"
            extracted_data.append(combined_content)
            print(f"[DEBUG] Extracted Page {p_num}: {len(text)} chars, {len(tables)} tables found.")
            
    return extracted_data
