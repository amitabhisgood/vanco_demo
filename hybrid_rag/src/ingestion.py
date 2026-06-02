import os
import json
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_physics_pdf(pdf_path, output_json_path):
    print("=== STARTING DOCUMENT INGESTION PIPELINE ===")
    
    if not os.path.exists(pdf_path):
        print(f"[!] Error: Could not find the Physics PDF file at: {pdf_path}")
        return False

    raw_documents = []
    
    print(f"[-] Parsing PDF layout and extracting page elements...")
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1  # Standardize to 1-based index page numbering
            text = page.extract_text(layout=False)
            
            if text and text.strip():
                raw_documents.append({
                    "text": text,
                    "metadata": {
                        "page_number": page_num,
                        "source": os.path.basename(pdf_path)
                    }
                })
                
    print(f"[+] Successfully extracted raw text from {len(raw_documents)} pages.")

    # Section & Page-aware chunking configuration
    # Text splitter uses a 1000 character block size with a 200 character overlap sliding window
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    processed_chunks = []
    chunk_id_counter = 0
    
    print("[-] Fragmenting documents into structural semantic chunks...")
    for doc in raw_documents:
        chunks = text_splitter.split_text(doc["text"])
        
        for chunk_text in chunks:
            # Clean trailing artifacts while keeping formulas intact
            cleaned_text = " ".join(chunk_text.split())
            
            processed_chunks.append({
                "chunk_id": chunk_id_counter,
                "text": cleaned_text,
                "metadata": {
                    "page_number": doc["metadata"]["page_number"],
                    "source": doc["metadata"]["source"]
                }
            })
            chunk_id_counter += 1

    # Save data locally to avoid recalculating pipelines
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding=\"utf-8\") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=4)
        
    print(f"[+] Ingestion complete! Generated {len(processed_chunks)} chunks.")
    print(f"[+] Structured workspace payload saved to: {output_json_path}\n")
    return True

if __name__ == "__main__":
    # Dynamically resolve project directory relative to this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Path configuration
    PDF_INPUT = os.path.join(BASE_DIR, "data", "NCERT-Class-12-Physics-Part-1.pdf")
    OUTPUT_JSON = os.path.join(BASE_DIR, "data", "processed_chunks.json")
    
    ingest_physics_pdf(PDF_INPUT, OUTPUT_JSON)