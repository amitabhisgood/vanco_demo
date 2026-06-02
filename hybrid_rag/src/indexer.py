import os
import json
import pickle
import chromadb
import networkx as nx
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def build_dual_indexes(json_path, base_dir):
    print("=== INITIALIZING DUAL-INDEX GENERATION ENGINE ===")
    
    # 1. Load preprocessed chunks
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[+] Loaded {len(chunks)} structural fragments.")

    # Create storage directories
    index_dir = os.path.join(base_dir, "data", "indexes")
    os.makedirs(index_dir, exist_ok=True)

    # 2. POPULATE VECTOR DATABASE (Dense Semantic Layer)
    print("[-] Initializing local ChromaDB vector space and generating embeddings...")
    chroma_client = chromadb.PersistentClient(path=os.path.join(index_dir, "chroma_db"))
    
    # Reset existing collection if it exists to keep database clean
    try:
        chroma_client.delete_collection("physics_chunks")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name="physics_chunks")
    
    # Load lightweight, high-performance local embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [c["text"] for c in chunks]
    ids = [str(c["chunk_id"]) for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    # Generate vectors locally
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print("[+] Dense semantic vector store populated successfully.")

    # 3. POPULATE BM25 KEYWORD MATRIX (Sparse Text Layer)
    print("[-] Building BM25 keyword matching architecture...")
    tokenized_corpus = [text.lower().split(" ") for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(os.path.join(index_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print("[+] Sparse keyword search index saved safely.")

    # 4. POPULATE KNOWLEDGE GRAPH (Relational Layer)
    print("[-] Structuring Knowledge Graph networks...")
    G = nx.Graph()
    
    # Define primary rules to dynamically extract formula and chapter hooks for nodes
    for chunk in chunks:
        c_id = f"chunk_{chunk['chunk_id']}"
        page = chunk["metadata"]["page_number"]
        text_content = chunk["text"]
        
        # Add Chunk Node
        G.add_node(c_id, type="chunk", text=text_content, page=page)
        
        # Link to its corresponding Page Node
        page_node = f"page_{page}"
        if not G.has_node(page_node):
            G.add_node(page_node, type="page", number=page)
        G.add_edge(c_id, page_node, relation="belongs_to_page")
        
        # Heuristic concept linker for common physics keywords/formulas
        physics_keywords = ["Coulomb", "Gauss", "Ohm", "Faraday", "Ampere", "Capacitance", "Electric Field"]
        for keyword in physics_keywords:
            if keyword.lower() in text_content.lower():
                concept_node = f"concept_{keyword.lower()}"
                if not G.has_node(concept_node):
                    G.add_node(concept_node, type="concept", name=keyword)
                G.add_edge(c_id, concept_node, relation="discusses")

    # Serialize NetworkX Graph matrix
    with open(os.path.join(index_dir, "knowledge_graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    print(f"[+] Knowledge Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("[+] Phase 2: Indexing pipeline complete!\n")

if __name__ == "__main__":
    BASE_DIR = r"E:\backup2026\vanco_repo\vanco-assessment_usecases\hybrid_rag"
    JSON_INPUT = os.path.join(BASE_DIR, "data", "processed_chunks.json")
    build_dual_indexes(JSON_INPUT, BASE_DIR)