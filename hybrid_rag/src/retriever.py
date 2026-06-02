import os

# Force absolute offline execution modes immediately at the OS level
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pickle
import chromadb
import networkx as nx
from sentence_transformers import SentenceTransformer

class HybridGraphRetriever:
    def __init__(self, base_dir):
        self.index_dir = os.path.join(base_dir, "data", "indexes")
        
        # 1. Connect to Local ChromaDB (Dense Semantic)
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.index_dir, "chroma_db"))
        self.collection = self.chroma_client.get_collection("physics_chunks")
        
        # Forces the model to load strictly from local cache to prevent network timeouts/pings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        
        # 2. Load Local BM25 Matrix (Sparse Keyword)
        with open(os.path.join(self.index_dir, "bm25_index.pkl"), "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.chunks = bm25_data["chunks"]
            
        # 3. Load Local Knowledge Graph
        with open(os.path.join(self.index_dir, "knowledge_graph.pkl"), "rb") as f:
            self.G = pickle.load(f)

    def dense_search(self, query, top_k=10):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["ids"][0]

    def sparse_search(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [str(i) for i in top_indices]

    def rrf_fusion(self, dense_ids, sparse_ids, top_n=4, k=60):
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_n]

    def extract_graph_context(self, fused_ids):
        graph_metadata_hints = set()
        for c_id in fused_ids:
            if self.G.has_node(c_id):
                for n in self.G.neighbors(c_id):
                    node_attrs = self.G.nodes[n]
                    if node_attrs.get("type") == "concept":
                        graph_metadata_hints.add(node_attrs.get("name"))
                        
        return list(graph_metadata_hints)

    def retrieve_hyper_context(self, query):
        dense_res = self.dense_search(query, top_k=10)
        sparse_res = self.sparse_search(query, top_k=10)
        
        fused_ids = self.rrf_fusion(dense_res, sparse_res, top_n=4)
        linked_concepts = self.extract_graph_context(fused_ids)
        
        evidence_payload = []
        for c_id in fused_ids:
            target_chunk = self.chunks[int(c_id)]
            evidence_payload.append({
                "id": c_id,
                "text": target_chunk["text"],
                "page": target_chunk["metadata"]["page_number"]
            })
            
        return evidence_payload, linked_concepts

if __name__ == "__main__":
    # Dynamically resolve project directory relative to this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    retriever = HybridGraphRetriever(BASE_DIR)
    
    test_query = "What is Coulomb law definition?"
    context, concepts = retriever.retrieve_hyper_context(test_query)
    
    print(f"Query: {test_query}")
    print(f"Concepts: {concepts}")
    print(f"Retrieved {len(context)} chunks.")