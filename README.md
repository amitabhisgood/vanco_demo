# 🚀 Technical Assessment Demonstration Prototypes for Vanco (vanco_demo)

This repository contains the complete suite of functional demonstration prototypes engineered for the **Technical Assessment by Vanco**. 

The workspace aggregates end-to-end implementation pipelines, data architecture designs, validation engines, and user evaluation dashboards across three isolated, standalone project workspaces. The implementations emphasize high-discipline software engineering, scalable multi-modal data processing, robust evaluation tracking, and strict alignment with enterprise production guardrails.

---

## 🏗️ Consolidated Repository Directory Structure

The repository splits each technical demonstration into its own independent root folder, housing standalone source code environments, specialized model weights, and localized data directories:

```
vanco_demo/
│
├── asl_detection/                          # Use Case: American Sign Language Detection
│   ├── data/
│   │   ├── annotations/                    # COCO/YOLO structured frame markings
│   │   ├── train/                          # Dynamic sign-language video frames
│   │   └── val/
│   ├── models/
│   │   └── asl_yolov8_best.pt              # Trained real-time vision weights
│   └── src/
│       ├── dataset_preprocessor.py         # Frame extraction and box normalization
│       ├── train_detector.py               # Hyperparameter tuned YOLO execution script
│       └── app_vision.py                   # Real-time Streamlit webcam dashboard
│
├── grocery_forecasting/                    # Use Case: Grocery Sales Forecasting
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── external_events.csv
│   │   └── engineered_features.parquet
│   ├── models/
│   │   └── forecasting_lgb_ensemble.pkl    # Serialized multi-horizon LightGBM regressors
│   └── src/
│       ├── feature_engineering.py          # Lags, rolling windows, calendar embeddings
│       ├── train_eval.py                   # Rolling time-series validation engine
│       └── inference.py                    # Scalable batch inference generation loop
│
└── hybrid_rag/                             # Use Case: Hybrid RAG Textbook Application
    ├── data/
    │   ├── NCERT-Class-12-Physics-Part-1.pdf # Immutable Core Source Document
    │   ├── processed_chunks.json           # Layout-parsed text chunk fragments
    │   └── indexes/
    │       ├── chroma_db/                  # Persistent ChromaDB vector store
    │       ├── bm25_index.pkl              # Serialized BM25Okapi matrix
    │       └── knowledge_graph.pkl         # Serialized NetworkX Relational Graph
    └── src/
        ├── ingestion.py                    # PDF Parser & Recursive Fragmenter
        ├── indexer.py                      # Multi-Layer Index Construction Pipeline
        ├── retriever.py                    # HybridGraphRetriever Engine (RRF Fusion)
        └── app_rag.py                      # Streamlit UI Dashboard & LLM Grounder

---

```

## 📊 1. Grocery Sales Forecasting with External Events

### Architecture Blueprint
Implements a performance-optimized demand forecasting engine engineered to process multi-item, highly volatile retail records across distinct geographical locations. The pipeline couples automated calendar feature construction with optimized gradient-boosted tree architectures.

### Key Architectural Strategies
* **Validation Discipline:** Rejects naive random K-Fold splitting to eliminate temporal leakage. Implements a programmatic **Rolling Window Time-Series Cross-Validation Engine** matching operational forecasting horizons.
* **Feature Engineering Vectors:** Dynamically maps external event calendars, promotional profiles, rolling sales aggregations (t-7, t-14, t-30), and localized socio-economic indicators.

### Execution Sequence

```
(For Windows)
cd grocery_forecasting

```

# 1. Execute robust feature engineering over raw transactions

```

python src/feature_engineering.py


```

# 2. Execute cross-validated model training and serialize ensemble models

```

python src/train_eval.py


```
# 3. Output batch predictions

```

python src/inference.py

---

```

## 📷 2. Real-Time American Sign Language (ASL) Detection

### Architecture Blueprint
A localized computer vision interface running high-frequency real-time bounding-box inference over low-latency edge camera arrays to recognize and translate continuous American Sign Language tokens.

### Key Architectural Strategies
* **Data Synthesis Framework:** Structural augmentations (spatial transforms, luminance shifting, mosaic variations) are applied systematically to protect the model against real-world lighting variations and dynamic environments.
* **Inference Loop Optimization:** Employs an ultra-lean tracking architecture backed by optimized tensor inference pipelines to achieve sustainable, high-frame-rate tracking bounds on standard host CPUs.

### Execution Sequence

```

cd asl_detection

```

# 1. Transform raw images and compile COCO/YOLO compliance matrices

```
python src/dataset_preprocessor.py


```

# 2. Initiate bounding-box detection optimization training

```


python src/train_detector.py


```


# 3. Launch the live camera tracking dashboard interface

```

streamlit run src/app_vision.py

```


---

## ⚛️ 3. Hybrid Graph-RAG Application for Physics PDF

### Architecture Blueprint
A multi-layered hybrid information retrieval matrix integrating dense semantic vectors, sparse statistical token patterns, and a dynamic local knowledge network to power uncompromised, verifiable question answering over complex textbooks.

```
                              ┌──> Dense Embedding Model ──> [ChromaDB Vector Store]
                              │
   [Source PDF Chunks] ───────┼──> Text Tokenization ───────> [BM25 Keyword Index]
                              │
                              └──> Relational Mapping ──────> [NetworkX Knowledge Graph]

```

### Key Architectural Strategies
* **Multi-Modal Retrieval Paths:** Queries execute across parallel dense semantic collections (all-MiniLM-L6-v2), token keyword frequencies (BM25Okapi), and graph topological paths (NetworkX) concurrently.
* **Reciprocal Rank Fusion (RRF):** Resolves scaling variances across vector cosine similarities and raw keyword counts via mathematical rank alignment.
* **Absolute Prompt Shielding:** Forwards context to gemini-2.5-flash at temperature=0.0. Enforces structural page citations (e.g., [Page 42]) and strict null state fallbacks: "I cannot answer this query based on the provided source documentation context boundaries."

### Execution Sequence

cd hybrid_rag

# 1. Extract raw layout arrays and compile structural text fragments

```

python src/ingestion.py

```

# 2. Build ChromaDB instances, BM25 matrices, and local knowledge networks

```

python src/indexer.py

```


# 3. Launch the grounded RAG analytical user interface

```

streamlit run src/app_rag.py


```



---

## ⚖️ System Architecture Trade-off Matrix

```

| Component Layer | Production Architecture Choice | Technical Justification | Managed Operational Trade-off |
| :--- | :--- | :--- | :--- |
| **Forecasting Engine** | LightGBM Ensemble | Lightning-fast gradient tree processing natively handling zero-inflated data profiles. | Requires strict, manual tabular schema processing relative to deep-learning architectures. |
| **Vision Model** | YOLOv8 (Edge Footprint) | Sub-millisecond localized inference overhead supporting high web-camera sample tracking streams. | Bounding box spatial accuracy limits grow complex if handling multiple macro scales simultaneously. |
| **Graph Relational Mapping** | Embedded NetworkX Matrix | Light-weight, hyper-efficient in-memory tracking footprint; avoids complex external infrastructure maintenance. | Designed for singular, high-density documents; scaling to absolute web-scale databases benefits from Neo4j/ArangoDB. |

```
---

## 🛑 Technical Limitations & System Upgrade Roadmap

1. **OCR Formula Processing Loss:** Linear document streams risk misaligning subscripts, fractions, and notation groupings. *Production Upgrade:* Integrate multimodal layout parser API models (e.g., Mathpix) to ingest mathematical variables systematically.
2. **Missing Dynamic Graph Upgrades:** Knowledge networks operate on heuristic vocabulary mapping rules. *Production Upgrade:* Incorporate an interactive Named Entity Recognition (NER) inference pipeline to continuously construct concepts.
3. **Forecasting Cold-Starts:** Accuracy tolerances drop when introducing new inventory units with zero behavioral history. *Production Upgrade:* Inject hierarchical category embedding profiles to transfer behavioral traits across related inventory networks.

---

## 🚀 Complete Suite Installation & Global Setup

### 1. Prerequisites Verification
Ensure you are executing code in a python environment bounded within **Python 3.10+**.

### 2. Environment Setup
Clone this repository and configure the global execution suite tools via pip:

git clone https://github.com/amitabhisgood/vanco_demo.git
cd vanco_demo

# Install all operational frameworks, analytical architectures, and modeling drivers
pip install streamlit pdfplumber langchain-text-splitters chromadb sentence-transformers rank-bm25 networkx google-genai ultralytics opencv-python-headless lightgbm scikit-learn pandas numpy pyarrow

### 3. Verification of Data Assets
Before running execution or training tasks inside the individual folders, verify that raw input assets (such as the textbook PDF file and transactional csv arrays) are correctly placed inside their respective subdirectories within the localized data/ structures.
