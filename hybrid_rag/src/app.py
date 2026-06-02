import streamlit as st
import os
from google import genai
from google.genai import types
from retriever import HybridGraphRetriever

# Set page layout configuration settings immediately at runtime
st.set_page_config(page_title="NCERT Physics Hybrid RAG Studio", layout="wide")

st.title("⚛️ NCERT Physics Hybrid Graph-RAG Studio")
st.markdown("---")

# Dynamically resolve the project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize our custom hybrid retrieval system pipeline state
@st.cache_resource
def init_retriever():
    # Defensive execution check to verify index paths exist before initializing
    index_path = os.path.join(BASE_DIR, "data", "indexes")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Database directory structure missing at: {index_path}")
    return HybridGraphRetriever(BASE_DIR)

# Render layout frames safely without stalling on errors
try:
    retriever = init_retriever()
    st.sidebar.success("✅ Local Vector Store, BM25 Index, and Knowledge Graph loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Initialization Failure: {str(e)}")
    st.error("The retrieval pipeline could not start. Please make sure you have run your indexer script successfully.")
    st.stop()

# Set up API configuration options securely via the sidebar
st.sidebar.header("Generation Settings")
system_instruction = st.sidebar.text_area(
    "System Instruction", 
    value="You are a helpful assistant for NCERT Physics. Use the provided context to answer questions."
)

# Main Query Interface
query_input = st.text_input("Ask a question about Physics:")

if query_input:
    with st.spinner("Retrieving relevant fragments..."):
        try:
            evidence_chunks, linked_concepts = retriever.retrieve_hyper_context(query_input)
            
            # Format context for the LLM
            context_block = "\n\n".join([f"Fragment {i+1}: {doc['text']}" for i, doc in enumerate(evidence_chunks)])
            
            # Display linked concepts found in the Knowledge Graph
            if linked_concepts:
                st.info(f"Graph-linked concepts identified: {', '.join(linked_concepts)}")

            # Grounded Generation
            client = genai.Client()
            with st.spinner("Synthesizing answer under strict grounding boundaries..."):
                # Execute generation against the standard gemini-2.5-flash model
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"Context Document Fragments:\n{context_block}\n\nUser Query: {query_input}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0
                    )
                )
                
                # Split layout into answer and source views
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Grounded Solution Response")
                    st.write(response.text)
                    
                with col2:
                    st.subheader("Retrieved Reference Fragments")
                    for idx, doc in enumerate(evidence_chunks):
                        with st.expander(f"Reference Fragment {idx+1} - Page {doc['page']}"):
                            st.caption(f"**Source Text:**")
                            st.write(doc['text'])
                                
        except Exception as gen_err:
            st.error(f"Generation Engine tracking failure: {str(gen_err)}")