import streamlit as st
from graph import app
from langsmith import Client
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

ls_client = Client()
if "run_id" not in st.session_state:
    st.session_state.run_id = None

st.title(" C-RAG Research Assistant")
st.markdown("*Corrective RAG with web fallback for accurate research answers*")

# File upload section
st.sidebar.header(" Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload research documents",
    type=['pdf', 'txt', 'md'],
    accept_multiple_files=True,
    help="Upload PDFs, text files, or markdown documents"
)

# Save uploaded files to docs directory
if uploaded_files:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = docs_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f" {len(uploaded_files)} file(s) uploaded to docs/")
    st.sidebar.info(" Run ingestion to index new documents (see below)")

# Ingestion button
if st.sidebar.button(" Re-index Documents"):
    with st.spinner("Indexing documents..."):
        from ingest import load_documents, split_documents, build_vectorstore, DOCS_DIR, CHROMA_DIR
        try:
            docs = load_documents(DOCS_DIR)
            chunks = split_documents(docs)
            build_vectorstore(chunks, CHROMA_DIR)
            st.sidebar.success(" Documents indexed successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f" Error: {str(e)}")

# Main query interface
st.header("Ask a Research Question")
query = st.text_area(
    "Your question:",
    placeholder="e.g., What are the main findings about...?",
    height=100
)

if st.button(" Search", type="primary") and query:
    with st.spinner(" Researching your question..."):
        try:
            result = app.invoke({"question": query, "retry_count": 0})
            config = {
                "run_name": "research-query",          
                "tags": ["production", "crag", "research"],            
                "metadata": {"user_query": query},         
            }
            
            runs = list(ls_client.list_runs(
                project_name="crag-tech-support",  # Consider renaming to "crag-research-assistant"
                limit=1,
                is_root=True,
            ))
            if runs:
                st.session_state.run_id = str(runs[0].id)    

            # Display results
            st.markdown("---")
            st.markdown("###  Answer")
            st.write(result["generation"])
            
            # Show sources used
            if "documents" in result and result["documents"]:
                with st.expander(" View Sources"):
                    for i, doc in enumerate(result["documents"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.markdown("---")
            
            # Show if web search was used
            if result.get("web_fallback"):
                st.info(" Web search was used to supplement document knowledge")
                
        except FileNotFoundError as e:
            st.error(" No vector database found. Please upload documents and click 'Re-index Documents'")
        except Exception as e:
            st.error(f" Error: {str(e)}")

# Feedback section
if st.session_state.run_id:
    st.markdown("---")
    st.markdown("##### Was this answer helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes, helpful"):
            ls_client.create_feedback(
                st.session_state.run_id,
                key="user-feedback",
                score=1,
                comment="User marked as helpful"
            )
            st.success("Thanks for the feedback!")
    with col2:
        if st.button("👎 Not helpful"):
            ls_client.create_feedback(
                st.session_state.run_id,
                key="user-feedback",
                score=0,
                comment="User marked as unhelpful"
            )
            st.warning("Feedback recorded. We'll improve!")