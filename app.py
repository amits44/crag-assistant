import streamlit as st
from graph import app
from langsmith import Client
from dotenv import load_dotenv
load_dotenv()

ls_client = Client()
if "run_id" not in st.session_state:
    st.session_state.run_id = None

st.title("C-RAG Assistant")
query = st.text_input("Upload your documents")

if st.button("Ask") and query:
    with st.spinner("Agent thinking..."):
        result = app.invoke({"question": query, "retry_count": 0})
        config={
        "run_name": "tech-support-query",          
        "tags": ["production", "crag"],            
        "metadata": {"user_query": query},         
    }
        
    runs = list(ls_client.list_runs(
        project_name="crag-tech-support",
        limit=1,
        is_root=True,
    ))
    if runs:
        st.session_state.run_id = str(runs[0].id)    

    st.markdown("### Answer")
    st.write(result["generation"])

if st.session_state.run_id:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Good answer"):
            ls_client.create_feedback(
                st.session_state.run_id,
                key="user-feedback",
                score=1,
                comment="User marked as helpful"
            )
            st.success("Thanks for the feedback!")
    with col2:
        if st.button("👎 Bad answer"):
            ls_client.create_feedback(
                st.session_state.run_id,
                key="user-feedback",
                score=0,
                comment="User marked as unhelpful"
            )
            st.warning("Feedback recorded.")


