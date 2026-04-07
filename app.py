import streamlit as st
from graph import app
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

st.title("Self-Correcting Tech Support")
query = st.text_input("Describe your issue...")

if st.button("Ask") and query:
    with st.spinner("Agent thinking..."):
        result = app.invoke({"question": query, "retry_count": 0})
        config={
        "run_name": "tech-support-query",          
        "tags": ["production", "crag"],            
        "metadata": {"user_query": query},         
    }
    st.markdown("### Answer")
    st.write(result["generation"])
    with st.expander("Sources used"):
        for doc in result["documents"]:
            st.write(doc.page_content[:300] + "...")