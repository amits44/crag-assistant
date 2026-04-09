🤖 C-RAG Assistant (Self-Correcting Tech Support)
An advanced, self-correcting AI agent built using LangGraph, LangChain, and Streamlit. This project implements a Corrective Retrieval-Augmented Generation (C-RAG) architecture to provide highly accurate, hallucination-free answers.
If the agent cannot find the answer in the provided local documents, it automatically falls back to a web search. It also features an internal "grader" loop that checks its own generated answers for hallucinations before presenting them to the user.

✨ Features
Graph-Based Orchestration: Uses LangGraph to control the logical flow of data, allowing for complex loops and conditional routing.
Intelligent Document Grading: Evaluates retrieved document chunks for relevance. If chunks are irrelevant, it triggers a web search.
Web Search Fallback: Integrates Tavily Search API to dynamically pull information from the web when local documents fall short.
Self-Correction & Hallucination Checking: Employs a secondary LLM chain to grade the final generated answer against the source documents. If it detects a hallucination or failure to answer the prompt, it triggers a retry.
Local Vector Database: Uses ChromaDB and HuggingFace Embeddings (all-MiniLM-L6-v2) for fast, local semantic search.
Interactive UI & User Feedback: Built with Streamlit, featuring integrated LangSmith feedback buttons (👍/👎) to track agent performance in production.

🏗️ Architecture Flow
This agent operates on a state machine built with LangGraph:
Retrieve: Grabs semantic chunks from ChromaDB based on the user's question.
Grade Documents: An LLM evaluates if the chunks are actually relevant to the query.
Web Search (Conditional): If the documents are not relevant, the agent searches the web via Tavily.Generate: The LLM drafts an answer using the confirmed context (Local + Web).
Check Hallucination: A strict grader checks if the drafted answer is entirely factual based on the context. If it hallucinates, the graph loops back to re-generate.

🛠️ Tech StackFramework: LangChain & LangGraphLLM: Groq (llama-3.3-70b-versatile for high-speed generation)Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)Vector Store: Chroma DBWeb Search: Tavily Search APIFrontend: StreamlitTracing & Observability: LangSmith🚀 Getting Started1. PrerequisitesYou will need API keys for the following services (all offer generous free tiers):Groq API Key: For the LLaMA 3 model.Tavily API Key: For the web search fallback.LangSmith API Key: For tracing and logging feedback.2. InstallationClone the repository and navigate into the directory:Bashgit clone https://github.com/yourusername/crag-assistant.git
cd crag-assistant
Create a virtual environment and install the dependencies:Bashpython3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
3. Environment VariablesCreate a .env file in the root directory and add your API keys:Ini, TOMLGROQ_API_KEY="your_groq_api_key"
TAVILY_API_KEY="your_tavily_api_key"

# LangSmith Configuration
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="crag-tech-support"
🏃‍♂️ UsageStep 1: Ingest Your DocumentsBefore running the chatbot, you need to build the local knowledge base. Place your .pdf, .md, or .txt files into a folder named docs/ in the root directory.Run the ingestion script:Bashpython ingest.py
This will parse your documents, split them into chunks, generate embeddings, and save the ChromaDB vector store locally to chroma_db/.Step 2: Run the ApplicationOnce ingestion is complete, start the Streamlit UI:Bashstreamlit run app.py
Open the provided localhost URL in your browser to interact with your C-RAG assistant!📂 Project StructureFileDescriptionapp.pyStreamlit frontend interface and LangSmith feedback handling.ingest.pyPipeline to load, split, embed, and persist documents to ChromaDB.retriever.pyConfigures the vector store retriever with similarity score thresholds.graph.pyDefines the LangGraph state machine, nodes, and conditional routing edges.nodes.pyContains the actual logic/functions for each step in the graph workflow.chains.pyConfigures the LLM prompts and Pydantic models for structured output/grading.state.pyDefines the GraphState TypedDict to manage variables across the workflow.requirements.txtPython package dependencies.🤝 ContributingContributions, issues, and feature requests are welcome! Feel free to check the issues page.












