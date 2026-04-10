from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    generation: str
    web_fallback: bool
    hallucination: bool
    retry_count: int
    documents: List[Document]
    sources_used: Optional[List[str]]
    search_type: Optional[str]