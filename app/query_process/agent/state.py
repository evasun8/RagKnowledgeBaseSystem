from typing_extensions import TypedDict
from typing import List


class QueryGraphState(TypedDict):
    """
    QueryGraphState defines the data structure flowing through the entire query pipeline.
    """
    session_id: str  # Unique session identifier
    original_query: str  # Original query submitted by the user

    # Intermediate data generated during the retrieval phase
    embedding_chunks: list  # Chunks retrieved via standard vector search
    hyde_embedding_chunks: list  # Chunks retrieved via HyDE (Hypothetical Document Embeddings) search
    kg_chunks: list  # Chunks retrieved from the Knowledge Graph
    web_search_docs: list  # Documents fetched via web search

    # Data processed during the ranking phase
    rrf_chunks: list  # Chunks sorted after Reciprocal Rank Fusion (RRF)
    reranked_docs: list  # Final Top-K documents after reranking

    # Data used in the generation phase
    prompt: str  # Fully assembled Prompt
    answer: str  # Final generated answer

    # Helper/Metadata information
    item_names: List[str]  # Extracted product/item names
    rewritten_query: str  # Rewritten query optimized for search
    history: list  # Historical conversation records
    is_stream: bool  # Flag indicating whether streaming output is enabled