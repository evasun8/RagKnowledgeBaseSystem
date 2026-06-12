import sys
import os
from typing import List, Dict, Any

from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

def step_3_reciprocal_rank_fusion(
    source_with_weight: List[tuple], 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs Reciprocal Rank Fusion (RRF) on multi-source retrieval results
    and applies corresponding weight coefficients to calculate unified scores.

    :param source_with_weight: List of tuples containing (retrieval_results, weight)
                               e.g., [(embedding_chunks, 1.0), (hyde_embedding_chunks, 1.0)]
    :param top_k: Number of top ranked results to return.
    :return: Ranked and merged list of chunk dictionaries.
    """
    # 1. Prepare container to track consolidated scores
    score_dict = {}  # Map of chunk_id -> calculated RRF score (float)
    
    # 2. Prepare container to preserve unique chunk payloads
    chunk_dict = {}  # Map of chunk_id -> chunk dictionary (avoids duplicate objects)
    # 3. Iterate through each retrieval source and calculate scores
    for source, weight in source_with_weight:
        # source format sample:
        # [{'id': 'milvus_pk', 'distance': 0.8, 'entity': {'chunk_id': '...', 'content': '...'}}, ...]
        if not source:
            continue
        for rank, chunk in enumerate(source, start=1):
            # Retrieve unique chunk_id safely
            chunk_id = chunk.get("id") or chunk.get("entity", {}).get("chunk_id")
            if not chunk_id:
                continue

            # Apply RRF score formula with custom weight multiplier:
            # RRF_Score = (1.0 / (60 + rank)) * weight
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (60 + rank)) * weight
            
            # Keep the first occurrence of the chunk to prevent duplicate entities
            chunk_dict.setdefault(chunk_id, chunk)
             # 4. Merge RRF scores with their respective chunk payloads
    merged = []
    for chunk_id, score in score_dict.items():
        chunk = chunk_dict.get(chunk_id)
        if chunk:
            merged.append((chunk, score))

     # Sort merged chunks in descending order based on unified scores
    merged.sort(key=lambda x: x[1], reverse=True)

    # 5. Extract top_k ranked candidates
    merged = merged[:top_k]

    # 6. Extract chunk bodies from the ranked list
    rank_chunks = [chunk for chunk, score in merged]
    logger.info(f"RRF ranking completed successfully. Results: {rank_chunks}")
    return rank_chunks


def node_rrf(state: dict) -> dict:
    """
    LangGraph node function: Reciprocal Rank Fusion.
    Consolidates and ranks retrieval results from multiple sources (standard vector, HyDE, etc.).
    
    :param state: LangGraph global state object.
    :return: Updated state dictionary containing ranked results inside "rrf_chunks".
    """
    print("---RRF---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1. Extract raw data chunks from both standard embedding search and HyDE search
    embedding_chunks = state.get("embedding_chunks") or []
    hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []

    # 2. Consolidate sources with their respective weights (currently defaulting to 1.0)
    source_with_weight = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]
    # 3. Apply RRF algorithm to obtain consolidated ranked chunks
    rrf_response = step_3_reciprocal_rank_fusion(source_with_weight)

    # 4. Inject ranked results into state under "rrf_chunks"
    state["rrf_chunks"] = rrf_response
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state