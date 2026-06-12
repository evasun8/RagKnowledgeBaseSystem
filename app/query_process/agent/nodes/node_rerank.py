import sys
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

load_dotenv()

# -----------------------------
# Rerank / TopK Global Constants
# -----------------------------
# Dynamic TopK Upper Hard Limit: Retrieves up to N records (<=10)
RERANK_MAX_TOPK: int = 10
# Minimum TopK: Retains at least N records (>=1, and <= RERANK_MAX_TOPK)
RERANK_MIN_TOPK: int = 1
# Relative cliff-drop ratio threshold
RERANK_GAP_RATIO: float = 0.25
# Absolute cliff-drop score threshold (Maximum split score gap)
RERANK_GAP_ABS: float = 0.5

def step_1_merge_rrf_mcp(state: dict) -> List[Dict[str, Any]]:
    """
    Consolidates RRF ranked chunks and Web Search (MCP) documents into a unified list.
    
    :param state: LangGraph global state object.
    :return: Merged list of unstructured documents.
    """
    # 1. Retrieve data from different sources
    rrf_chunks = state.get("rrf_chunks") or []
    web_search_docs = state.get("web_search_docs") or []
    
    chunks_list = []
    # ---------------------------------------------------------
    # 2. Merge RRF and Web Search results
    # ---------------------------------------------------------
    for idx, chunk in enumerate(rrf_chunks, start=1):
        # Convert RRF chunk to a dictionary
        entity = chunk.get('entity') or {}
        chunk_id = entity.get('chunk_id') or chunk.get('id') or ""
        content = entity.get('content') or chunk.get('content') or ""
        title = entity.get('title') or ""
        
        chunks_list.append({
            "chunk_id": chunk_id,
            "text": content,
            "title": title,
            "source": "local",
            "url": ""
        })
    #  Web Search (MCP) results
    for doc in web_search_docs:
        text = doc.get("snippet") or ""
        url = doc.get("url") or ""
        title = doc.get("title") or ""
        
        chunks_list.append({
            "chunk_id": "",
            "text": text,
            "title": title,
            "source": "web",
            "url": url
        })
    logger.info(f"Multi-route data consolidation completed. Total merged items: {len(chunks_list)}")
    return chunks_list
 
def step_2_rerank_doc_list(doc_list: List[Dict[str, Any]], state: dict) -> List[Dict[str, Any]]:
    """
    Performs precise ranking on the merged documents using the Reranker Cross-Encoder model.
    
    :param doc_list: Consolidated list of documents.
    :param state: LangGraph global state object.
    :return: Ranked documents list decorated with cross-encoder scores.
    """
    if not doc_list:
        logger.warning("Document list is empty; skipping cross-encoder rerank.")
        return [] 
    # 1. Retrieve the validated user query
    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    # 2. Extract text fields from documents
    text_list = [doc['text'] for doc in doc_list]    
    # 3. Load Rerank model
    reranker = get_reranker_model() 
    # 4. Formulate query-document pairs to compute relevance scores
    questions_pairs = [[rewritten_query, text] for text in text_list]
    # Compute relevance scores scaled to a [0, 1] range via normalize=True
    scores = reranker.compute_score(questions_pairs, normalize=True)
    doc_list_with_score = []
    for score, item in zip(scores, doc_list):
        item['score'] = float(score)
        doc_list_with_score.append(item)
    # Sort in descending order of relevance scores
    doc_list_with_score.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"Precision reranking completed. Sorted results: {doc_list_with_score}")
    return doc_list_with_score

def step_3_apply_cliff_drop_filter(reranked_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters and truncates the sorted documents list based on cliff-dropping (gap analysis) heuristics
    and dynamic Top-K constraints.
    
    Cliff-Drop logic:
    We iterate through the sorted scores. If we encounter a score gap between two adjacent items
    that is relatively large (relative gap > RERANK_GAP_RATIO or absolute gap > RERANK_GAP_ABS),
    we truncate the results at that gap, provided the retained count falls within [RERANK_MIN_TOPK, RERANK_MAX_TOPK].
    """
    if not reranked_list:
        return []
    
    retained_count = len(reranked_list)
    search_limit = min(RERANK_MAX_TOPK, len(reranked_list))
    
    # Identify the cliff-dropping gaps
    for i in range(1, search_limit):
        current_score = reranked_list[i-1].get("score")
        next_score = reranked_list[i].get("score") 
        gap = current_score - next_score
        
        # Check for cliff-dropping gaps
        relative_threshold = RERANK_GAP_RATIO * current_score if current_score > 0 else 0
        
        if gap > relative_threshold or gap > RERANK_GAP_ABS:
            retained_count = i
            break
    final_topk = min(RERANK_MAX_TOPK, retained_count)
    final_topk = max(RERANK_MIN_TOPK, final_topk)
    truncated_results = reranked_list[:final_topk]
    logger.info(f"Cliff-drop & TopK dynamic filtering applied. Retained {len(truncated_results)} out of {len(reranked_list)} documents.")
    return truncated_results
    # Truncate the resul
         
def node_rerank(state: dict) -> dict:
    """
    LangGraph node function: Rerank.
    Consolidates data (RRF + MCP Web Docs) -> Performs precision reranking -> Applies cliff-dropping filter -> Saves Top-K results.
    
    Heuristic algorithm details:
    - Retains between RERANK_MIN_TOPK (1) and RERANK_MAX_TOPK (10) results.
    - Dynamically drops low-ranking documents when a sharp cliff (absolute/relative score drop) is identified.
    """
    logger.info("---Starting Rerank Node Execution---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    
    # 1. Consolidate multi-route retrieval outputs into a unified list
    doc_list = step_1_merge_rrf_mcp(state)
    
    # 2. Perform cross-encoder precision reranking
    rerank_score_list = step_2_rerank_doc_list(doc_list, state)
    
    # 3. Apply cliff-dropping filter & Dynamic Top-K truncation logic
    final_retained_docs = step_3_apply_cliff_drop_filter(rerank_score_list)
    
    # 4. Save results back into LangGraph state
    state["reranked_docs"] = final_retained_docs
 
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
 

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> Starting node_rerank Local Unit Test")
    print("=" * 50)
 
    # 1. Setup Mock Inputs
    # 1.1 RRF local document mock results
    mock_rrf_chunks = [
        {"entity": {"chunk_id": "local_1", "content": "RRF (Reciprocal Rank Fusion) is an ensemble ranking algorithm.", "title": "RRF Introduction"}},
        {"entity": {"chunk_id": "local_2", "content": "BGE is a powerful cross-encoder reranking model.", "title": "BGE Model Overview"}},
        {"entity": {"chunk_id": "local_3", "content": "Completely irrelevant mock document regarding weather.", "title": "Weather Details"}}  # Expected low score
    ]
 
    # 1.2 Web Search (MCP) mock results
    mock_web_docs = [
        {"title": "Detailed Rerank Tech Guide", "url": "http://web.com/1", "snippet": "Reranking processes document chunks sequentially during stage two of RAG pipelines."},
        {"title": "Unrelated Web Article", "url": "http://web.com/2", "snippet": "It is a nice day for a picnic outside."}  # Expected low score
    ]
 
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "What are RRF and Rerank?",  # Query Intent: Wants to understand both algorithms
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }
 
    try:
        # Run node execution
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])
 
        print("\n" + "=" * 50)
        print(">>> Test Results Summary:")
        print(f"Total Input Documents: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"Total Output Documents: {len(reranked)}")
        print("-" * 30)
 
        print("Final ranked documents:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:40]}...")
 
        # Logical Verification:
        # - "local_1", "local_2", and "Detailed Rerank Tech Guide" should score highly.
        # - "local_3" and "Unrelated Web Article" should have lower scores and might get filtered out via cliff-dropping rules.
        if reranked:
            top1_score = reranked[0].get("score", 0.0)
            if top1_score > 0:
                print("\n[PASS] Rerank scored successfully.")
            else:
                print("\n[FAIL] Rerank returned anomalous score configurations.")
        else:
            print("\n[FAIL] Rerank returned zero documents.")
 
        print("=" * 50)
 
    except Exception as e:
        logger.exception(f"Unhandled exception encountered during test execution: {e}")