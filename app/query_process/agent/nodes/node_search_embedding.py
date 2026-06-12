import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def node_search_embedding(state: QueryGraphState) -> Dict[str, Any]:
    """
    Core node function: Executes Milvus vector database hybrid retrieval based on the 
    confirmed product names and the rewritten user query.
    
    Flow: Vectorize user query -> Construct hybrid search request with product name filtering -> 
          Perform dense + sparse hybrid retrieval -> Return retrieval results.
          
    :param state: Dict - Session state dictionary containing core information passed from upstream nodes. Key fields:
                  {
                      "session_id": str,        # Unique session identifier
                      "rewritten_query": str,   # Rewritten query from step 3 (containing product name)
                      "item_names": list[str],  # Standardized product names list confirmed in step 6
                      "is_stream": bool/None    # Flag indicating if streaming is enabled, optional
                  }
    :return: Dict - Retrieval results dictionary containing only the embedding_chunks field, to be consumed by downstream nodes:
             {
                 "embedding_chunks": List[Dict]  # Milvus retrieval results, empty list if no matches.
                                                 # Each element is a matching vector record with business metadata fields.
             }
    """
    logger.info("---search_milvus starting execution---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    
    # 1. Extract core input parameters from session state to prepare for retrieval
    query = state.get("rewritten_query")  # Extract rewritten query (containing product name, self-contained)
    item_names = state.get("item_names")  # Extract confirmed standardized product names list (used for precise filtering)
    
    logger.info(f"Extracted core inputs: query='{query}', item_names={item_names}")
    
    # 2. Vectorize the rewritten user query to generate BGEM3 dense + sparse vectors
    logger.info(f"Starting embedding generation for text: {query[:50]}..." if len(query) > 50 else f"Starting embedding generation for text: '{query}'...")
    # Call the embedding generation function (accepts a list; only a single query here)
    # Generate semantic vectors matching the product name for similarity retrieval
    embeddings = generate_embeddings([query])
    
    dense_vec = embeddings.get("dense")[0]
    sparse_vec = embeddings.get("sparse")[0]
    
    # Log dense and sparse vector metadata for debugging purposes
    logger.debug(f"Embeddings generated successfully: dense_dim={len(dense_vec)}, sparse_len={len(sparse_vec)}")
    
    # 3. Prepare configurations for Milvus vector database connection, specifying the target collection
    # Retrieve Milvus collection name for text chunk embeddings from env variables to avoid hardcoding
    collection_name = os.environ.get("CHUNKS_COLLECTION")
    logger.info(f"Connecting to Milvus and preparing collection: '{collection_name}'...")
    
    # If item_names is empty, skip retrieval and return an empty result
    if not item_names:
        logger.warning("item_names is empty; skipping retrieval and returning empty list")
        return {"embedding_chunks": []}
     # Add double quotes to each product name and format as Milvus 'in' syntax
    quoted = ", ".join(f'"{v}"' for v in item_names)
    # Construct final filter expression
    expr = f"item_name in [{quoted}]"
    logger.info(f"Created search request filter expression: {expr}")
 
    # Construct dense + sparse hybrid search requests, combining vectors, filter expressions, and limits
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vec,  # Fetch dense vector of the query (index 0 for single query)
        sparse_vector=sparse_vec,  # Fetch sparse vector of the query (index 0 for single query)
        expr=expr,  # Filter expression narrowing the scope to the confirmed standard items
        limit=10  # Low-level retrieval limit (will be filtered down to 5 later, reserving extra results for reranking)
    )
    # 5. Perform Milvus dense + sparse hybrid retrieval (core call)
    logger.info("Executing Milvus hybrid retrieval...")
    client = get_milvus_client()
    res = hybrid_search(
        client=client,
        collection_name=collection_name,  # Target vector collection
        reqs=reqs,  # Dense + sparse hybrid search requests structure
        ranker_weights=(0.8, 0.2),  # Adjust score weight ratios to optimize search precision
        norm_score=True,  # Normalize scores to transform distance outputs into 0-1 similarity ratings
        limit=5,  # Limit the final aligned results to Top-5
        output_fields=["chunk_id", "content", "item_name"]  # Specify target business fields
    )
    # Log successful processing and output sample raw results for debugging
    hit_count = len(res[0]) if res and len(res) > 0 else 0
    logger.info(f"Node search_embedding executed successfully. Retrieved {hit_count} relevant chunks.")
    if hit_count > 0:
        logger.debug(f"Top 1 retrieval result sample: {res[0][0]}")
         
    # Mark task as completed and update task status
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
 
    # 6. Construct and return results: Extract res[0] (matching Milvus batch search format) if results exist, otherwise return empty list
    # res[0] represents retrieval hits for the single query, containing Top-5 matching records with metadata
    return {"embedding_chunks": res[0] if res else []}
 
 