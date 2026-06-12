# Import base libraries: system, paths, and type annotations (type annotations improve code readability and maintainability)
import os
import sys
from typing import List, Dict, Any, Tuple

# Import Milvus client (core vector database operations) and data type enums (for defining Collection Schema)
from pymilvus import MilvusClient, DataType

# Import LangChain message classes (standardizes the conversation message format for LLMs)
from langchain_core.messages import SystemMessage, HumanMessage

# Import custom modules:
# 1. Pipeline state carrier: ImportGraphState serves as the unified state management object for the LangGraph workflow
from app.import_process.agent.state import ImportGraphState

# 2. Milvus utilities: Retrieves the singleton Milvus client to implement connection reuse
from app.clients.milvus_utils import get_milvus_client

# 3. LLM utilities: Retrieves the LLM client to provide a unified entry point for model calls
from app.lm.lm_utils import get_llm_client

# 4. Embedding utilities: BGE-M3 model instances and vector generation methods (dense + sparse vectors)
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings

# 5. Sparse vector utilities: Normalization processing to ensure vector length is 1, improving retrieval accuracy
from app.utils.normalize_sparse_vector import normalize_sparse_vector

# 6. Task utilities: Updates task execution status for task monitoring and management
from app.utils.task_utils import add_running_task

# 7. Logging utilities: Unified project logging entry point for tiered output levels (info/warning/error)
from app.core.logger import logger

# 8. Prompt utilities: Loads local prompt templates to decouple prompts from the source code
from app.core.load_prompt import load_prompt

# 9. String escaping utilities: Escapes special characters for Milvus-compatible string handling
from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- Configuration Parameters ---

# Number of context chunks for LLM item name recognition: 
# Take the first 5 chunks to prevent the context from becoming too long and exceeding LLM input limits.
DEFAULT_ITEM_NAME_CHUNK_K = 5

# Truncation length for a single chunk's content: 
# Prevents a single chunk from being excessively long and flooding the LLM context window.
SINGLE_CHUNK_CONTENT_MAX_LEN = 800

# Upper limit for total characters in the LLM context: 
# Adapts to the input limits of mainstream LLMs, defaults to 2500 characters.
CONTEXT_TOTAL_MAX_CHARS = 2500

def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]: 
    """
    Step 1: Receive and validate pipeline inputs (preprocessing data for item name recognition)
    
    Core Functions:
    1. Extract the file title and text chunk core data from the pipeline state.
    2. Perform multi-layer null-value fallback handling to prevent subsequent pipeline stages from crashing due to null values.
    3. Validate basic data types to ensure the validity of inputs for downstream stages.
    
    Dependent State Data (produced by upstream nodes):
    - state["file_title"]: File title extracted upstream (used with higher priority).
    - state["file_name"]: Original file name (used as fallback when file_title is empty).
    - state["chunks"]: List of text chunks (each chunk is a dict containing fields like title/content).
    
    Returns:
    Tuple[str, List[Dict]]: (Processed file title, validated list of text chunks)
    """ 
    # Multi-layer fallback to get the file title: priority goes to file_title -> then file_name -> default to empty string
    file_title = state.get("file_title", "") or state.get("file_name", "") 
    
    # Retrieve the text chunk list: return an empty list if it is a null value to prevent errors in subsequent loops
    chunks = state.get("chunks") or [] 
    
    # Secondary fallback: if file_title is still empty, try extracting it from the first valid chunk
    if not file_title: 
        if chunks and isinstance(chunks[0], dict): 
            file_title = chunks[0].get("file_title", "") 
            logger.warning("No valid file_title found in state; extracted fallback title from the first chunk instead.") 
            
    # Null value log prompt: do not interrupt the pipeline if the file title is empty, just log a warning
    if not file_title: 
        logger.warning("Both file_title and file_name are missing from state; subsequent LLM recognition accuracy may decrease.") 
        
    # Data type validation: ensure chunks is a valid non-empty list, otherwise return an empty list
    if not isinstance(chunks, list) or not chunks: 
        logger.warning("The 'chunks' field in state is either empty or not a list type; item name recognition cannot proceed.") 
        return file_title, [] 
        
    logger.info(f"Step 1: Input validation completed; retrieved {len(chunks)} valid text chunks.") 
    return file_title, chunks

def step_2_build_context(chunks: List[Dict], k: int = DEFAULT_ITEM_NAME_CHUNK_K, max_chars: int = CONTEXT_TOTAL_MAX_CHARS) -> str:
    """
    Step 2: Construct standardized context for LLM item/product name recognition.
    Core Functions:
    1. Restrict chunk volume: Evaluates only the top-k chunks to prevent oversized contexts.
    2. Enforce character limits: Dual-layer constraint (single chunk threshold + overall context threshold) to respect LLM input windows.
    3. Format content: Generates an indexed, structured text layout to optimize LLM recognition accuracy.
    4. Filter invalid chunks: Bypasses empty strings or non-dict structures to maintain high data quality.
    
    :param chunks: List of text chunks (each element must be a dictionary containing 'title' and 'content' keys).
    :param k: Maximum number of chunks to process. Defaults to 5 (tunable via configuration).
    :param max_chars: Hard ceiling for total character count in the final context string. Defaults to 2500.
    :return: A formatted, structured context string ready for LLM processing. Returns an empty string if input is blank.
    """
    # Empty chunks boundary check: return immediately to save compute overhead
    if not chunks:
        return ""
        
    # Stores the formatted text snippets to guarantee a structured final context layout
    parts: List[str] = []
    
    # Track the aggregated character count in real time to prevent context overflow
    total_chars = 0
    # Iterate through the first k chunks to protect against massive, multi-page contexts
    for idx, chunk in enumerate(chunks[:k]):
        # Filter non-dict types to eliminate key-lookup runtime crashes
        if not isinstance(chunk, dict):
            logger.debug(f"Chunk at index {idx+1} is not a dictionary type; skipped.")
            continue
            
        # Extract title and body text, stripping surrounding trailing spaces and filtering invalid formats
        chunk_title = chunk.get("title", "").strip()
        chunk_content = chunk.get("content", "").strip()
        # Skip the snippet entirely if both fields yield empty text
        if not (chunk_title or chunk_content):
            logger.debug(f"Chunk at index {idx+1} contains blank content; skipped.")
            continue
        # Single-chunk truncation layer: stops any individual rogue chunk from hijacking the full token quota
        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]
            logger.debug(f"Chunk at index {idx+1} content is too long; truncated to {SINGLE_CHUNK_CONTENT_MAX_LEN} characters.")
        
        # Apply structural layout formatting: embeds indices, headings, and context markers to boost LLM ingestion performance
        piece = f"[Chunk {idx + 1}]\nTitle: {chunk_title} \nContent: {chunk_content}"
        parts.append(piece)
        # Accumulate total string length including structural spacing overhead
        total_chars += len(piece)
    # Join text fragments using empty line spacing breaks and apply a final trim for structural safety
    context = "\n\n".join(parts).strip()
    
    # Secondary fallback truncation to explicitly guarantee constraint boundaries are respected
    final_context = context[:max_chars]
    logger.info(f"Step 2: Context construction complete. Final length: {len(final_context)} characters.")
    return final_context


def step_3_call_llm(file_title: str, context: str) -> str:
    """
    Step 3: Invoke the Large Language Model to achieve precise item/product name and model recognition.
    Core Logic:
    1. Empty Context -> Directly return the file_title (Fallback bypass, skipping LLM invocation).
    2. Non-Empty Context -> Load standardized prompt templates and construct chat messages.
    3. Clean the response strings returned by the LLM, stripping whitespaces and control characters.
    4. Blank Responses / Exception Overheads -> Uniformly fall back to file_title to guarantee workflow continuity.
    
    Core Features:
    - Decoupled Prompts: Utilizes 'load_prompt' to fetch external template files, eliminating hardcoded strings.
    - Format Compatibility: Tolerates diverse structural variations across different LLM client responses to prevent object attribute crashes.
    - Exception Resilience: Global try-except block traps errors, ensuring main loop pipeline remains unblocked during LLM downtime.
    
    :param file_title: The pre-processed filename title, serving as the master fallback string.
    :param context: The structured, text-chunk context compiled in Step 2 (the primary analytical input for the LLM).
    :return: The cleaned product name string. Returns the original file_title upon error or empty results.
    """
    logger.info("Initiating Step 3: Invoking LLM for product name recognition.")
    
    # Boundary check: If context is empty, skip calling the LLM entirely and immediately fall back to the filename title.
    if not context:
        logger.warning("Context is empty. Skipping LLM invocation and utilizing the file title directly as the product name.")
        return file_title
    try:
        # Load the dedicated item name recognition prompt template, dynamically embedding the file title and context variables.
        human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
        # Load system prompts to define the agent persona (Product Recognition Expert instruction to output clean text only).
        system_prompt = load_prompt("product_recognition_system")
        logger.debug(f"LLM dialog prompt constructed: System prompt length {len(system_prompt)}, Human prompt length {len(human_prompt)}")
        # Instantiate the unified LLM client: set json_mode to False to request raw string text over JSON payloads.
        llm = get_llm_client(json_mode=False)
        if not llm:
            logger.error("Failed to retrieve an active LLM client instance. Using file title fallback.")
            return file_title
        # Structure standardized LangChain chat payloads: SystemMessage configures behavioral rules + HumanMessage passes task arguments.
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        # Trigger model inference execution and capture response metadata.
        resp = llm.invoke(messages)
        # Compatibility handling across various client objects: safely grab 'content' attribute, defaulting to an empty string.
        item_name = getattr(resp, "content", "").strip()
        # Fallback check: If post-processing leaves the item name blank, resort back to the file title.
        if not item_name:
            logger.warning("LLM returned an empty or unparsable token response. Defaulting to file title fallback.")
            return file_title
            
        logger.info(f"Step 3: LLM product name recognition successful. Identified output string: {item_name}")
        return item_name
        
    # Global exception safety net: trap server timeouts, network disconnects, or serialization bugs to protect the pipeline.
    except Exception as e:
        logger.error(f"Step 3: LLM invocation failed. Root Cause: {str(e)}", exc_info=True)
        # Return file title fallback under failure conditions to preserve state execution continuity.
        return file_title
    
def step_4_update_chunks(state: ImportGraphState, chunks: List[Dict], item_name: str): 
    """
    Step 4: Backfill the item name to the pipeline state and all text chunks
    
    Core Functions:
    1. Global state update: Save item_name into the state so that all downstream nodes can use it directly.
    2. Chunk data completion: Add the item_name field to each chunk to ensure data consistency.
    3. State synchronization: Update chunks in the state to ensure that modifications to chunks take effect globally.
    
    Design Strategy:
    Associate all chunks with the same item name to ensure dimensional consistency during subsequent vector storage and retrieval.
    
    Parameters:
    state: The pipeline state object (ImportGraphState), which is the global data carrier.
    chunks: The validated list of text chunks (output from Step 1).
    item_name: The item name recognized and cleaned in Step 3.
    """ 
    # Save the item name into the global state for downstream nodes to call
    state["item_name"] = item_name 
    
    # Iterate through all chunks and add the item_name field to each chunk to ensure full-pipeline data consistency
    for chunk in chunks: 
        chunk["item_name"] = item_name 
        
    # Synchronously update the chunk list in the state to ensure the modifications take effect globally
    state["chunks"] = chunks 
    logger.info(f"Step 4: Item name backfill completed; added the item_name field to a total of {len(chunks)} chunks with a value of: {item_name}")

def step_5_generate_vectors(item_name: str) -> Tuple[Any, Any]: 
    """
    Step 5: Generate BGE-M3 dense and sparse hybrid vectors for the item name (core of Milvus vector retrieval)
    
    Core Description:
    - Dense vector (dense_vector): BGE-M3 fixed 1024-dimensional vector, captures the deep semantic information of the text.
    - Sparse vector (sparse_vector): Variable-length key-value pairs, captures text keywords and feature position information.
    
    Dependent Tools:
    generate_embeddings: Encapsulates the BGE-M3 model to batch-generate hybrid vectors; compatible with both single and batch inputs.
    
    Parameters:
    item_name: The item name recognized in Step 3 (must be non-empty; returns empty vectors directly if null).
    
    Returns:
    Tuple[Any, Any]: (Dense vector list, sparse vector dictionary); returns (None, None) upon null values or exceptions.
    """ 
    logger.info(f"Starting Step 5: Generating BGE-M3 hybrid vectors for item name [{item_name}]") 
    
    # If the item name is empty, skip model invocation and return empty vectors directly
    if not item_name: 
        logger.warning("Item name is empty; skipping embedding generation and returning empty vectors.") 
        return None, None 
    try: 
        # Invoke the embedding generation utility: pass a list to support batch generation; use a list for single entries to ensure unified formatting
        vector_result = generate_embeddings([item_name]) 
        
        # Proceed with parsing only if the embedding generation result is non-empty
        if vector_result and "dense" in vector_result and "sparse" in vector_result: 
            # Dense vector parsing: take the first item of the batch results, formatted as a Python list (required for Milvus storage)
            dense_vector = vector_result["dense"][0] 
            
            # Sparse vector parsing: take the first item of the batch results, parsing the CSR matrix into a dictionary format
            sparse_vector = vector_result["sparse"][0] 
            logger.info("Step 5: BGE-M3 dense and sparse vector generation successful.") 
        else: 
            logger.warning("Step 4: The embedding utility returned an empty result; unable to extract hybrid vectors.") 
            dense_vector, sparse_vector = None, None 
            
    # Catch all exceptions: model loading failures, embedding timeouts, formatting errors, etc.
    except Exception as e: 
        logger.error(f"Step 5: Embedding generation failed. Reason: {str(e)}", exc_info=True) 
        dense_vector, sparse_vector = None, None 
    return dense_vector, sparse_vector

def step_6_save_to_milvus(state: ImportGraphState, file_title: str, item_name: str, dense_vector, sparse_vector): 
    """
    Step 6: Persist the item name, file title, and hybrid vectors into the Milvus vector database
    
    Core Logic:
    1. Configuration Validation: Check Milvus connection URL and collection name settings; skip if missing.
    2. Client Retrieval: Fetch the singleton Milvus client instance; skip if connection fails.
    3. Collection Initialization: Create the collection (define Schema + index) if non-existent, otherwise reuse it directly (preserving existing configs).
    4. Idempotency Handling: Delete historical data with the same item name to prevent duplicate storage.
    5. Data Insertion: Construct data aligning with the Schema; append only non-null vector attributes.
    6. Collection Loading: Force-load the collection post-insertion to ensure data is immediately searchable and visible in Attu.
    
    Parameters:
    state: The pipeline state object used for final state synchronization.
    file_title: The processed file title string.
    item_name: The recognized item name (acts as the primary deduplication key).
    dense_vector: The dense vector generated in Step 5 (a 1024-dimensional list).
    sparse_vector: The sparse vector generated in Step 5 (dictionary format).
    """ 
    # Read core Milvus configs from environment variables, aligning with the MilvusConfig settings class
    milvus_uri = os.environ.get("MILVUS_URL") 
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
     # Configuration missing check: skip Milvus storage and log a warning if either config is empty
    if not all([milvus_uri, collection_name]): 
        logger.warning("Milvus configuration missing (MILVUS_URL/ITEM_NAME_COLLECTION); skipping data persistence.") 
        return 
    try: 
        # Retrieve the Milvus singleton client; return immediately if connection fails
        client = get_milvus_client() 
        if not client: 
            logger.error("Failed to retrieve the Milvus client (connection error); skipping data persistence.") 
            return
        # Collection Initialization: create if non-existent (define Schema + Indexes), use directly if it exists
        if not client.has_collection(collection_name=collection_name): 
            logger.info(f"Milvus collection [{collection_name}] does not exist; starting Schema and Index creation.") 
            
            # Create collection Schema: auto-increment primary key + dynamic fields to adapt to flexible data storage
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            # Add auto-increment primary key field: INT64 type, uniquely identifying each record
            schema.add_field( 
                field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True 
            ) 
            
            # Add file title field: VARCHAR type, max length 65535, supporting long titles
            schema.add_field( 
                field_name="file_title", datatype=DataType.VARCHAR, max_length=65535 
            ) 
            
            # Add item name field: VARCHAR type, max length 65535, used as deduplication criteria
            schema.add_field( 
                field_name="item_name", datatype=DataType.VARCHAR, max_length=65535 
            ) 
            
            # Add dense vector field: FLOAT_VECTOR, 1024 dimensions (fixed dimension for BGE-M3)
            schema.add_field( 
                field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024 
            ) 
             # Add sparse vector field: SPARSE_FLOAT_VECTOR, variable length
            schema.add_field( 
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR 
            ) 
             # Build index parameters: create indexes for vector fields to optimize retrieval performance
            index_params = client.prepare_index_params() 
            
            # Optimized dense vector index: HNSW + COSINE (restoring best performance configuration)
            index_params.add_index( 
                field_name="dense_vector", 
                index_name="dense_vector_index", 
                # HNSW (Hierarchical Navigable Small World) is currently the highest-performing and most widely used 
                # graph-based index, providing lightning-fast search speeds and extreme precision.
                index_type="HNSW", 
                # Use COSINE similarity for dense vector distance metric calculations
                metric_type="COSINE", 
                # M: Maximum number of connection links per node in the graph (commonly 16-64)
                # efConstruction: Search scope during index construction (larger values increase build time but improve precision, commonly 100-200)
                params={"M": 16, "efConstruction": 200} 
            ) 
            # Sparse vector index: Dedicated SPARSE_INVERTED_INDEX + IP, turning off quantization to guarantee precision
            index_params.add_index( 
                field_name="sparse_vector", 
                index_name="sparse_vector_index", 
                # Sparse Inverted Index: An inverted index purpose-built for sparse vectors (e.g., text TF-IDF vectors or keyword 
                # weight vectors, where most elements are 0 and only a few dimensions hold values). It is the standard index type for sparse retrieval.
                index_type="SPARSE_INVERTED_INDEX", 
                # IP (Inner Product): If the vector represents "text semantic vector + keyword weights" where length dictates the 
                # correlation strength between the text and topic, using IP captures both "semantic matching degree" and "correlation intensity".
                metric_type="IP", 
                # DAAT_MAXSCORE is a highly efficient algorithm for sparse retrieval. quantization="none" ensures no loss in sparse vector weights.
                params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"} 
            ) 
            # Create collection: Schema + Index parameters
            client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params) 
            logger.info(f"Milvus collection [{collection_name}] created successfully with Schema and Vector Indexes.") 
        # Idempotency handling: Delete historical data with the same item name to prevent duplicate entries (Core: collection must be loaded before deletion)
        clean_item_name = (item_name or "").strip() 
        if clean_item_name: 
            client.load_collection(collection_name=collection_name) 
            
            # Escape the item name string to prevent special characters from breaking the filtering expression parser
            safe_item_name = escape_milvus_string(clean_item_name) 
            filter_expr = f'item_name=="{safe_item_name}"' 
            
            # Execute deletion operation
            client.delete(collection_name=collection_name, filter=filter_expr) 
            logger.info(f"Milvus idempotency cleanup completed; historical data for [{clean_item_name}] has been purged from the collection.") 
            
        # Construct payload for Milvus insertion: basic fields + non-null vector fields
        data = { 
            "file_title": file_title, 
            "item_name": item_name 
        } 
        # Append dense vector only if it is non-null to prevent DB errors during insertion
        if dense_vector is not None: 
            data["dense_vector"] = dense_vector 
            
        # Normalize and append sparse vector if non-null to maximize retrieval accuracy
        if sparse_vector is not None: 
            data["sparse_vector"] = normalize_sparse_vector(sparse_vector) 
            
        # Insert data: wrapped in a list to support batch payloads, keeping single record formats standardized
        client.insert(collection_name=collection_name, data=[data]) 
        
        # Force-load collection post-insertion to ensure data is immediately searchable and visible in the Attu visualization UI
        client.load_collection(collection_name=collection_name) 
        
        # Sync the final item name to the global state object
        state["item_name"] = item_name 
        logger.info(f"Step 6: Item name [{item_name}] successfully persisted into Milvus collection [{collection_name}]. Fields stored: {list(data.keys())}") 
    # Catch all Milvus runtime exceptions (connection drops, payload failures, index mismatches, etc.) without halting the pipeline
    except Exception as e: 
        logger.error(f"Step 6: Failed to persist data into Milvus. Reason: {str(e)}", exc_info=True)

        
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    [Core Node] Item/Subject Name Recognition (node_item_name_recognition)
    
    Overall Process: Extract inputs -> Build context -> LLM recognition -> Backfill data -> Generate embeddings -> Store in Milvus
    Core Purpose: Leverage LLMs to accurately recognize item/subject names from document chunks, generate hybrid embeddings (dense + sparse), and save them to the database.
    Future Extension Points: Support multi-subject recognition, add item attribute extraction, integrate with other vector databases, etc.
    
    :param state: Project state dictionary (ImportGraphState), must contain 'chunks', 'file_title', and 'task_id'.
    :return: Updated state dictionary, adding the 'item_name' key, with each element in the 'chunks' list updated with an 'item_name' field.
    """ 
    # Initialize current node info for task monitoring and log traceability
    node_name = sys._getframe().f_code.co_name 
    logger.info(f">>> Starting core node execution: [Item Name Recognition] {node_name}") 
    
    # Add current node to the running tasks to update global task status
    add_running_task(state.get("task_id", ""), node_name) 
    try:
        # ===================================== Step 1: Extract & Validate Input Data ===================================== 
        # Action: Extract file title and chunk list from the state dict, validate data integrity.
        # Output: File title, chunk list; throws exception or terminates if no chunks are found.
        file_title, chunks = step_1_get_inputs(state) 
        if not chunks: 
            logger.warning(f">>> Node execution warning: {node_name} (No valid chunk data found), skipping recognition") 
            return state 
        
        # ===================================== Step 2: Build LLM Recognition Context ===================================== 
        # Action: Intercept the first N chunks and concatenate their content into an LLM-readable context to aid recognition.
        # Output: Concatenated context string.
        context = step_2_build_context(chunks) 

        # ===================================== Step 3: Call LLM to Recognize Item Name ===================================== 
        # Action: Construct Prompt and call the LLM to extract the core item name from the context and title.
        # Output: Recognized item name string (e.g., "iPhone 15 Pro").
        item_name = step_3_call_llm(file_title, context)
        
        # ===================================== Step 4: Backfill Item Name to State and Chunks ===================================== 
        # Action: Write the recognition result into the state dict and synchronously update the metadata of each Chunk object.
        # Output: The 'item_name' key is added to the state dict; the 'chunks' list is modified in place.
        step_4_update_chunks(state, chunks, item_name) 
        
        # ===================================== Step 5: Generate Hybrid Embeddings (Dense + Sparse) ===================================== 
        # Action: Call BGE-M3 model to generate dense semantic vectors and sparse keyword vectors for the item name.
        # Output: dense_vector (List[float]), sparse_vector (Dict[int, float]).
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)

         # ===================================== Step 6: Save to Milvus Vector Database ===================================== 
        # Action: Save the item name and its hybrid embeddings into the 'item_names' collection in Milvus for subsequent retrieval.
        # Output: No return value; data is successfully persisted.
        step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector) 

        # Node completion log
        logger.info(f">>> Core node execution completed: [Item Name Recognition] {node_name}. Recognition result: {item_name}, stored in Milvus.") 
        
    except Exception as e: 
        # Global exception handling: ensure a node failure won't crash the entire pipeline; log detailed error for troubleshooting.
        logger.error(f">>> Core node execution failed: [Item Name Recognition] {node_name}. Error message: {str(e)}", exc_info=True) 
        # Optional: Set a default value or flag state upon failure
        state["item_name"] = "Unknown Item" 

    # Return the updated state (for downstream nodes to consume)
    return state


# ===================== Local Testing Method (Run directly to debug without starting LangGraph) =====================

def test_node_item_name_recognition(): 
    """
    Local testing method for the item name recognition node.
    
    Functionality: Mocks LangGraph pipeline inputs to independently test the full-pipeline logic of the node_item_name_recognition node.
    Applicable Scenarios: Local development, debugging, and single-node feature validation without needing to spin up the entire LangGraph orchestration engine.
    
    Prerequisites:
    1. Ensure all project environment variables are properly configured (e.g., MILVUS_URL, ITEM_NAME_COLLECTION).
    2. Ensure the LLM provider, Milvus instance, and BGE-M3 embedding service are fully accessible.
    3. Ensure the prompt templates (item_name_recognition / product_recognition_system) are present locally.
    
    Usage:
    Execute the function directly: if __name__ == "__main__": test_node_item_name_recognition()
    """ 
    logger.info("=== Starting Local Testing for Item Name Recognition Node ===") 
    try: 
        # 1. Construct a mocked ImportGraphState object (simulating outputs generated by upstream nodes)
        mock_state = ImportGraphState({ 
            "task_id": "test_task_123456",                            # Test Task ID
            "file_title": "Huawei Mate60 Pro Smartphone User Manual",  # Mocked file title
            "file_name": "Huawei_Mate60Pro_Manual.pdf",               # Mocked original filename (used as a fallback)
            # Mocked text chunk list (produced by upstream chunking nodes, containing title and content fields)
            "chunks": [ 
                { 
                    "title": "Product Overview", 
                    "content": "The Huawei Mate60 Pro is a flagship smartphone released by Huawei in 2023. It is powered by the Kirin 9000S chipset, supports satellite calling features, boasts a 6.82-inch display screen, and features a resolution of 2700×1224." 
                }, 
                { 
                    "title": "Camera Capabilities", 
                    "content": "The Huawei Mate60 Pro features a rear camera setup consisting of a 50MP Ultra Aperture camera, a 12MP Ultra-Wide Angle camera, and a 48MP Telephoto camera, supporting 5x optical zoom and 100x digital zoom." 
                }, 
                { 
                    "title": "Battery Specifications", 
                    "content": "Equipped with a 5000mAh battery capacity, it supports 88W wired SuperCharge, 50W wireless SuperCharge, and reverse wireless charging capabilities." 
                } 
            ] 
        }) 
        
        # 2. Invoke the core item name recognition node function
        result_state = node_item_name_recognition(mock_state) 
        
        # 3. Print test execution metrics (for debugging purposes)
        logger.info("=== Local Testing for Item Name Recognition Node Completed ===") 
        logger.info(f"Test Task ID: {result_state.get('task_id')}") 
        logger.info(f"Final Recognized Item Name: {result_state.get('item_name')}") 
        logger.info(f"Total Chunks Processed: {len(result_state.get('chunks', []))}") 
        logger.info(f"First Chunk Item Name Field: {result_state.get('chunks', [{}])[0].get('item_name')}") 
        
        # 4. Validate Milvus storage data persistence (Optional)
        milvus_client = get_milvus_client() 
        collection_name = os.environ.get("ITEM_NAME_COLLECTION") 
        if milvus_client and collection_name: 
            milvus_client.load_collection(collection_name) 
            
            # Retrieve test transaction results from the database
            item_name = result_state.get('item_name') 
            safe_name = escape_milvus_string(item_name) 
            res = milvus_client.query( 
                collection_name=collection_name, 
                filter=f'item_name=="{safe_name}"', 
                output_fields=["file_title", "item_name"] 
            ) 
            logger.info(f"Data retrieved from Milvus query matching fingerprint: {res}") 
            
    except Exception as e: 
        logger.error(f"Local testing for item name recognition node failed. Reason: {str(e)}", exc_info=True) 

# Test execution entrypoint: execute this file directly to trigger the workflow testing cycle
if __name__ == "__main__": 
    # Execute the local test harness
    test_node_item_name_recognition()