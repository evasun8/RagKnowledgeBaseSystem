import sys 
import os 
from typing import Any, List, Dict 

from app.import_process.agent.state import ImportGraphState 
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings 
from app.utils.task_utils import add_running_task, add_done_task 
from app.core.logger import logger 

# ========================================== 
# BGE-M3 Embedding Core Node 
# Core Capabilities: Converts text chunks into dense and sparse hybrid vectors, providing the data foundation for Milvus vector retrieval. 
# Dependent Model: BAAI/bge-m3 (multi-lingual, multi-granular; simultaneously supports semantic and keyword retrieval). 
# Vector Description: 
# 1. Dense Vector: Fixed length of 1024 dimensions, captures deep semantic information of the text, used for semantic similarity matching. 
# 2. Sparse Vector: Variable-length key-value pairs, captures keywords and feature position information of the text, used for precise keyword matching. 
# Core Design: 
# - Singleton Model: Avoids duplicate model loading, saving VRAM and runtime. 
# - Batch Processing: Generates embeddings in batches to prevent Out-Of-Memory (OOM) errors caused by massive batch sizes. 
# - Text Enrichment: Concatenates item name + chunk content to reinforce core features, enhancing retrieval accuracy. 
# ========================================== 

def step_1_validate_input(state: ImportGraphState) -> List[Dict[str, Any]]: 
    """
    Embedding Pre-step 1: Input Data Validity Validation
    
    Core Functions:
    1. Extract the text chunks list awaiting vectorization from the global state.
    2. Strictly validate the data type and non-emptiness of chunks, terminating vectorization if no valid data exists.
    
    Parameters:
    state: ImportGraphState - The global pipeline state object.
    
    Returns:
    List[Dict[str, Any]] - The validated list of text chunks.
    
    Exceptions:
    Raises a ValueError if chunks is not a list or is empty, terminating the current embedding pipeline stage.
    """ 
    # Extract chunk data from the state object
    texts_to_embed = state.get("chunks") 
    
    # Validation: Must be a non-empty list; otherwise, vectorization cannot proceed
    if not isinstance(texts_to_embed, list) or not texts_to_embed: 
        logger.error("Embedding input validation failed: 'chunks' field is empty or not a valid list.") 
        raise ValueError("Error: No valid text chunk data found; unable to execute embedding processing.") 
        
    logger.info(f"Embedding input validation passed. Number of text chunks to process: {len(texts_to_embed)}") 
    return texts_to_embed

def step_2_init_model(): 
    """
    Embedding Step 2: Initialize BGE-M3 Model Instance (Singleton Pattern)
    
    Core Functions:
    1. Call the singleton function get_bge_m3_ef to ensure the model is loaded only once globally.
    2. Validate the validity of the model instance, throwing an explicit exception if loading fails.
    
    Returns:
    Any - A valid BGE-M3 model instance (embedding function).
    
    Exceptions:
    Raises a ValueError pointing out configuration problems when model loading fails (due to invalid paths, VRAM insufficiency, or missing dependencies).
    """ 
    try: 
        # Retrieve the singleton model instance to prevent duplicate loading and resource waste
        ef = get_bge_m3_ef() 
        
        # Validate whether the model instance is valid
        if ef is None: 
            raise ValueError("BGE-M3 model instance is None: pymilvus.model module not found or model loading failed.") 
            
        logger.info("BGE-M3 model instance initialized successfully (Singleton Pattern).") 
        return ef 
    except Exception as e: 
        # Wrap the exception details to clarify the root cause and provide troubleshooting directions
        error_msg = f"BGE-M3 model initialization failed: {e}. Please check if the model path or environment variable configurations are correct." 
        logger.error(error_msg) 
        raise ValueError(error_msg)

def step_3_generate_embeddings(texts_to_embed: List[Dict[str, Any]], bge_m3_ef: Any) -> List[Dict[str, Any]]: 
    """
    Embedding Core Step 3: Batch-generate dense and sparse hybrid vectors
    
    Core Logic (Executed in batches with isolated exception handling per batch):
    1. Text Enrichment: Concatenates item_name + newline + content to reinforce core contextual features.
    2. Batch Invocations: Passes the enriched texts into the model to generate hybrid vectors in batches.
    3. Vector Binding: Clones original metadata records for each chunk and attaches newly injected 'dense_vector' and 'sparse_vector' fields.
    4. Exception Fallback: Retains raw chunk structures if a specific batch fails, smoothly continuing execution of subsequent batches.
    
    Parameters:
    texts_to_embed: List[Dict[str, Any]] - The validated list of text chunks containing 'item_name' and 'content' fields.
    bge_m3_ef: Any - The BGE-M3 model instance initialized in Step 2.
    
    Returns:
    List[Dict[str, Any]] - The list of text chunks containing embedded vector fields; failed batches retain raw data layouts.
    
    Key Configuration:
    batch_size: Set to 5 records per batch. This can be adapted based on server VRAM capacity (scale up for higher VRAM, scale down for lower VRAM).
    """ 
     # Initialize the results container to warehouse vector-enriched chunk data
    output_data = [] 
    
    # Batch size configuration: Balance memory consumption footprints against computational scaling throughput. Adjust to actual environments.
    batch_size = 5 
    
    # Iterate progressively in chunks to safeguard against VRAM Out-of-Memory (OOM) exceptions caused by massive single-payload submittals
    total = len(texts_to_embed) 
    for i in range(0, total, batch_size): 
        # Slices the current sub-batch workload; the final step naturally accommodates remaining fractional elements (grabs 5 items at a time)
        batch_texts = texts_to_embed[i:i + batch_size] 
        
        # Calculate human-readable 1-based indexing offsets for execution log visual displays (harmless to iteration bounds)
        start_idx, end_idx = i + 1, min(i + len(batch_texts), total) 
        try: 
            # Construct model entry payloads: Concat parent names with content arrays to reinforce index characteristics
            input_texts = [] 
            for doc in batch_texts: 
                item_name = doc["item_name"] 
                content = doc["content"] 
            # Prepend the subject identity if present (newlines boost parsing capability), else revert strictly to context structures.
                # Almost all embedding architectures (especially those adapting BERT-derived weights) center maximum attention matrices 
                # around the first 128 tokens. Tokens positioned later exert structurally weaker vector pulling forces.
                # **"Core Keyword Front-loading"** design tenet.
                # Option 1: Apply robust punctuation marks instead of carriage-returns (Simplest and highly recommended).
                # Pre-optimized: Apple Phone\nExtreme runtime capabilities...
                # Post-optimized: Apple Phone. Extreme runtime capabilities...
                # Option 2: Embellish with a microscopic pinch of semantic boilerplate (Best for strictly predefined feature-set configurations).
                text = f"Product: {item_name}, Description: {content}" if item_name else content 
                
                # Embedding models strictly prioritize formatting symmetry. If processing Chinese text, enforce native double-byte punctuation; 
                # if handling English prose, apply regular ASCII symbols. Maintaining context purity optimizes downstream vector generation quality!
                input_texts.append(text)
            # Query the underlying driver function to compute multi-vectors. Format expectation: {"dense": [dense_list], "sparse": [sparse_list]}
            docs_embeddings = generate_embeddings(input_texts) 
            if not docs_embeddings: 
                logger.warning(f"Chunks {start_idx}-{end_idx}: Embedding suite returned empty payload. Reverting to original source structures.") 
                output_data.extend(batch_texts) 
                continue
            
            # Bind the matching vector slices onto each individual chunk, shallow-copying objects to prevent up-stream reference contamination
            for j, doc in enumerate(batch_texts): 
                item = doc.copy() 
                item["dense_vector"] = docs_embeddings["dense"][j]    # Bind deep semantic dense float arrays
                item["sparse_vector"] = docs_embeddings["sparse"][j]  # Bind keyword positional sparse weight maps (already normalized)
                output_data.append(item) 
                
            logger.info(f"Chunks {start_idx}-{end_idx}: Dual vector generation successful.")
        except Exception as e: 
            # Intercept sub-batch operational exceptions, tracking state diagnostics without breaking loop processing
            logger.error( 
                f"Chunks {start_idx}-{end_idx}: Embedding generation pipeline failed. Retaining fallback states | Reason: {str(e)}", 
                exc_info=True 
            ) 
            # Preserve raw chunk layouts during faulty runs to maintain data integrity for downstream manual auditing
            output_data.extend(batch_texts) 
            continue 
            
    return output_data     
    
    
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState: 
    """
    LangGraph Core Node: BGE-M3 Text Vectorization Processing
    
    Main Pipeline (Sequential execution with full-process exception isolation):
    1. Input Validation: Verifies the validity of chunks; terminates the current node execution if core data is missing.
    2. Model Initialization: Fetches the BGE-M3 singleton model instance to prevent redundant loading overhead.
    3. Batch Embedding: Concatenates text and generates hybrid vectors in batches, binding vector attributes to individual chunks.
    4. State Synchronization: Flushes the vector-enriched chunks back to the global state for consumption by the downstream Milvus storage node.
    
    Parameters:
    state: ImportGraphState - The global pipeline state object containing chunks, task_id, and other data passed downstream.
    
    Returns:
    ImportGraphState - The updated state object, where chunks now contain newly attached 'dense_vector' and 'sparse_vector' fields.
    
    Exception Handling:
    All exceptions within the node are caught. The overall LangGraph pipeline will not terminate; errors are simply recorded in the logs.
    """ 
    # Retrieve the current node name for logging and task status recording
    current_node = sys._getframe().f_code.co_name 
    logger.info(f">>> Starting execution of LangGraph node: {current_node}") 
    
    # Mark task execution status for task monitoring and frontend progress dashboard displays
    add_running_task(state.get("task_id", ""), current_node) 
    logger.info("--- BGE-M3 Text Embedding Processing Triggered ---")
    try: 
        # Step 1: Input data validation; throws an exception if the core chunks are invalid
        texts_to_embed = step_1_validate_input(state) 
        
        # Step 2: Initialize the BGE-M3 model (Singleton pattern, loads only once)
        bge_m3_ef = step_2_init_model() 
        
        # Step 3: Batch-generate hybrid vectors and bind vector fields to chunks
        output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef) 
        
        # Step 4: Update the global state, passing the vector-enriched chunks to downstream nodes
        state['chunks'] = output_data 
        logger.info(f"--- BGE-M3 Embedding process completed. Processed a total of {len(output_data)} text chunks ---") 
        add_done_task(state.get("task_id", ""), current_node) 
        
    except Exception as e: 
        # Catch all node exceptions and log the error stack trace without interrupting the overall pipeline orchestration
        logger.error(f"BGE-M3 Embedding node execution failed: {str(e)}", exc_info=True) 
        
    # Return the updated state object to be handed over to downstream nodes
    return state


# ========================================== 
# Local Unit Testing Entrypoint 
# Functionality: Independently validates the full-pipeline logic of the embedding node without needing to spin up the entire LangGraph orchestration engine. 
# Applicable Scenarios: Local development, debugging, and verifying model deployment validity. 
# ========================================== 

if __name__ == '__main__': 
    # Load environment variables: Locates the .env file under the project root directory to read model paths and device runtime settings
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(os.path.dirname(current_dir)) 
    
    # Construct a mocked test state object: Simulates text chunk datasets produced by upstream nodes to mirror production environments
    test_state = ImportGraphState({ 
        "task_id": "test_task_embedding_001",  # Test Task ID 
        "chunks": [ 
            # Mocked text chunks enriched with 'item_name' fields (produced by the upstream item name recognition node) 
            { 
                "content": "This is the content of a test document, used to verify whether the vectorization process is successful.", 
                "title": "Test Document Title", 
                "item_name": "Test Project Item", 
                "file_title": "test_file.pdf" 
            }, 
            { 
                "content": "This is the content of a second test document, used to validate multi-vector batch processing logic.", 
                "title": "Test Document Title 2", 
                "item_name": "Test Project Item", 
                "file_title": "test_file.pdf" 
            } 
        ] 
    }) 
    
    # Execute the local test harness
    logger.info("=== Local Unit Testing for BGE-M3 Embedding Node Triggered ===") 
    try: 
        # Invoke the core node function 
        result_state = node_bge_embedding(test_state) 
        
        # Extract the processed testing outputs 
        result_chunks = result_state.get("chunks", []) 
        
        # Print runtime test statistics 
        logger.info("=== Local Testing for Embedding Node Completed ===") 
        logger.info(f"Test Task ID: {test_state.get('task_id')}") 
        logger.info(f"Chunks Awaiting Processing: 2 | Actual Chunks Processed: {len(result_chunks)}") 
        logger.info(f"Vector Dimensions Trace: {result_chunks}") 
        
        # Verify execution outcomes (asserting whether the injected vector attributes are resident) 
        for idx, chunk in enumerate(result_chunks): 
            has_dense = "dense_vector" in chunk 
            has_sparse = "sparse_vector" in chunk 
            logger.info( 
                f"Chunk Record #{idx + 1}: Dense vector generation {'succeeded' if has_dense else 'failed'} | " 
                f"Sparse vector generation {'succeeded' if has_sparse else 'failed'}." 
            ) 
            
    except Exception as e: 
        logger.error( 
            f"=== Local Testing for Embedding Node Failed === " 
            f"Reason for Failure: {str(e)}", 
            exc_info=True 
        ) 
        # Onboarding-friendly diagnostic notification: Outlines core system troubleshooting paths 
        logger.warning("Troubleshooting Tips: Please check the BGE-M3 local path configurations, VRAM capacity thresholds, and project environment variable maps.")