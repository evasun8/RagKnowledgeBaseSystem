from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# Model singleton object to avoid duplicate initialization
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    Retrieves the BGE-M3 model singleton instance, automatically loading configurations from environment variables.
    :return: An initialized BGEM3EmbeddingFunction instance.
    """
    global _bge_m3_ef
    # Singleton pattern: return the instance directly if already initialized to avoid reloading the model
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3 model singleton instance already exists; returning the instance directly.")
        return _bge_m3_ef

    # Load configuration from environment variables; fall back to defaults if not configured
    # Local paths can be used if available! Otherwise, "BAAI/bge-m3" will be automatically downloaded!
    # A URL address can also be used for cloud deployments!
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    # Log model initialization configurations to ease troubleshooting
    logger.info(
        "Starting initialization of the BGE-M3 model",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True # Vector Normalization
        }
    )

    try:
        # Initialize BGE-M3 model with native L2 normalization enabled (tailored for Milvus IP inner product retrieval)
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True  # Native L2 normalization applied by the model for both dense and sparse vectors
        )
        logger.success("BGE-M3 model initialized successfully with native L2 normalization enabled.")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3 model initialization failed: {str(e)}", exc_info=True)
        raise  # Re-raise the exception to be handled by the caller


def generate_embeddings(texts):
    """
    Generates dense + sparse hybrid vector embeddings for a list of texts (with model-native L2 normalization).
    :param texts: List of texts to generate embeddings for; a single text must also be wrapped in a list.
    :return: A dictionary containing vector results where keys are 'dense'/'sparse', mapping to nested lists/list of dicts.
    :raise: Exceptions during vector generation, to be caught and handled by the caller.
    """
    # Validate input argument compliance
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("Invalid input argument for vector generation; 'texts' must be a non-empty list.")
        raise ValueError("The 'texts' parameter must be a non-empty list containing text data.")

    logger.info(f"Starting hybrid vector embedding generation for {len(texts)} text items.")
    try:
        # Load the BGE-M3 model singleton instance
        model = get_bge_m3_ef()
        # Encode documents using the model, which returns dense vectors + CSR-formatted sparse vectors
        embeddings = model.encode_documents(texts)
        logger.debug(f"Model encoding completed; starting sparse vector format parsing for {len(texts)} items.")

        # Initialize sparse vector processing results, parsing into dictionary formats (tailored for serialization/storage)
        processed_sparse = []
        for i in range(len(texts)):
            # Extract sparse vector indices for the i-th text: np.int64 -> Python int (satisfies dictionary key hashable requirement)
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # Extract sparse vector weights for the i-th text: np.float32 -> Python float (tailored for JSON serialization/API responses)
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # Construct the sparse vector dictionary formatted as {feature_index: normalized_weight}
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # Construct the final return result, converting dense vectors to lists (resolves NumPy array non-serializable issue)
        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # Nested list, one-to-one mapping with input texts
            "sparse": processed_sparse  # List of dictionaries, natively L2-normalized by the model
        }
        logger.success(f"Vector generation completed for {len(texts)} text items; format adapted for production-grade utilization.")
        return result

    except Exception as e:
        logger.error(f"Text vector generation failed: {str(e)}", exc_info=True)
        raise  # Do not swallow exceptions; propagate upward to allow the caller to handle retries/graceful degradation


"""
Core Design Highlights & Adaptation Specifications:
1. Model-Native Normalization: By enabling normalize_embeddings=True, L2 normalization is automatically applied to both dense and sparse vectors. This perfectly adapts to Milvus IP (Inner Product) retrieval (once unit-normalized, IP is mathematically equivalent to Cosine similarity but computes significantly faster).
2. Elimination of NumPy-as-Key Issues: Appending .tolist() to sparse_indices converts np.int64 to native Python ints, satisfying the hashable requirement for dictionary keys and eliminating runtime crash risks.
3. Serialization-Ready Sparse Values: Appending .tolist() to sparse_data converts np.float32 to native Python floats, seamlessly supporting JSON writes, API responses, Milvus data ingestion, and other production scenarios.
4. Singleton Pattern Optimization: The model is initialized exactly once, avoiding costly and resource-heavy reloads, thereby boosting batch processing efficiency.
5. Business-Logic Format Alignment: Returns dense nested lists and sparse dictionary lists, aligning perfectly with standard downstream parsing patterns like vector_result["dense"][0] / sparse_vector["sparse"][0].
6. Layered Logging Coverage: Comprehensive logs span from model initialization and vector generation to exception traces, facilitating seamless root-cause analysis in production environments.
7. Input Compliance Validation: Prevents internal execution errors triggered by empty lists or invalid types, significantly enhancing utility class robustness.
"""