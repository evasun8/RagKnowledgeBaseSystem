import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# Global Milvus client instance to implement singleton reuse
_milvus_client = None


def get_milvus_client():
    """
    Singleton retrieval method for the Milvus client.
    Implements client connection reuse to avoid resource consumption from repeated connection creation.
    :return: MilvusClient instance; returns None if connection fails.
    """
    try:
        global _milvus_client
        # Singleton check: create a new connection if not initialized
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            # Validate Milvus connection URI configuration
            if not milvus_uri:
                logger.error("Milvus client connection failed: Missing MILVUS_URL environment variable configuration")
                return None
            # Initialize Milvus client
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus client connected successfully")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus client connection exception: {str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """
    Converts chunk_ids to the INT64 type required by Milvus (primary key field schema is INT64).
    Filters out invalid IDs, separating convertible and non-convertible IDs.
    :param ids: List of chunk_ids to be converted.
    :return: A tuple of (ok_ids, bad_ids), where ok_ids is a list of successfully converted int64 IDs, and bad_ids is a list of invalid IDs.
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """
    Batch queries chunk data in Milvus via chunk_id primary keys.
    Used to complete chunk information in scenarios where only "chunk_id" is available without text content.
    Prioritizes the get method (primary key direct look-up, optimal performance), falling back to query filtering on failure.
    :param client: MilvusClient instance.
    :param collection_name: Collection name.
    :param chunk_ids: List of chunk_ids to be queried.
    :param output_fields: List of fields to be returned; defaults to core chunk fields.
    :param batch_size: Batch size for querying to avoid oversized single requests; defaults to 100.
    :return: List[dict], a list of Milvus entity dictionaries; returns an empty list if the query fails.
    """
    # Pre-validation: return empty immediately if client or collection_name is invalid
    if client is None:
        return []
    if not collection_name:
        return []
    # Default return fields: core chunk identification and content fields
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    # Convert IDs to INT64 type, separating valid and invalid IDs
    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        # Log invalid IDs and skip querying them
        logger.warning(f"Invalid chunk_ids found that cannot be converted to INT64; skipping query for: {bad_ids}")

    # Return empty directly if no valid IDs exist
    if not ok_ids:
        return []

    results = []
    # Batch querying: slice valid IDs by batch_size and loop queries
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # Approach 1: Prioritize the primary key 'get' method query (optimal performance)
        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus 'get' method query failed; falling back to 'query' method: {str(e)}")

        # Approach 2: If 'get' method fails, fall back to using filter-based query
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus 'query' method batch lookup for chunk_ids failed: {str(e)}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    Constructs Milvus hybrid search request objects.
    Creates separate search requests for dense and sparse vectors respectively, to be used later for hybrid search fusion.
    :param dense_vector: Dense vector generated from text.
    :param sparse_vector: Sparse vector generated from text.
    :param dense_params: Search parameters for dense vector; defaults to cosine similarity.
    :param sparse_params: Search parameters for sparse vector; defaults to inner product similarity.
    :param expr: Search filter expression used for precise data screening.
    :param limit: Number of search results returned per single vector; defaults to 5.
    :return: List of search requests containing [dense_req, sparse_req].
    """
    # Default dense vector search params: Cosine similarity (COSINE), matching BGE-M3 dense vectors and indexing parameters
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # Default sparse vector search params: Inner Product (IP), matching BGE-M3 sparse vectors
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # Construct the dense vector search request, mapped to the 'dense_vector' field in Milvus (Core class for ANN search requests)
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # Construct the sparse vector search request, mapped to the 'sparse_vector' field in Milvus
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    Executes Milvus dense + sparse vector hybrid search.
    Implements weighted fusion of dual-vector search results based on WeightedRanker to enhance retrieval accuracy.
    :param client: MilvusClient instance.
    :param collection_name: Collection name.
    :param reqs: List of search requests, fixed as [dense_req, sparse_req].
    :param ranker_weights: Weighted fusion weights, defaults to (0.5, 0.5), corresponding to dense/sparse vectors sequentially.
    :param norm_score: Whether to normalize scores before fusion to prevent weight invalidation caused by differences in score scale.
    :param limit: Final number of results returned from hybrid search; defaults to 5.
    :param output_fields: List of fields to be returned; defaults to item_name.
    :param search_params: Search parameters, such as ef/topk, etc.; defaults to None.
    :return: List of hybrid search results; returns None if search fails.
    """
    try:
        # Initialize the weighted ranker: fuses search results of dense and sparse vectors by weights
        # norm_score=True: normalizes scores of both vectors into the 0~1 interval first, then calculates weighted values
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # Default return fields: document identification field
        if output_fields is None:
            output_fields = ["item_name"]

        # Execute hybrid search: fuses dense + sparse vector results and reranks based on weights
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus hybrid search completed; retrieved {len(res[0])} results from collection [{collection_name}]")
        return res
    except Exception as e:
        logger.error(f"Failed to execute Milvus hybrid search on collection [{collection_name}]: {str(e)}", exc_info=True)
        return None
    