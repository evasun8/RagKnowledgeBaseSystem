import numpy as np
def normalize_sparse_vector(sparse_vec):
    """
    Performs L2 normalization on a sparse vector (processes only non-zero dimensions, leaving zero dimensions unaffected).
    :param sparse_vec: Raw sparse vector (dict format: {dimension: value})
    :return: L2 normalized sparse vector
    """
    if not sparse_vec:  # Return directly if the vector is empty
        return sparse_vec

    # Extract values of non-zero dimensions
    values = np.array(list(sparse_vec.values()), dtype=np.float64)
    # Calculate L2 norm (avoiding division by zero)
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:  # If the norm is close to 0, return the original vector directly to prevent division-by-zero errors
        return sparse_vec

    # Normalization: divide each value by the L2 norm
    normalized_values = values / l2_norm
    # normalized_values = (values / l2_norm).astype(np.float32)  # Unified conversion to float32
    # Reconstruct the sparse vector dict
    return dict(zip(sparse_vec.keys(), normalized_values))