# Import core dependencies: dataclasses, environment variable parser, and path utilities
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load the .env configuration file in advance (aligned with original code, executed only once)
load_dotenv()

@dataclass
class RerankerConfig:
    bge_reranker_large: str  # Local path to the model
    bge_reranker_device: str # Execution device (cuda:0/cpu)
    bge_reranker_fp16: bool  # Whether to enable half-precision (1=True/0=False)

# Instantiate the configuration object, aligning with the architectural style of lm_config
reranker_config = RerankerConfig(
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE"),
    # Special handling: convert 1/0 from .env into boolean values, ensuring compatibility with common numeric/string formats
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1)
)