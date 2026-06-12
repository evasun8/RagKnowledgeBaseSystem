# Import core dependencies: dataclasses, environment variable parser, and path utilities
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load the .env configuration file in advance (aligned with original code, executed only once)
load_dotenv()

# Define the Embedding configuration (supports all specifications for BGE-M3, class named embedding_config)
@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # Local path to the model
    bge_m3: str       # Model repository identifier
    bge_device: str   # Execution device (cuda:0/cpu)
    bge_fp16: bool    # Whether to enable half-precision (1=True/0=False)

# Instantiate the configuration object, aligning with the architectural style of lm_config
embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    # Special handling: convert 1/0 from .env into boolean values, ensuring compatibility with common numeric/string formats
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)
)