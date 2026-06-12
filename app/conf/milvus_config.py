# Import core dependencies (shared with other config classes, imported only once)
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load the .env configuration file in advance (executed once globally, no need to duplicate)
load_dotenv()

# ===================== Other configuration classes (LLM/Embedding) can be placed above, keeping original code unchanged =====================
# ... Your LLMConfig, EmbeddingConfig code ...

# Define the configuration class for the Milvus Vector Database
@dataclass
class MilvusConfig:
    milvus_url: str          # Connection URI/URL for the Milvus server
    chunks_collection: str   # Name of the collection storing text chunks
    entity_name_collection: str  # Reserved - Name of the collection for entity names
    item_name_collection: str    # Name of the collection storing item/document classes

# Instantiate the Milvus configuration object (aligning with the naming convention of other config objects)
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)