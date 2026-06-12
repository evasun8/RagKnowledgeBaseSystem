from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    Defines the graph state, containing all data fields produced and consumed by the nodes.
    TypedDict provides autocompletion and type checking within the IDE.
    Access fields using dictionary-style routing (e.g., state["session_id"], state.get("embedding_chunks")).
    """
    task_id: str          # Unique task ID for execution logging and lifecycle tracking

    # --- Control Flow Flags ---
    is_md_read_enabled: bool   # Flag to enable/disable the Markdown parsing pipeline
    is_pdf_read_enabled: bool  # Flag to enable/disable the PDF ingestion pipeline

    # --- Chunking-Related Fields --- [Deprecated / Unused]
    is_normal_split_enabled: bool
    is_silicon_flow_api_enabled: bool
    is_advanced_split_enabled: bool
    is_vllm_enabled: bool

    # --- Path-Related Configuration ---
    local_dir: str        # Current working directory or artifact output path
    local_file_path: str  # Original source input file path
    file_title: str       # File title extracted (filename stripped of extension)
    pdf_path: str         # Source PDF filepath (populated if input is a PDF)
    md_path: str          # Markdown filepath (converted or explicitly routed)
    split_path: str       # Target filepath for text chunking slices [Deprecated / Unused]
    embeddings_path: str  # Vector database file layout configuration [Deprecated / Unused]

    # --- Content & Payload Storage ---
    md_content: str       # Full raw text string parsed from the Markdown file
    chunks: list          # Granular slice list containing text payloads and descriptive metadata
    item_name: str        # Extracted domain entity identifier (e.g., "Multimeter") for hybrid search boost

    # --- Storage & Database Mapping ---
    embeddings_content: list # Array structure encapsulating vectors, ready for Milvus batch ingestion


# Recommended to define an initialization template for downstream node mutations
# Defines the baseline initial values for the Graph State
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    """
    Factory function to initialize a default graph state with dictionary overrides.

    Args:
        **overrides: Key-value pairs representing state fields to mutate during instantiation.

    Returns:
        ImportGraphState: A freshly instantiated graph state dictionary.

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # Create a deep copy of the default state template
    state = copy.deepcopy(graph_default_state)
    # Patch the state instance using keyword arguments
    state.update(overrides)
    # Return the validated state dict instance
    return state

def get_default_state() -> ImportGraphState:
    """
    Returns a fresh state instance to prevent global variable mutability side-effects.
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    Unit Testing / Smoke Test
    """
    # Instantiate a mock state with a local source file payload
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    logger.info(state)