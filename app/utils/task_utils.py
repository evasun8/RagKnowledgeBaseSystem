from typing import Dict, List
from .sse_utils import push_to_session

# ---------------------------
# In-Memory Task Tracking (Single Process)
# ---------------------------
# key: task_id
# value: List of node names (Raw English / Node ID)
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

# key: task_id
# value: status string (e.g., pending/processing/completed/failed)
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: Task results (e.g., the answer to a query)
_tasks_result: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# Node Name -> English Display Name Mapping (Used for front-end presentation)
# Note: The keys here must strictly match the node names defined in LangGraph's add_node("xxx", ...)
_NODE_NAME_TO_EN: Dict[str, str] = {
    "upload_file": "Uploading File",
    "node_entry": "Validating File",
    "node_pdf_to_md": "Converting PDF to Markdown",
    "node_md_img": "Processing Markdown Images",
    "node_item_name_recognition": "Extracting Entity Names",
    "node_document_split": "Splitting Document Chunks",
    "node_bge_embedding": "Generating Vector Embeddings",
    "node_import_kg": "Importing to Knowledge Graph",
    "node_import_milvus": "Importing to Vector DB",
    "__end__": "Processing Completed",
    "END": "Processing Completed",
    # --- Query Flow Nodes (kb/query_process/main_graph.py) ---
    "node_item_name_confirm": "Confirming Product Context",
    "node_answer_output": "Generating Final Answer",
    "node_rerank": "Reranking Search Results",
    "node_rrf": "Executing Reciprocal Rank Fusion",
    "node_web_search_mcp": "Performing Web Search",
    "node_search_embedding": "Searching Text Chunks",
    "node_search_embedding_hyde": "Searching Chunks via HyDE",
    "node_multi_search": "Executing Multi-Route Search",
    "node_query_kg": "Querying Knowledge Graph",
    "node_join": "Merging Multi-Route Search Results",
}


def _ensure_task(task_id: str) -> None:
    """Ensures that the data structures corresponding to the task_id are initialized."""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_en(node_name: str) -> str:
    """Converts a node name into its English display name; returns the original name if no mapping exists."""
    return _NODE_NAME_TO_EN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    Adds a node task to the "running" list.

    Parameters:
    - task_id: The ID of the task.
    - node_name: The name of the node (Node ID).
    """
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    # Prevent duplicate appending
    if node_name not in running:
        running.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    Adds a node task to the "done" list.

    Note: When adding a completed task, any "running" task with the same name will be removed.

    Parameters:
    - task_id: The ID of the task.
    - node_name: The name of the node (Node ID).
    """
    _ensure_task(task_id)

    # 1) Remove the same-named node from the running list (handles potential duplicates, removes all)
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 2) Append to the done list (maintains execution completion order) while preventing duplicates
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    Stores a task result field (such as 'answer' or 'error').
    """
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    Retrieves a task result field (such as 'answer' or 'error').
    """
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """
    Retrieves the current task status.

    Parameters:
    - task_id: The ID of the task.

    Returns:
    - str: The status name; returns an empty string if it has not been set.
    """
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """
    Retrieves the list of completed nodes (formatted as English display names).
    """
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    return [_to_en(n) for n in done]


def get_running_task_list(task_id: str) -> List[str]:
    """
    Retrieves the list of currently running nodes (formatted as English display names).
    """
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    return [_to_en(n) for n in running]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """
    Updates the overall task status.

    Parameters:
    - task_id: The ID of the task.
    - status_name: The name of the status (string).
    """
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    """Pushes the current task progress state to the SSE stream session."""
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str):
    """Purges all in-memory tracking states for the specified task_id."""
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)