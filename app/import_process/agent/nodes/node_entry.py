import sys
from app.core.logger import logger
from pathlib import Path
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Entry Node (node_entry)
    Naming Purpose: Serves as the Entry Point of the graph, responsible for intercepting external payloads and routing downstream workflows.
    State Contract: local_file_path [ is_read_md_enabled is_read_pdf_enabled ] md_path pdf_path file_title
    Roadmap / Future Implementations:
       1. Node-Inbound Logging: [Node Name + Core Parameters]
          Record task lifecycle status [Which task has kicked off] -> Push runtime telemetry to the front-end (Event Tracking / Analytics).
       2. Parameter / Payload Verification: (local_file_path Missing -> Early Return to End / local_dir Missing -> Scaffold a temporary directory).
       3. Filetype Parsing & State Mutation: Evaluate local_file_path extensions (MD vs PDF)
          -> Set is_md_read_enabled = True  ||  Set is_pdf_read_enabled = True
          -> Route md_path = local_file_path || Route pdf_path = local_file_path
          -> Parse and bind file_title = Extracted Filename
       4. Node-Outbound Logging: [Node Name + Core Parameters]
          Record task lifecycle status [Which task has finished executing] -> Push runtime telemetry to the front-end (Event Tracking / Analytics).
    """
    
    # 1. Node-Inbound Logging: [Node Name + Core Parameters] Record task execution status (Telemetry push to front-end)
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] execution started! Current State: {state}")
    add_running_task(state['task_id'], function_name)
    # 2. Perform mandatory non-null verification/guard checks on payload parameters
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.error(f"[{function_name}] Verification failed: Missing source input file path. Aborting ingestion lifecycle!!")
        return state
    # 3. Evaluate file extension and execute downstream state mutations
    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith('.pdf'):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.error(f"[{function_name}] Unsupported format (Expected .md or .pdf). Aborting ingestion lifecycle!!")
    # Extract file_title: /xx/xxx/aaaa.pdf -> aaaa
    # Purpose: Serves as a deterministic fallback mechanism in case the LLM fails to extract the core 'item_name' downstream.
    # Comparison note: os.path.basename(local_file_path).split(".")[0] vs Path(local_file_path).stem
    file_title = Path(local_file_path).stem  # Extracts base filename stripped of extensions (.name, .suffix)
    state['file_title'] = file_title
    # 4. Node-Outbound Logging: [Node Name + Core Parameters] Record task completion status (Telemetry push to front-end)
    logger.info(f">>> [{function_name}] execution finished! Updated State: {state}")
    add_done_task(state['task_id'], function_name)
    return state