import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# Third-party libraries
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# Internal tools/configs/clients
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # Compiled LangGraph workflow instance
from app.core.logger import logger  # Project unified logger tool

# Initialize FastAPI application instance
# Title and description will be displayed in the Swagger UI (http://ip:port/docs)
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD -> Parsing -> Splitting -> Embedding -> Milvus/KG Storage)"
)
 
# CORS middleware configuration: Resolve cross-origin resource sharing limitations for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all frontend domains to access (recommended to specify explicit domains in production)
    allow_credentials=True,  # Allow credentials like Cookies
    allow_methods=["*"],  # Allow all HTTP methods (GET/POST/PUT/DELETE, etc.)
    allow_headers=["*"],  # Allow all request headers
)

# 8080/import. -> import.html
@app.get("/import", response_class=FileResponse)
async def get_import_page():
    """Returns the file import frontend page: import.html"""
    # Build the absolute path of the HTML file based on the project root directory
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"
    # Log the page access path to facilitate troubleshooting if the file is missing
    logger.info(f"Frontend page accessed. Absolute file path: {html_abs_path}")
    # Validate if the file exists; raise a 404 HTTP Exception if it does not
    if not os.path.exists(html_abs_path):
        logger.error(f"Frontend page file does not exist at path: {html_abs_path}")
        raise HTTPException(status_code=404, detail="import.html page not found")
    # Return HTML file as a FileResponse so the browser renders it automatically
    return FileResponse(
        path=html_abs_path,
        media_type='text/html'
    )
    
# --------------------------
# Background Task: LangGraph Pipeline Execution
# Runs independently of the main request thread, triggered by BackgroundTasks to avoid blocking API responses
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    LangGraph full workflow background task execution.
    Core process: Initialize state -> Stream graph nodes -> Real-time status update -> Exception handling.
    Global task status updates: pending -> processing -> completed/failed.
    Node progress updates: Append completed node name to `done_list` for frontend polling.
 
    :param task_id: Globally unique task ID associated with a single file workflow.
    :param local_dir: Local storage directory for this task (contains temporary files/parsed results).
    :param local_file_path: Absolute local path of the uploaded file.
    """
    try:
        # 1. Update the task global status to: processing
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] Starting LangGraph execution. Local file path: {local_file_path}")
        # 2. Initialize LangGraph state: Load default state + inject core parameters of the current task
        init_state = get_default_state()
        init_state["task_id"] = task_id  # Associate task ID
        init_state["local_dir"] = local_dir  # Associate task local directory
        init_state["local_file_path"] = local_file_path  # Associate uploaded file path
        # 3. Stream-execute the LangGraph pipeline (stream mode: retrieve execution results of each node in real-time)
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                # Log completion of each node with Task ID and node name to trace execution order
                logger.info(f"[{task_id}] LangGraph node execution completed: {node_name}")
                # Append completed node to the 'done list' so frontend polling can retrieve progress in real-time
                add_done_task(task_id, node_name)
        # 4. Full workflow execution completed. Update global status to: completed
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph workflow executed successfully. Task completed.")
 
    except Exception as e:
        # 5. Capture any exceptions during execution, update status to: failed, and log detailed error with traceback
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph workflow failed. Exception: {str(e)}", exc_info=True)
 
 
 
 
# --------------------------
# Core API: File Upload Endpoint
# Supports batch multi-file upload. Process: Receive -> Save locally -> Upload to MinIO -> Start background task
# URL: http://localhost:8000/upload (POST request with form-data parameters)
# --------------------------
@app.post("/upload", summary="File Upload Interface", description="Supports batch multi-file upload, automatically triggering the knowledge base import pipeline.")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    File Upload Core Endpoint
    1. Receives multiple uploaded files (primarily PDF/MD).
    2. Saves files locally in date-stratified folders 'output/YYYYMMDD' to prevent namespace conflicts.
    3. Uploads files to MinIO object storage for persistent backup.
    4. Generates a unique Task ID for each file and triggers an independent asynchronous LangGraph background task.
    5. Dynamically updates task progress, allowing the frontend to poll for updates.
 
    :param background_tasks: FastAPI BackgroundTasks object to run LangGraph asynchronously.
    :param files: List of files uploaded by the frontend (using form-data format).
    :return: JSON response containing upload details and generated Task IDs.
    """
    # 1. Construct local storage root directory: project_root/output/YYYYMMDD (stratified by date)
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))
    # Initialize a list of Task IDs to return to the frontend (one Task ID per file)
    task_ids = []
    logger.info('date_based_root_dir..',date_based_root_dir)
    # 2. Iterate through and process each uploaded file (batch processing; each file gets a separate Task ID)
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] Started processing upload file: {file.filename}, type: {file.content_type}")
        # 3. Mark the 'file upload' stage as 'running', accessible via frontend status polling
        add_running_task(task_id, "upload_file")
        # 4. Construct a unique task subdirectory: output/YYYYMMDD/TaskID to avoid duplicate filename conflicts
        task_local_dir = os.path.join(date_based_root_dir, task_id)
        os.makedirs(task_local_dir, exist_ok=True)  # Create if it doesn't exist, skip otherwise
        # Build the absolute local path to save the uploaded file
        local_file_abs_path = os.path.join(task_local_dir, file.filename)
        # 5. Save the uploaded file to the local temporary directory (downstream MinIO and parsing operations depend on this file)
        with open(local_file_abs_path,'wb') as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] File saved locally to path: {local_file_abs_path}")
        # 6. Upload the local file to MinIO object storage for persistent storage
        # Get MinIO directory configuration from environment variables
        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")  # Default fallback: pdf_files
        # Construct object name in MinIO: base_dir/YYYYMMDD/filename (date-stratified, aligning with local structure)
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        try:
            # Get MinIO client instance
            minio_client = get_minio_client()
            if minio_client is None:
                # Raise 500 error if client instantiation fails
                raise HTTPException(status_code=500,
                                    detail="MinIO service connection failed, please check MinIO config")
            # Get target bucket name from environment variables
            minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")  # Default: kb-import-bucket
             # Upload local file to MinIO (will overwrite files with identical paths to ensure the latest version is kept)
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=local_file_abs_path,
                content_type=file.content_type  # Pass original MIME content type
            )
            logger.info(f"[{task_id}] File successfully uploaded to MinIO. Bucket: {minio_bucket_name}, Object: {minio_object_name}")
        except Exception as e:
            # If MinIO upload fails, log a warning but do not break execution (local workflow can still proceed)
            logger.warning(f"[{task_id}] Failed to upload file to MinIO. Proceeding with local workflow processing. Error: {str(e)}", exc_info=True)
        
        # 7. Mark the 'file upload' stage as 'completed'
        add_done_task(task_id, "upload_file")
        # 8. Dispatch the LangGraph workflow to FastAPI BackgroundTasks for asynchronous processing without blocking the API response
        background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path)
        logger.info(f"[{task_id}] LangGraph pipeline added to background tasks successfully.")
     # 9. All files uploaded and background workers started. Return success response along with all Task IDs
    logger.info(f"Multi-file upload processing finished. Handled {len(files)} files. Generated Task IDs: {task_ids}")
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }
    
# --------------------------
# Core API: Task Status Polling Endpoint
# Frontend polls this endpoint to fetch the processing progress and status of an individual task
# URL: http://localhost:8000/status/{task_id} (GET request)
# --------------------------
@app.get("/status/{task_id}", summary="Task Status Query", description="Query individual file processing progress and global state via TaskID")
async def get_task_progress(task_id: str):
    """
    Task status query endpoint.
    The frontend polls this endpoint (e.g., once per second) to retrieve real-time task progress.
    Returned data is sourced entirely from the in-memory task management dictionary (task_utils.py), ensuring high performance with zero I/O.
    :param task_id: Globally unique Task ID (originally returned by the /upload endpoint)
    :return: JSON response containing global task status, completed nodes, and currently running nodes.
    """
    # Construct the task status response body
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # Global task status: pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # List of completed nodes/steps
        "running_list": get_running_task_list(task_id)  # List of currently executing nodes/steps
    }
    # Log the status query request to facilitate tracking of frontend polling frequency
    logger.info(
        f"[{task_id}] Task status query. Current status: {task_status_info['status']}, Completed nodes: {task_status_info['done_list']}")
    return task_status_info
 
 
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
 
 
        