import os
import sys
import time
import requests
import zipfile
import shutil
from pathlib import Path
 
# Internal Project Libraries
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config
from app.core.logger import logger  # Unified Logging Utility
 
# MinerU Configuration (Cached Context Layouts)
MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_key

def step_1_validate_path(state):
    """
    Executes deep validation and path sanitization. 
    If pdf_path is invalid or non-existent, explicit exceptions are raised immediately!
    If local_dir is missing, assigns a fallback default path.

    :param state: Current graph state channel payload.
    :return: Tuple[Path, Path] -> Validated Path objects for both the source PDF and output directory.
    """
    logger.debug(f">>> [step_1_validate_paths] Initiating file format validation for PDF-to-Markdown pipeline!!")
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']
    
    if not pdf_path:
        logger.error('step_1_validate_paths check failed: Missing source input file path. Aborting execution!!')
        raise ValueError("step_1_validate_paths check failed: Missing source input file path. Aborting execution!!")
    if not local_dir:
        # local_dir = str(PROJECT_ROOT/"output")
        logger.info(f"step_1_validate_paths detected empty local_dir, assigning fallback storage target: {local_dir}!")
    
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    
    if not pdf_path_obj.exists():
        logger.error(f"[step_1_validate_paths] Filesystem error: Source pdf_path does not exist. Verify input argument layout!!")
        raise FileNotFoundError(f"[step_1_validate_paths] Filesystem error: Source pdf_path does not exist. Verify input argument layout!!")
    
    if not local_dir_obj.exists():
        logger.error(f"[step_1_validate_paths] Output directory does not exist. Orchestrating auto-scaffolding for target paths!!!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj


def step_2_upload_and_poll(pdf_path_obj: Path, output_dir_obj: Path) -> str:
    """
    Step 2: Upload PDF source to MinerU cluster and asynchronously poll processing task lifecycles.
    Core Workflow: Configuration sanitization -> Request Presigned URL -> Binary chunk upload (with fallback) -> Task polling (until finish/fail/timeout).
    Parameters: pdf_path_obj - Validated source PDF Path; output_dir_obj - Workspace output target Path.
    Returns: A complete archival remote bundle address (full_zip_url).
    Exceptions: ValueError (Missing config blocks), RuntimeError (Network/API business aborts), TimeoutError (Lifecycle expiration).
    """
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        raise ValueError("MinerU credentials missing: Configure MINERU_BASE_URL and MINERU_API_TOKEN inside your local .env configuration.")
    logger.info(f"[Config Validation] MinerU infrastructure bindings parsed successfully. Ingesting file payload: {pdf_path_obj.name}")
    # Construct HTTP standard headers with Bearer Authentication tokens
    request_headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}"
    }
    # 1. Dispatch batch processing requests to retrieve Presigned URLs and task batch_ids
    url_get_upload = f"{MINERU_BASE_URL}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm" # Officially recommended foundational vision-language extraction engine
    }
    logger.debug(f"[Fetch Upload Link] Querying downstream routing endpoint: {url_get_upload}. Payload layout: {req_data}")
    resp = requests.post(url=url_get_upload, headers=request_headers, json=req_data, timeout=30)
    # Response Assertion: Validate HTTP protocol wrapper statuses before parsing internal API business codes
    resp_data = resp.json()
    if resp.status_code != 200 or resp_data['code'] != 0:
       raise RuntimeError(f"[Fetch Upload Link] Network layout failure. HTTP status: {resp.status_code}. Telemetry context: {resp.text}") 
    # Isolate key payload tokens: Binary stream upload path and unique orchestration identifier
    signed_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]
    logger.info(f"[Fetch Upload Link] Successfully registered pipeline batch_id: {batch_id}. Cloud target presigned route built.")
    # 2. Extract local binary data layout to stream up to processing clusters
    logger.info(f"[Payload Upload] Buffering system binary streams for file: {pdf_path_obj.name}")
    with open(pdf_path_obj, "rb") as f:
        file_data = f.read()
    # Initialize a clean request session to reuse persistent TCP sockets and bypass proxy signature mutations
    upload_session = requests.Session()
    upload_session.trust_env = False # proxy is not allowed
    
    try:
        put_resp = upload_session.put(url=signed_url, data=file_data, timeout=500)
        if put_resp.status_code != 200:
            logger.warning(f"[Payload Upload] Ingestion bounce encountered (HTTP status: {put_resp.status_code}). Executing explicit MIME-type override sequence.")
        logger.info(f"[Payload Upload] Ingestion finalized. Source: {pdf_path_obj.name} successfully pushed to remote object stores.")
    except Exception as e:
        raise RuntimeError(f"[Payload Upload] Structural network anomalies aborted upload sequence: {str(e)}")
    finally:
        upload_session.close()
        
    # 3. Poll processing status tracks using the validated batch_id until resolution or boundary exhaustion
    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    start_time = time.time()
    timeout_seconds = 600  # 10-minute lifecycle threshold (optimized to absorb intensive docs up to 600 pages)
    poll_interval = 3      # 3-second throttle interval to match background ingestion speeds without server throttling
    logger.info(f"[Lifecycle Polling] Monitoring execution tracker spans. batch_id: {batch_id}. Expiration roof: {timeout_seconds}s")
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            raise TimeoutError(f"[Lifecycle Polling] Boundary exhaustion encountered! Ingestion exceeded {int(timeout_seconds)}s limit for batch_id: {batch_id}")
        # Fire structural polling requests with light 10s socket drop roofs, catch network breaks gracefully
        try:
            poll_resp = requests.get(url=poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            logger.warning(f"[Lifecycle Polling] Transient network break. Re-attempting sequence in {poll_interval}s: {str(e)}")
            time.sleep(poll_interval)
            continue
        
        if poll_resp.status_code != 200:
            if 500 <= poll_resp.status_code < 600:
                logger.warning(f"[Lifecycle Polling] Downstream parsing clusters under high load (HTTP status: {poll_resp.status_code}). Backing off for {poll_interval}s.")
                time.sleep(poll_interval)
                continue
            else:
               raise RuntimeError(f"[Lifecycle Polling] HTTP channel verification broken. Status: {poll_resp.status_code}. Telemetry context: {poll_resp.text}") 
        poll_data = poll_resp.json()
        if poll_data["code"] != 0:
            raise RuntimeError(f"[Lifecycle Polling] API engine error payload caught: {poll_data}")
        extract_results = poll_data["data"]["extract_result"]
        if not extract_results:
            logger.debug(f"[Lifecycle Polling] Resource frame empty. Total wait time: {int(elapsed_time)}s. Holding position...")
            time.sleep(poll_interval)
            continue
            
        # Extract operational tracks and route state decisions
        result_item = extract_results[0]
        state_status = result_item["state"]
        # Branch 1: Successful processing cycle. Pull the remote bundle download link.
        if state_status == "done":
            logger.info(f"[Lifecycle Polling] Pipeline extraction finalized! Ingestion lifecycle duration: {int(elapsed_time)}s. batch_id: {batch_id}")
            full_zip_url = result_item.get("full_zip_url")
            if not full_zip_url:
                raise RuntimeError(f"[Lifecycle Polling] State returned as 'done' but resolution asset path full_zip_url is missing. batch_id: {batch_id}")
            logger.info(f"[Lifecycle Polling] Isolated remote bundle link localized: {full_zip_url}...")
            return full_zip_url
            
        # Branch 2: Execution failure. Isolate upstream crash telemetry and kill execution loops fast.
        elif state_status == "failed":
            err_msg = result_item.get("err_msg", "An undisclosed upstream pipeline exception occurred.")
            raise RuntimeError(f"[Lifecycle Polling] Remote parsing sequence collapsed. batch_id: {batch_id}. Error trace context: {err_msg}")
        # Branch 3: Transition states (processing, analyzing). Stream live runtime duration overlays to CLI.
        else:
            logger.debug(
                f"[Lifecycle Polling] Processing active (Duration: {int(elapsed_time)}s). Status track: {state_status} | Throttling: {poll_interval}s",
                end="\r"
            )
            time.sleep(poll_interval)


def step_3_download_and_extract(zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
    """
    Step 3: Download remote archive bundles, clear collision environments, extract assets, and surface Markdown layers.
    Core Workflow: Asset pull -> Purge dirty environments and extract zip structures -> Search markdown footprints -> Apply strict unified file naming conventions.
    Parameters: zip_url - Download URI; output_dir_obj - Target extraction workspace Path; pdf_stem - Original clean base filename.
    Returns: A verified, concrete absolute filesystem path string to the targeted Markdown payload.
    Exceptions: RuntimeError (Transport issues), FileNotFoundError (No markdown structures recovered).
    """
    logger.info(f"===== Initiating Extraction Processing Spans for Document Identity: [{pdf_stem}] =====")
 
    # 1. Pull processed bundle asset streams, wrapped inside a robust 120s buffer to absorb massive datasets safely
    logger.info(f"[Ingestion 1/4] Accessing remote archive bundle, URL: {zip_url}...")
    resp = requests.get(zip_url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"[Ingestion 1/4] Failed to download archive bundle. Transport verification broken with HTTP status: {resp.status_code}")
    # Generate unique asset filename tokens to isolate file interactions
    zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
    with open(zip_save_path, 'wb') as f:
        f.write(resp.content)
    logger.info(f"[Ingestion 1/4] Bundle download successful. Cached filesystem blueprint: {zip_save_path}")
    # 2. Enforce clean folder workspaces and extract payloads (Guards against cache contamination across execution nodes)
    logger.info(f"[Ingestion 2/4] Initializing archive decompression sequence...")
    extract_target_dir = output_dir_obj / pdf_stem
    # Clean up dirty artifacts from prior runtime crashes without throwing cascading faults
    if extract_target_dir.exists():
        try:
            # Recursively delete the entire directory tree, including the directory itself, all subdirectories, and files.
            shutil.rmtree(extract_target_dir)
            logger.info(f"[Ingestion 2/4] Flushed legacy filesystem tracks from dirty worker nodes: {extract_target_dir}")
        except Exception as e:
            logger.warning(f"[Ingestion 2/4] Non-critical error while wiping legacy tracks. Continuing decompression fallback: {str(e)}")
 
    # Re-scaffold pristine extraction environments
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    
    # Decompress binary payloads while maintaining structural relative directory indices
    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_obj:
        zip_file_obj.extractall(extract_target_dir)
    logger.info(f"[Ingestion 2/4] Bundle decompression finalized. Target workspace environment: {extract_target_dir}")
    # 3. Execute deep recursive sweeps across extraction outputs to discover generated Markdown layouts
    logger.info(f"[Ingestion 3/4] Searching extraction output files for Markdown footprints...")
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(f"[Ingestion 3/4] Structural validation failed: No .md layouts found inside directory tree: {extract_target_dir}")
    logger.info(f"[Ingestion 3/4] Isolated {len(md_file_list)} Markdown structure variants. Evaluating prioritised structural matches.")
    # 4. Route document identity assignments through explicit preference layers (Strict Matching -> Default Output -> Fallback Head)
    target_md_file = None 
    # Priority Track 1: Locate files matching the original source PDF base name tokens exactly
    for md_file in md_file_list:
        if md_file.stem == pdf_stem:
            target_md_file = md_file
            logger.info(f"[Ingestion 4/4] Priority 1 Match verified: Exact name intersection with source PDF identity found -> {target_md_file.name}")
            break
            
    # Priority Track 2: Fall back to default structural artifacts generated by MinerU pipelines (full.md)
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                logger.info(f"[Ingestion 4/4] Priority 2 Match verified: Found pipeline standard document layout structure -> {target_md_file.name}")
                break
    # Priority Track 3: Graceful Failback Guard. Bind the first index element to preserve operational loop health.
    if not target_md_file:
        target_md_file = md_file_list[0]
        logger.info(f"[Ingestion 4/4] Preference matches missed. Executing graceful fallback guard to grab leading index file -> {target_md_file.name}")
    # Normalize File Naming Layouts: Align filename identities with source PDF tokens to ease index linking downstream
    if target_md_file.stem != pdf_stem:
        logger.info(f"[Ingestion 4/4] Re-aligning document signatures to enforce unified naming rules: {pdf_stem}.md")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        try:
            # Commit atomic renaming changes directly to disk allocations
            target_md_file.rename(new_md_path)
            # Update working pointers
            target_md_file = new_md_path
            logger.info(f"[Ingestion 4/4] Document signature normalization successful. Output: {pdf_stem}.md")
        except OSError as e:
            logger.warning(f"[Ingestion 4/4] Failed to normalize layout signature. Operating on legacy filename patterns to avoid pipeline halts: {str(e)}")
    # Cast to localized absolute path strings to guarantee system tool compatibility down-stream
    final_md_path = str(target_md_file.absolute())
    logger.info(f"===== Extraction Processing Lifecycle Finalized for [{pdf_stem}]. Target Path: {final_md_path} =====")
    return final_md_path   
        
        
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    Node: PDF to Markdown Transformer (node_pdf_to_md)
    Naming Purpose: The core mission is translating unstructured PDF layouts into structured Markdown documents.
    Roadmap / Future Implementations:
        1. Node Inbound Telemetry: Bind runtime logging parameters and status updates.
        2. Parameter Sanitization: Set local_dir defaults | Map shallow payload checks to deep filesystem validation.
        3. Document Parsing (MinerU Client): Dispatches local_file_path to upstream parsing clusters, returning an archival tracking asset (.zip URL).
        4. Ingestion / Asset Extraction: Downloads and unpacks archive clusters into the mapped local_dir target.
        5. State Mutation: Binds the extracted md_path target and buffers text streams directly into the md_content channel.
        6. Node Outbound Telemetry: Clean up block footprints, persist telemetry logs, and close trace spans.
        
        Fault Tolerance Layout: Enclosed inside deterministic try-except guard scopes.
    """
    
    # 1. Node Inbound status
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>>[{function_name}]execution started! Current State: {state}")
    add_running_task(state['task_id'], function_name)
    
    try:
        # 2. Parameter Sanitization: Map shallow payload checks to deep filesystem validation
        # Inputs: state -> local_file_path | local_dir
        # Outputs: Validated Path objects for target source files and staging environments
        pdf_path_obj, local_dir_obj = step_1_validate_path(state)
        
        # 3. Document Parsing (MinerU Client): Dispatches data to parsing microservices and monitors async extraction status
        # Inputs: Target source filepath. Outputs: Downstream bundle layout location (.zip remote URI)
        zip_url = step_2_upload_and_poll(pdf_path_obj,local_dir_obj)
        
        # 4. Ingestion / Asset Extraction: Downloads and extracts asset streams into target environments
        # Inputs: 1. Remote download link 2. local_dir_obj extract target 3. Baseline filename token (e.g., "manual_v1")
        # Outputs: Fully materialised, concrete filesystem path to the extracted Markdown layout
        md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
        
        # 5. State Mutation: Update the graph state payload with newly compiled artifacts
        state['md_path'] = md_path
        state['local_dir'] = str(local_dir_obj)
        
        # Open I/O stream buffers to pull processed markdown text layers into the state payload
        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content'] = f.read()
    except Exception as e:
       # Fault Tolerance Block: Log lifecycle failures and crash the workflow state engine
        logger.error(f">>> [{function_name}] Critical lifecycle anomaly during MinerU extraction sequence: {e}")
        raise
    finally:
        # 6. Node Outbound Telemetry: Update trace statuses and push analytics spans back to front-end UI 
        logger.info(f">>> [{function_name}] execution finished! Updated State: {state}")
        add_done_task(state['task_id'], function_name)
    return state

if __name__ == "__main__":

    # unit test: node_pdf_to_md
    logger.info("===== Start node_pdf_to_md node unit test =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"Root directory:{PROJECT_ROOT}")

    test_pdf_name = os.path.join("docs", "aag-cisco-umbrella.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # initialize state
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== End of node_pdf_to_md node unit test =====")