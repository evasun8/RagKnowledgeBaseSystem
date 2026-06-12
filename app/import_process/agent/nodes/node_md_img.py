import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque
 
# MinIO Related Dependencies
from minio import Minio
from minio.deleteobjects import DeleteObject
 
# [Core Refactoring 1: Remove native OpenAI, import LangChain utilities and multimodal message modules]
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task
# LLM Client Utility (Core reuse, replacing native OpenAI calls)
from app.lm.lm_utils import get_llm_client
# LangChain Multimodal Dependencies (Message construction + Exception capturing)
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# Project Configurations
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# Project Logging Utility (Unified usage)
from app.core.logger import logger
# API Access Rate Limiting Utility
from app.utils.rate_limit_utils import apply_api_rate_limit
# Prompt Loading Utility
from app.core.load_prompt import load_prompt
 
# Set of image extensions supported by MinIO (lowercase suffixes for unified matching standard)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# Step 1: Initialize core MD data, retrieving content, file path, and images directory path
def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    Extract and initialize the core data required for MD processing from the global state.
    :param state: Global graph state object for the import pipeline.
    :return: A tuple of (MD file content, MD file Path object, Images directory Path object).
    :raise FileNotFoundError: Raised when no valid MD file path is found in the state.
    """
    md_file_path = state["md_path"]
    if not md_file_path:
        raise FileNotFoundError(f"No valid MD file path found in the global state: {state['md_path']}")
    path_obj = Path(md_file_path)
    # Prioritize using existing MD content in state; otherwise, read from file
    if not state["md_content"]:
        with open(path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        logger.debug(f"MD content read from file completed. File size: {len(md_content)} characters")
    else:
        md_content = state["md_content"]
        logger.debug(f"MD content retrieved from global state completed. Content size: {len(md_content)} characters")
 
    # The images directory is fixed as the "images" directory at the same level as the MD file
    images_dir = path_obj.parent / "images"
    return md_content, path_obj, images_dir

def is_supported_image(filename: str) -> bool:
    """
    Determines if a file is an image format supported by MinIO (suffix is case-insensitive).
    :param filename: Filename (including extension).
    :return: True if supported, False otherwise.
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def find_image_in_md(md_content: str, image_filename: str, context_len: int = 100) -> List[Tuple[str, str]]:
    """
    Find all reference positions of a specific image within the MD content and return the context text for each position.
    :param md_content: Complete content of the MD file.
    :param image_filename: Image filename (including extension).
    :param context_len: Bounding context truncation length, defaulting to 100 characters before and after.
    :return: A list of contexts, where each element is a (pre_text, post_text) tuple. Returns an empty list if no match is found.
    """
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
    results = []
    for m in pattern.finditer(md_content):
        start, end = m.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        # Print image context for debugging convenience
        logger.debug(f"Image [{image_filename}] matched a reference. Pre-text: {pre_text.strip()}")
        logger.debug(f"Image [{image_filename}] matched a reference. Post-text: {post_text.strip()}")
        results.append((pre_text, post_text))
    if not results:
        logger.debug(f"No reference found for image [{image_filename}] in MD content")
    return results
    
# Step 2: Scan the images directory and filter supported format images actually referenced in the MD file
def step_2_scan_images(md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    Scan the images directory, filter out images that match "supported format + actually referenced in MD", and assemble processing metadata.
    :param md_content: Complete content of the MD file.
    :param images_dir: Images directory Path object.
    :return: A list of target images to process, where each element is a (image_filename, absolute_image_path, image_context) tuple.
    """
    targets = []
    # Traverse all files in the images directory
    for image_file in os.listdir(images_dir):
        # Filter out unsupported image formats
        if not is_supported_image(image_file):
            logger.debug(f"Image format not supported, skipping: {image_file}")
            continue
        # Assemble the full local path of the image
        img_path = str(images_dir / image_file) 
        # Find the reference context of the image within the MD content
        context_list = find_image_in_md(md_content, image_file)
        # Filter out images not referenced in the MD file
        if not context_list:
            logger.warning(f"Image not referenced in MD, skipping processing: {image_file}")
            continue
        # Assemble metadata for the image to be processed, capturing the leading matched context index
        targets.append((image_file, img_path, context_list[0]))
        logger.info(f"Image added to processing queue: {image_file}")
    logger.info(f"Image scanning completed. Total images selected for processing: {len(targets)}")
    return targets

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode a local image file into a Base64 string (for multimodal large model ingestion).
    :param image_path: Absolute local path of the image.
    :return: The Base64 encoded string of the image (UTF-8 decoded).
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"Image Base64 encoding completed for file: {image_path}, encoded length: {len(base64_str)}")
    return base64_str

def summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
    """
    Invoke a multimodal vision model to generate an image content summary (adapted for LangChain utilities, reusing the unified project LLM client).
    The generated summary is used as the Markdown image alt-text title, strictly limited to a description within 50 characters.
    :param image_path: Absolute local path of the image.
    :param root_folder: Document folder name / stem, providing context for the vision model.
    :param image_content: The image's context tuple within the MD file, formatted as (pre_text, post_text).
    :return: The image content summary (returns a default string "image description" upon exception).
    """
    # Encode image to Base64 to adapt to multimodal model input requirements
    base64_image = encode_image_to_base64(image_path)
    try:
        # 1. Retrieve the project unified LLM client (automatically cached, passing the multimodal model name)
        lvm_client = get_llm_client(model=lm_config.lv_model)
 
        # Load and render the prompt structure (Core: passing all variables corresponding to placeholders)
        prompt_text = load_prompt(
            name="image_summary",  # Prompt filename (without .prompt extension)
            root_folder=root_folder,  # Maps to {root_folder}
            image_content=image_content  # Maps to {image_content[0]} and {image_content[1]}
        )
        # 2. Construct the standard LangChain multimodal HumanMessage (compatible with Qwen/OpenAI vision models)
        messages = [
            HumanMessage(
                content=[
                    # Text prompt: carries context layers and enforces summary boundary constraints
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    # Multimodal core: Base64 encoded image binary streams
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]
        # 3. Standard LangChain invocation: invoke method (the utility class encapsulates timeout/retry params)
        response = lvm_client.invoke(messages)
        # 4. Parse response payload (LangChain returns the content field uniformly, no need for multi-layer parsing)
        summary = response.content.strip().replace("\n", "")
        logger.info(f"Image summary generated successfully for file: {image_path}, Summary: {summary}")
        return summary
    except LangChainException as e:
        logger.error(f"Image summary generation failed (LangChain framework exception) for file: {image_path}, Error: {str(e)}")
        return "Image Description"
    except Exception as e:
        logger.error(f"Image summary generation failed (System exception) for file: {image_path}, Error: {str(e)}")
        return "Image Description"
    
    
def step_3_generate_summaries(doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                             requests_per_minute: int = 9) -> Dict[str, str]:
    """
    Step 3: Batch generate content summaries for target images, with API rate limiting to prevent triggering model throttling.
    :param doc_stem: Document filename (without extension), acting as the prompt context for the model.
    :param targets: List of target images to process, elements formatted as (image_filename, absolute_image_path, image_context).
    :param requests_per_minute: Maximum API requests per minute, defaulting to 9 times (adjust based on model rate limits).
    :return: A dictionary of image summaries, Key: image filename, Value: image content summary string.
    """
    summaries = {}
    request_times = deque()
    for img_file, image_path, context in targets:
        # Directly invoke the abstracted public utility method with identical parameters to the original logic
        apply_api_rate_limit(request_times, requests_per_minute, window_seconds=60)
        logger.debug(f"Starting image summary generation for: {image_path}")
        summaries[img_file] = summarize_image(image_path, root_folder=doc_stem, image_content=context)
 
    logger.info(f"Batch image summary generation completed. Processed {len(summaries)} images.")
    return summaries

def clean_minio_directory(minio_client: Minio, prefix: str) -> None:
    """
    Idempotently clean up all stale files under the specified MinIO directory path to prevent name collision and garbage accumulation.
    Idempotency: Multiple invocations yield identical results; does not throw errors when no files exist.
    :param minio_client: Initialized MinIO client object instance.
    :param prefix: MinIO directory prefix (the directory path to clean up).
    """
    try:
        # List all objects under the specified prefix (recursively traversing subdirectories)
        objects_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=prefix,
            recursive=True
        )
        # Construct the list of deletion objects
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
        if delete_list:
            logger.info(f"Starting MinIO stale file cleanup. Files to delete: {len(delete_list)}, Directory path: {prefix}")
            # Batch remove objects
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            # Traverse deletion error responses to log anomalies
            for error in errors:
                logger.error(f"MinIO file deletion failed: {error}")
        else:
            logger.debug(f"No stale files found in MinIO directory; cleanup skipped: {prefix}")
    except Exception as e:
        logger.error(f"MinIO directory cleanup failed for path: {prefix}, Error metadata: {str(e)}")
    
def upload_images_batch(minio_client: Minio, upload_dir: str, targets: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[
    str, str]:
    """
    Batch upload staging images up to MinIO and return the mapping relationship between image filenames and access URLs.
    :param minio_client: Initialized MinIO client object instance.
    :param upload_dir: MinIO target upload root directory.
    :param targets: List of target images to process, elements formatted as (image_filename, absolute_image_path, image_context).
    :return: A dictionary of image URLs, Key: image filename, Value: resolved MinIO public access URL endpoint.
    """
    urls = {}
    for img_file, img_path, _ in targets:
       # Construct the MinIO target object name
        object_name = f"{upload_dir}/{img_file}"
        logger.debug(f"MinIO target object name construction completed: {object_name}")
        logger.debug(f"MinIO target object name construction completed: {object_name}")
        # Upload a single image asset and retrieve its public URL endpoint
        """
        := is the Walrus Operator introduced in Python 3.8+. Its core purpose is "expression-bounded assignment + condition evaluation integration":
        It completes variable assignment and evaluates the result within a single statement, replacing traditional multi-line assignments to keep logic concise.
        """
        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url
    logger.info(f"Batch image upload sequence completed. Successfully hosted {len(urls)}/{len(targets)} images.")
    return urls

def upload_to_minio(minio_client: Minio, local_path: str, object_name: str) -> str | None:
    """
    Upload a single local image asset to MinIO Object Storage and return its publicly accessible URL endpoint.
    :param minio_client: Initialized MinIO client object instance.
    :param local_path: Absolute local path of the target image file.
    :param object_name: Destination object name structure in MinIO (including directory structures).
    :return: The public MinIO access URL string (returns None if upload sequence fails).
    """
    try:
        logger.info(f"Starting image upload to MinIO: Local Path={local_path}, MinIO Object Name={object_name}")
        # Upload the local file to MinIO (fput_object: streams file parts from disk, optimal for file scaling)
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,  # MinIO storage bucket name (fetched from config structures)
            object_name=object_name,  # Target object naming layout in MinIO
            file_path=local_path,  # Local file source path
            # Automatically deduce image Content-Type headers (e.g., image/png, image/jpeg)
            # Input parameter: File path string (can contain directories, e.g., /a/b/test.jpg, demo.tar.gz);
            # Return value: Tuple (root, ext), where:
            # root: File base name (including directory, stripping the final suffix extension);
            # ext: File extension (leading with a dot ., capturing only the final extension part, e.g., .jpg, .gz. Returns empty string "" if none);
            # Critical Rule: Only flags the **last .** as the suffix delimiter. Multi-suffix files only split the final tail (e.g., test.tar.gz splits into ("test.tar", ".gz")).
            content_type=f"image/{os.path.splitext(local_path)[1][1:]}"
        )
        
        # Handle path special characters to prevent URL parsing breakdowns
        # Scenario: If object_name is "images\\logo.png", it morphs into "images%5Clogo.png" post-replacement.
        # This string configuration is URL-legal and recognized properly by target server/browser layers.
        # Upon receipt of %5C, MinIO expands it back to \\ natively to preserve object name integrity.
        # Downstream URL navigation unrolls %5C safely without injecting broken path routes.
        object_name = object_name.replace("\\", "%5C")
        # Evaluate HTTP vs HTTPS protocols based on security configuration variables
        protocol = "https" if minio_config.minio_secure else "http"
        # Construct the base target URL for MinIO
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
        # Splice the fully resolved public image access URL endpoint. base_url carries a trailing / delimiter.
        img_url = f"{base_url}{object_name}"
        logger.info(f"Image uploaded successfully. Public access URL: {img_url}")
        return img_url
    except Exception as e:
        logger.error(f"Image upload sequence to MinIO failed for local target: {local_path}, Error description: {str(e)}")
        return None

def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    Merge the image summaries dictionary and URLs dictionary, filtering out images that failed to upload or lack a valid URL.
    :param summaries: Image summaries dictionary, Key: image filename, Value: content summary description.
    :param urls: Image URLs dictionary, Key: image filename, Value: MinIO access URL endpoint.
    :return: Merged image information dictionary, Key: image filename, Value: (summary, URL) data tuple.
    """
    image_info = {}
    # Iterate through summaries, preserving only instances that successfully secured a remote hosting URL string
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"Image summary and URL aggregation completed. Valid image metadata rows count: {len(image_info)}")
    return image_info

def step_4_upload_and_replace(minio_client: Minio, doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                             summaries: Dict[str, str], md_content: str) -> str:
    """
    Step 4: Pipeline Hub - Upload images to MinIO + Merge summaries & URLs + Transform MD image references.
    Complete Lifecycle Sequence: Evict stale target MinIO paths → Upload new image batches → Merge metadata maps → Transform Markdown layouts.
    :param minio_client: Initialized MinIO client object instance.
    :param doc_stem: Document filename base (without extension), acting as the namespace isolation folder directory name in MinIO.
    :param targets: List of target images to process, elements formatted as (image_filename, absolute_image_path, image_context).
    :param summaries: Image summaries dictionary, Key: image filename, Value: content summary.
    :param md_content: Source original markdown text layer.
    :return: New updated markdown text content with remote image pointers.
    """
    # Construct MinIO upload directory: Configured Base Route + Document main name (spaces removed to prevent routing bugs)
    minio_img_dir = minio_config.minio_img_dir
    upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")
    # Action 1: Evict stale target folders matching this document namespace to preserve pipeline idempotency
    clean_minio_directory(minio_client, upload_dir)
    # Action 2: Batch upload image streams to MinIO to get URL asset maps
    urls = upload_images_batch(minio_client, upload_dir, targets)
    # Action 3: Merge summaries and URLs maps, dropping records that crashed out during file transfer spans
    # Dict[str, Tuple[str, str]] -> Key: image filename, Value: (summary, URL) tuple
    image_info = merge_summary_and_url(summaries, urls)
    # Action 4: Substitute local file system links inside markdown documents with resolved cloud storage URL markers
    if image_info:
        md_content = process_md_file(md_content, image_info)
 
    return md_content

def process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """
    Core Transformer: Swivel local markdown image paths to target remote hosted MinIO URL endpoints.
    Transformation Rule: ![original_desc](local_path) → ![image_summary](MinIO_access_URL)
    :param md_content: Source original markdown text block layers.
    :param image_info: Merged image information dictionary mapping image filenames to (summary, URL) tuples.
    :return: Transformed markdown content block string layers.
    """
    for img_filename, (summary, new_url) in image_info.items():
        # Regex matches markdown image tags case-insensitively, allowing path flexibility
        # Pattern structure: ![any_description](any_path + image_filename + any_suffix)
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",
            re.IGNORECASE
        )
        # Substitute matching structures: embed the fresh summary into description blocks and new URLs into paths
        # Note: If your summary and new_url are deterministic text blocks (free of backslashes), string vs lambda strategies behave identically.
        # Defensively coding to safeguard from special characters parsing failures down the road, lambda structures are the most secure choice.
        # md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
        md_content = pattern.sub(f"![{summary}]({new_url})", md_content)
        logger.debug(f"Completed MD image reference transformation: {img_filename} → {new_url}")
 
    logger.info(f"Markdown document image references transformation completed. Injected {len(image_info)} modifications.")
    logger.debug(f"Transformed MD snippet preview: {md_content[:500]}..." if len(md_content) > 500 else f"Transformed MD full layout: {md_content}")
    return md_content 

def step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """
    Step 5: Persist the transformed markdown contents into a fresh distinct file to avoid data mutation or loss.
    File Naming Schema: Original Filename Base + "_new.md" suffix (e.g., test.md → test_new.md).
    :param origin_md_path: Original markdown absolute file path string.
    :param md_content: Transformed markdown text block to be written to disk.
    :return: The absolute destination file path string of the newly generated markdown document.
    """
    # Construct destination file routes: append the "_new.md" identifier to the parsed file root base
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
    # Persist the new markdown layout to disk (Overwrites existing instances if name matches)
    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)
 
    logger.info(f"Processed markdown file persisted successfully. Target destination path: {new_md_file_name}")
    return new_md_file_name

    
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    Core Pipeline Node: Markdown Embedded Image Processor - Orchestrates image lifetimes via a 5-step transaction pattern.
    Core Lifecycle Phases:
    1. Initialize data payloads extracting MD contents, absolute text paths, and staging image asset folder routes.
    2. Scan matching image assets directories, filtering valid formats referenced within document text.
    3. Dispatch Base64 encoded payload streams to multimodal vision models to gather context-aware text summaries.
    4. Stream local assets up to public MinIO Object Storage buckets, swiveling legacy links to cloud endpoints with alt-text titles.
    5. Persist transformed text models to a distinct backup markdown file without mutating source records, and update pipeline tracking states.
    :param state: Global graph tracking state pipeline channel carrying task_id, md_path, and md_content tokens.
    :return: Mutated graph state tracking dictionary containing updated file system targets and text channels.
    """
    # Track task state registration inside monitoring loops for telemetry traceability
    add_running_task(state["task_id"], sys._getframe().f_code.co_name)
    # Step 1: Initialize data fields, extracting core MD metadata parameters
    md_content, path_obj, images_dir = step_1_get_content(state)
    # Short-circuit processing flows early if no valid image folder is detected on disk
    if not images_dir.exists():
        logger.info(f"Target image directory does not exist on disk, skipping image processing: {images_dir.absolute()}")
        return state
    minio_client = get_minio_client()
    if not minio_client:
        logger.warning("MinIO client initialization sequence failed. Aborting image processing lifecycle hooks.")
        return state
    # Step 2: Scan asset directories filtering files referenced inside markdown matching target extensions
    # Struct layout mapping: (image_file, img_path, context_list[0])
    targets = step_2_scan_images(md_content, images_dir)
    if not targets:
        logger.info("No matching image assets referenced inside markdown content. Skipping downstream hooks.")
        return state
 
    # Step 3: Invoke multimodal models to gather image summary text blocks (Fixing a variable mismatch from legacy codebase: passing the document base stem instead of raw text layers)
    summaries = step_3_generate_summaries(path_obj.stem, targets)
    
    # Step 4: Stream assets up to target bucket regions and transform local reference tags inside file strings
    new_md_content = step_4_upload_and_replace(minio_client, path_obj.stem, targets, summaries, md_content)
    state["md_content"] = new_md_content
    
    # Step 5: Save processed models into secondary files and sync path variables across graph nodes
    new_md_file_name = step_5_backup_new_md_file(state['md_path'], new_md_content)
    state["md_path"] = new_md_file_name
    logger.info(f"Markdown asset transformation completed safely. Target tracking token: {new_md_file_name}")
 
    return state

# ============ TEST=============
if __name__ == "__main__":
    """Local Test Entrypoint: Executes the complete Markdown image processing workflow for standalone testing."""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"Local Testing Suite initiated - Project Base Root: {PROJECT_ROOT}")
 
    # Target test file routes (Verify file layout paths match before launching runtime execution loops)
    test_md_name = os.path.join(r"output/aag-cisco-umbrella", "aag-cisco-umbrella.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)
 
    # Verify presence of validation files on disk before running test harnesses
    if not os.path.exists(test_md_path):
        logger.error(f"Local Testing Suite aborted - Target file layout missing on disk: {test_md_path}")
        logger.info("Verify path properties or mount your test files inside the output folder path before rerunning the test engine.")
    else:
        # Construct standard mockup state schemas mirroring orchestration pipeline conditions
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("Launching Local Test Pipeline - Markdown Image Processing Workflow")
        # Direct execution of the core workflow node block
        result_state = node_md_img(test_state)
        logger.info(f"Local Testing Suite completed successfully - Resulting runtime state properties: {result_state}")