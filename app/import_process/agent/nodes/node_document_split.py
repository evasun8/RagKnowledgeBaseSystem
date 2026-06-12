import re
import json
import os
import sys
# Unified type annotations to avoid mixing up any/Any
from typing import List, Dict, Any, Tuple
# LangChain text splitters (annotated core purpose for readability)
from langchain_text_splitters.character import RecursiveCharacterTextSplitter 
# Internal project utilities/state/logger imports (keeping original paths)
from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # Unified project logging tool, core replacement for print

# Maximum character length for a single chunk: Triggers secondary splitting if exceeded (tailored for LLM context windows).
DEFAULT_MAX_CONTENT_LENGTH = 2000
# Threshold for merging short chunks: Short chunks sharing the same parent title will be merged to reduce fragmentation.
MIN_CONTENT_LENGTH = 500
    

def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int]:
        """
        [Step 1] Get and Preprocess Input Data
        Function: Extract MD content, file title, and maximum length from the state dictionary, and perform basic standardization.
        :param state: Project state dictionary (ImportGraphState), containing core keys such as `md_content`.
        :return: Standardized MD content, file title, and max length for a single chunk (returns None, None, None if no content is found).
        """     
        # Extract raw MD content from the state
        content = state.get("md_content")
        # Fallback for empty content: If no MD content exists, return directly to terminate subsequent processing.
        if not content:
            logger.warning("No valid MD content found in the state dictionary. Terminating document splitting.")
            return None, None, None
        # Basic standardization: Unify line breaks to avoid downstream processing exceptions caused by differences between Windows/Linux line endings.
        # Raw mixed line breaks: "#  Manual\r\n## Product Overview\n is a scanner\r\n\r\n### Operational Steps"
        # Unified: "#  Manual\n## Product Overview\n is a scanner\n\n### Operational Steps"
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        # Extract file title: Use if available, fallback to "Unknown File" by default
        file_title = state.get("file_title", "Unknown File")
        # Extract maximum chunk length: Use the configuration from state if available, fallback to the global default value otherwise
        max_len = DEFAULT_MAX_CONTENT_LENGTH

        logger.info(f"Step 1: Input data loaded successfully. File Title: {file_title}, Max Chunk Length: {max_len}")
        return content, file_title, max_len
    
def step_2_split_by_titles(content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    [Step 2] Initial Splitting by Markdown Titles (Core: Split hierarchically by #, skip titles inside code blocks)
    LangChain pre-processing concept: Break the whole MD into independent sections by titles, laying the foundation for subsequent fine-grained chunking.
    :param content: Standardized full MD content (string).
    :param file_title: The title of the source file, used to tag section ownership.
    :return: A tuple containing (list of split sections, count of valid titles, total lines of raw text).
    """
    # Regex to match Markdown headings level 1-6 (Core rule, accommodating indentations/standard formats)
    # ^\s*: Allows 0 or more spaces/tabs at the beginning of the line (compatible with indented headings)
    # #{1,6}: Matches 1 to 6 '#' characters (corresponding to MD heading levels 1-6)
    # \s+: Must be followed by at least one space (to distinguish '#' headings from plain text)
    # .+: Title text must contain at least 1 character (avoids empty headings)
    title_pattern = r'^\s*#{1,6}\s+.+'
    # Split MD content into a list of lines by newline characters for line-by-line processing
    lines = content.split("\n")
    sections = []       # List to store the final split sections
    current_title = ""  # Title of the current section
    current_lines = []  # Line buffer for the current section
    title_count = 0     # Count of valid titles (excluding those inside code blocks)
    in_code_block = False  # Code block flag: Prevents misidentifying '#' inside code blocks as headings
    
    def _flush_section():
        """Internal helper function: Commits the currently cached lines into the sections list; skips if empty."""
        if not current_lines:
            return
        sections.append({
            "title": current_title,
            # Separate each line with \n when joining the content
            "content": "\n".join(current_lines),
            "file_title": file_title,
        })
        
    # Iterate through the document line by line to identify headings and split sections.
    for line in lines:
        stripped_line = line.strip()      
        # Identify code block boundaries (``` or ~~~): toggles state when entering or exiting a code block
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue
        # Determine if the line is a valid heading: must be outside code blocks AND match the heading regex
        is_valid_title = (not in_code_block) and re.match(title_pattern, line)
        if is_valid_title:
            # Encountered a new heading: flush the previous section to results, then initialize the new section
            _flush_section()
            current_title = line.strip()       # Strip whitespaces around the heading
            current_lines = [current_title]    # Start the new section block with its heading
            title_count += 1
            logger.debug(f"Markdown heading identified: {current_title}")
        else:
            # Normal line: append directly to the line cache of the current section
            current_lines.append(line)
            
        # Handle the final section: flush the remaining lines left in the cache after the loop finishes
        _flush_section()
        logger.info(f"Step 2: Markdown heading splitting completed. Identified {title_count} valid headings out of {len(lines)} total original lines.")
    return sections, title_count, len(lines)

def step_3_handle_no_title(content: str, sections: List[Dict[str, Any]], title_count: int, file_title: str) -> List[Dict[str, Any]]:
    """
    [Step 3] Fallback handling for documents with no headings.
    Purpose: If no headings are identified within the Markdown content, treats the entire text as a single section to prevent downstream runtime errors.
    
    :param content: The standardized full Markdown content.
    :param sections: The list of sections generated from Step 2.
    :param title_count: The count of valid headings identified in Step 2.
    :param file_title: The title of the parent file.
    :return: The final list of sections after applying fallback logic if needed.
    """
    if title_count == 0:
        # Fallback scenario (no headings found): wrap the entire text into a single section with a default title.
        logger.warning(f"Step 3: No Markdown headings identified. Treating the full text as a single section for file: {file_title}")
        return [{"title": "No Heading", "content": content, "file_title": file_title}]
        
    # Standard scenario (headings exist): directly return the initial split sections from Step 2.
    logger.debug(f"Step 3: Detected {title_count} valid headings; no fallback handling required.")
    return sections

def _split_long_section(section: Dict[str, Any], max_length: int = DEFAULT_MAX_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """
    [Helper Function] Secondary splitting for oversized sections (Core adaptation for LangChain splitters).
    Purpose: When a single section exceeds the maximum character threshold, it breaks down from coarse to fine 
             following a "Paragraph -> Sentence -> Space" hierarchy to maximize semantic preservation.
    Splitting Rules: 1. Split by empty lines (paragraphs) first 2. Split by newlines 3. Split by Chinese/English punctuation or spaces.
    
    :param section: The original section dictionary. Must contain a 'content' key, optional 'title'/'file_title'.
    :param max_length: Maximum character length for a single chunk. Defaults to the global configuration value.
    :return: A list of newly split sub-sections containing parent titles, index positions, and metadata.
    """
    # Fallback for null or empty content: return the original section directly
    content = section.get("content", "") or ""
    
    # Under limit check: if length doesn't exceed the limit, skip splitting and return original section in a uniform list format
    if len(content) <= max_length:
        return [section]
    # Standardization preprocessing: unify newline formats to prevent parsing inconsistencies across different OS platforms (\r\n vs \n)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # Extract section heading to construct sub-chunk prefixes (retains heading context)
    title = section.get("title", "") or ""
    # Heading prefix: paired with empty lines to clearly isolate it from the body text
    prefix = f"{title}\n\n" if title else ""
    # Compute the usable length for the remaining body text: total max length minus the title prefix length
    # This prevents the heading context from consuming the entire chunk quota.
    available_len = max_length - len(prefix)
     # Extreme scenario: heading length itself exceeds the max_length threshold, making splitting impossible. Return original section.
    if available_len <= 0:
        logger.warning(f"Section title is too long to split: {title[:20]}...")
        return [section]
    # Deduplicate repeating titles in the body: prevents redundant tokens if the body text starts with its own title string
    body = content
    if title and body.lstrip().startswith(title):
        body = body[body.find(title) + len(title):].lstrip()
         # Initialize LangChain RecursiveCharacterTextSplitter (Core utility: splits sequentially using a prioritized delimiter list)
    # separators: Priority order from coarse to fine. Prioritizes largest semantic blocks, falling back to a hard split last.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len, # Maximum allowed length for the body text section (after deducting title)
        chunk_overlap=0,          # Zero overlap: heading splits retain complete semantics, making overlap redundancies unnecessary
        # Delimiter priorities: Empty line (paragraphs) -> Newline -> Chinese punctuation -> English punctuation -> Space, followed by a hard split.
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )
    # Slice the body text and construct sub-sections populated with end-to-end trace metadata
    sub_sections = []
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        # Filter empty strings: bypass chunks that boil down to whitespaces after splitting
        text = chunk.strip()
        if not text:
            continue
            
        # Assemble sub-chunk content: combine the heading prefix with the sliced body text
        full_text = (prefix + text).strip()
         # Sub-section metadata generation: preserves parent tracking relationships and sequence indices for retrieval and mapping
        sub_sections.append({
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # Sub-chunk heading (indexed)
            "content": full_text,                                   # Complete split content text
            "parent_title": title,                                  # Parent heading reference (critical for eventual merges)
            "part": idx,                                            # Sub-chunk partition sequence ID
            "file_title": section.get("file_title"),                # Parent document file title
        })
        
    logger.debug(f"Oversized section splitting completed: {title} -> Generated {len(sub_sections)} sub-chunks.")
    return sub_sections

def _merge_short_sections(sections: List[Dict[str, Any]], min_length: int = MIN_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """
    [Helper Function] Merges overly short sections (Minimizes text fragmentation to optimize retrieval accuracy).
    Core Rule: Only adjacent chunks that share the exact same 'parent_title' AND whose accumulated length falls 
               below the minimum threshold will be merged. This strictly prevents cross-heading or cross-chapter pollution.
               
    :param sections: The list of chunks pending evaluation (typically the output flattened from _split_long_section).
    :param min_length: The minimum character length threshold. Any chunk shorter than this value will trigger a merge.
    :return: An aggregated list of moderately sized chunks with perfectly retained metadata and tracking history.
    """
    # Boundary handling: If input list is empty, return immediately to prevent indexing errors or trace failures downstream.
    if not sections:
        logger.debug("The list of chunks pending merge is empty; returning directly.")
        return []
    merged_sections = []  # Holds the final collection of properly sized, aggregated chunks
    current_chunk = None  # Iterative accumulator: caches the chunk currently being built/extended
    for sec in sections:
        # Initialization: Capture the first chunk in the sequence as our initial baseline accumulator block
        if current_chunk is None:
            current_chunk = sec
            continue
        # Evaluation criteria: 1. Current accumulated chunk length is under limit 2. Shares the same parent title context
        is_current_short = len(current_chunk["content"]) < min_length
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")
        parent_title = sec.get("parent_title", "")
        next_content = sec["content"]
        if parent_title and next_content.startswith(parent_title):
            next_content = next_content[len(parent_title):].lstrip()
        # Append content: join using an empty line spacing delimiter to keep block formatting clean and structured
            current_chunk["content"] += "\n\n" + next_content

            # Update sub-chunk partitioning sequence: carry over the latest part index to preserve lineage tracing
            if "part" in sec:
                current_chunk["part"] = sec["part"]  
                logger.debug(f"Merged short chunk under: {current_chunk.get('parent_title')} -> New accumulated length: {len(current_chunk['content'])}")
        else:
            # Criteria not met: flush the finalized accumulator block into results, then reset accumulator with the new incoming chunk
            merged_sections.append(current_chunk)
            current_chunk = sec
     # Post-loop finalization: ensure the absolute last remaining block in the accumulator is safely flushed to results
    if current_chunk is not None:
        merged_sections.append(current_chunk)

    logger.debug(f"Short chunk aggregation complete: reduced from {len(sections)} fragments down to {len(merged_sections)} optimized chunks.")
    return merged_sections

def step_4_refine_chunks(sections: List[Dict[str, Any]], max_len: int) -> List[Dict[str, Any]]:
    """
    [Step 4] Refined chunk processing (Core logic: split long sections and merge short ones to optimize for LLMs/Retrieval).
    Execution flow: 1. Split oversized sections 2. Merge overly short sections 3. Provide parent title fallbacks (to adapt to Milvus vector DB schema).
    
    :param sections: The list of sections passed down from Step 3.
    :param max_len: The maximum character length permitted for a single chunk.
    :return: The final list of moderately-sized, low-fragmentation chunks.
    """
    # Boundary handling: If maximum length configuration is invalid (null or <= 0), return original sections directly to prevent parsing errors.
    if not max_len or max_len <= 0:
        logger.warning(f"Step 4: Invalid chunk maximum length configuration ({max_len}). Skipping refinement processing.")
        return sections
    # Phase 1: Split oversized sections -> Restrict all individual section lengths within max_len.
    refined_split = []
    for sec in sections:
        # Perform oversize splitting on each section and flatten the results into the main list (avoids nested lists).
        # The purpose of 'extend' is to unpack items from another list (or iterable) and append them individually to the tail of the current list.
        refined_split.extend(_split_long_section(sec, max_len))
    logger.info(f"Step 4-1: Oversized section splitting completed. Generated {len(refined_split)} initial sub-chunks.")
    # Phase 2: Merge overly short sections -> Minimize text fragmentation to improve downstream retrieval accuracy and LLM comprehension.
    final_sections = _merge_short_sections(refined_split)
    logger.info(f"Step 4-2: Overly short section merging completed. Produced {len(final_sections)} final chunks.")
    # Phase 3: Parent title fallback -> Adapts to the Milvus vector database schema (where parent_title is a mandatory field).
    # Fallback rules: If parent_title is missing, use the section's own title; if that is also missing, fall back to an empty string.
    for sec in final_sections:
        if not isinstance(sec, dict):
            continue
        # Complement missing 'part' field (defaults to 0) to align with Milvus schema requirements
        if "part" not in sec:
            sec["part"] = 0
        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""
            
    logger.debug(f"Step 4-3: Parent title fallback completed. All chunks now contain a 'parent_title' field.")
    return final_sections

def step_5_print_stats(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """
    [Step 5] Print document chunking statistical information (logging for monitoring/debugging)
    :param lines_count: Total number of lines in the original Markdown text
    :param sections: The final list of processed chunks
    """
    chunk_num = len(sections)
    # Print core statistical information: original line count / final chunk count / first chunk preview
    logger.info("-" * 50 + " Document Chunking Stats " + "-" * 50)
    logger.info(f"Total lines in original MD text: {lines_count}")
    logger.info(f"Final number of chunks generated: {chunk_num}")
    if sections:
        first_title = sections[0].get("title", "No Title")
        logger.info(f"First chunk title:{first_title}")
    logger.info("-" * 110)
    
def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None: 
    """
    [Step 6] Local JSON backup of chunk results (for debugging/troubleshooting, retaining processing results)
    :param state: Project state dictionary, must contain 'local_dir' (backup directory)
    :param sections: The final list of processed chunks
    """ 
    # Extract backup directory: return immediately if missing, do not execute backup
    local_dir = state.get("local_dir") 
    if not local_dir: 
        logger.warning("Step 6: Backup directory (local_dir) not configured, skipping chunk backup") 
        return 
    try: 
        # Create backup directory: do not throw error if it already exists (exist_ok=True)
        os.makedirs(local_dir, exist_ok=True) 
        
        # Concatenate backup file path: local_dir + chunks.json (fixed filename for easy lookup)
        backup_path = os.path.join(local_dir, "chunks.json") 
        
        # Write JSON file: preserve Chinese characters / format with indentation for easy manual inspection
        with open(backup_path, "w", encoding="utf-8") as f: 
            """
            'sections' is a nested Python data structure (List[Dict[str, Any]]—a list containing dictionaries, 
            which may further nest strings, numbers, etc.). Ordinary file writing (like f.write(sections)) 
            only supports writing strings; attempting to write a Python data structure directly will throw an error.
            
            The core purpose of json.dump is to directly serialize Python native data structures (lists, 
            dictionaries, strings, numbers, etc.) and write them into a JSON file without manual string conversion. 
            At the same time, it ensures the data format is standardized and readable across different languages/scenarios, 
            making it a perfect fit for the "Chunk list backup" requirement.
            """ 
            json.dump( 
                sections, 
                f, 
                # ensure_ascii=True -> "title": "\u4e00\u7ea7\u6807\u9898" (encoded, unreadable directly); 
                # ensure_ascii=False -> "title": "一级标题" (normal Chinese characters, directly readable by humans). 
                ensure_ascii=False, # Retain characters, do not escape to \u unicode encoding 
                indent=2            # Format with indentation for readability
            ) 
        logger.info(f"Step 6: Chunk results backed up successfully. Backup file path: {backup_path}") 
    except Exception as e: 
        # Backup failure only logs the error; does not terminate the main process flow
        logger.error(f"Step 6: Chunk results backup failed. Error message: {str(e)}", exc_info=False)
    
    
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    [Core Node] Document Splitting Main Node (node_document_split)
    Overall Workflow: Load input -> Initial split by MD titles -> Fallback for text without titles -> Split long / Merge short -> Output statistics -> Result backup
    Core Purpose: Split long Markdown documents into chunks of appropriate length, tailored for LLM context windows and vector retrieval.
    Future Extension Points: Metadata enrichment for chunks, custom splitting rules, or pre-vector-database processing can be added between steps.
    :param state: Project state dictionary (ImportGraphState), must contain `md_content`/`task_id`; optionally contains `local_dir`/`max_content_length`/`file_title`.
    :return: Updated state dictionary with a new `chunks` key (stores the final processed list of chunks, where each chunk is a dictionary containing title/content/parent_title).
    """  
    
    # Initialize current node info for task monitoring and log traceability
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> Started executing core node: [Document Splitting] {node_name}")
    # Add current node to running tasks and update the global task state
    add_running_task(state["task_id"], node_name)
    try:
        # ===================================== Step 1: Load and Standardize Input Data =====================================
        # Function: Extract MD content/file title/max chunk length from the state dict, unify line breaks to eliminate OS differences, and handle empty values.
        # Output: Standardized `md_content`, file title, and max length for a single chunk; terminates node execution directly if no valid MD content is found.
        content, file_title, max_len = step_1_get_inputs(state)
        if content is None:
            logger.info(f">>> Node execution terminated: {node_name} (No valid MD content)")
            return state
        # ===================================== Step 2: Initial Splitting by Markdown Titles =====================================
        # Function: Split the document into independent sections based on Markdown titles (#/##/###), automatically skipping pseudo-titles within code blocks to ensure semantic integrity.
        # Output: List of initially split sections, count of identified valid titles, and total lines of raw MD text (for subsequent statistics/logging).
        sections, title_count, lines_count = step_2_split_by_titles(content, file_title)
        # ===================================== Step 3: Fallback Handling for Titleless Scenarios =====================================
        # Function: Resolve edge cases where the MD document has no titles at all, avoiding errors in subsequent splitting logic.
        # Output: Returns the section list from Step 2 if titles exist; otherwise, wraps the entire text into a single "Untitled" section to ensure a uniform data format.
        sections = step_3_handle_no_title(content, sections, title_count, file_title)
        # ===================================== Step 4: Fine-grained Chunk Processing (Split Long / Merge Short) =====================================
        # Function: Core splitting logic. First, split over-length sections further by "paragraph -> sentence", then merge sections that are too short under the same parent title to reduce fragmentation.
        # Extra Handling: Apply a fallback `parent_title` to all chunks to satisfy the required field constraints of the Milvus vector database.
        # Output: A final list of chunks with appropriate length, complete semantics, and low fragmentation (ready for vector DB insertion/LLM invocation).
        sections = step_4_refine_chunks(sections, max_len)
        # ===================================== Step 5: Output Document Splitting Statistics =====================================
        # Function: Print core statistical data to easily monitor splitting performance and debug issues (raw line count / final chunk count / first chunk preview).
        # Output: No return value, outputs standardized statistical logs via logger only.
        step_5_print_stats(lines_count, sections)
        # ===================================== Step 6: Local JSON Backup of Chunk Results + State Update =====================================
        # Function 1: Back up the final chunk list to `chunks.json` in the `local_dir` directory for troubleshooting and data reuse.
        # Function 2: Write the chunk list into the state dictionary to pass it to downstream nodes (e.g., vector database insertion, LLM summarization, etc.).
        # Output: Adds the `chunks` key to the state dictionary; skips backup if `local_dir` is absent without affecting the main workflow.
        state["chunks"] = sections
        step_6_backup(state, sections)
        
    except Exception as e:
        # Global exception capture: Ensures that node execution failure does not crash the entire workflow; records detailed error logs for troubleshooting.
        logger.error(f">>> Core node execution failed: [Document Splitting] {node_name}, error message: {str(e)}", exc_info=True)
    # Return the updated state dictionary to pass chunk results to downstream nodes
    return state

if __name__ == '__main__':
    """
    Integration Test: Joint test executing node_md_img (image processing node) and current splitting node.
    Prerequisites: 1. .env must be configured (MinIO/LLM parameters setup).
                   2. A valid target Markdown file must exist.
                   3. node_md_img must be importable.
    Execution Flow: Run image extraction/processing first -> execute structural chunk splitting -> verify end-to-end flow.
    """
    
    """Local Testing Entrypoint: Executes the comprehensive end-to-end MD image processing suite when executed as a standalone script."""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img
    
    logger.info(f"Local Test - Project Root Directory: {PROJECT_ROOT}")
    
    # Path to the test Markdown file (ensure the target manual folder and file exist at this location)
    test_md_name = os.path.join(r"output/aag-cisco-umbrella", "aag-cisco-umbrella.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)
    
    # Verify whether the designated test document exists
    if not os.path.exists(test_md_path):
        logger.error(f"Local Test - Target test file does not exist: {test_md_path}")
        logger.info("Please verify the file path string, or manually place your test Markdown folder under the 'output' directory at the project root.")
    else:
        # Construct the testing state dictionary to simulate standard upstream workflow inputs
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "aag-cisco-umbrella",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        
        logger.info("Initiating Local Test - Full-Scale MD & Image Processing Workflow")
        
        # Invoke the core upstream image handling node pipeline
        result_state = node_md_img(test_state)
        logger.info(f"Local Test Phase 1 Complete - Intermediate Result State: {result_state}")
        
        logger.info("\n=== Starting Integration Test Suite for Document Splitting Node ===")
        logger.info(">> Spinning up target node invocation: node_document_split (Document Chunking)")
        
        # Invoke the current document slicing and structural compilation node
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        
        logger.info(f"✅ Test Execution Successful: Generated {len(final_chunks)} valid, processed target Chunks: {final_chunks}")