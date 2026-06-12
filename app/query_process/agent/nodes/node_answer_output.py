import sys
import json
import asyncio
import re
import time
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
from dotenv import load_dotenv, find_dotenv
 
load_dotenv(find_dotenv())
 
_IMAGE_BLOCK_MARKER = "[Image]"
MAX_CONTEXT_CHARS = 12000

def step_1_check_answer(state: dict) -> bool:
    """
    Phase 1: Check if an answer already exists in the state.
    - If it exists: Push streaming delta (for SSE) if requested, or set task result, and return True.
    - If it does not exist: Return False.
    """
    answer = state.get("answer", None)
    is_stream = state.get("is_stream", False)
    if answer:
        if is_stream:
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["session_id"], "answer", answer)
        return True
    return False

def step_2_construct_prompt(state: dict) -> str:
    """
    Phase 2: Construct Prompt
    Organizes the prompt template using the original query, rewritten query, chat history,
    target item names, and reranked content from the state.
    """
    # 1. Retrieve the original query and rewritten query
    query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query", "")
    question = rewritten_query or query
    history = state.get("history", [])
    item_names = state.get("item_names", [])
    reranked_docs = state.get("reranked_docs", [])
    
    # 2. Extract context string from reranked docs without exceeding limits
    # Prioritize structured reranked_docs (containing source, chunk_id, url, score, etc.)
    # ---------------------------------------------------------
    # Process Explanation:
    # 1. Iterate through precision reranked documents list (reranked_docs) sorted by relevance.
    # 2. Extract metadata fields (text, source, chunk_id, url, title, score) for each.
    # 3. Format as "[Index] [Source] [Chunk metadata] \n Content Text".
    # 4. Accumulate characters. If it exceeds MAX_CONTEXT_CHARS (e.g. 12000 chars), stop adding
    #    to ensure prompt size fits within the LLM context window, preventing token overflow.
    # ---------------------------------------------------------
    docs = []
    used = 0
    for i, doc in enumerate(reranked_docs, start=1):
        # Extract metadata fields
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        source = doc.get("source") or ""
        chunk_id = doc.get("chunk_id") or ""
        url = doc.get("url") or ""
        title = doc.get("title") or ""
        score = doc.get("score") or 0.0
        
        meta_parts = [f"[{i}]"]
        if source:
            meta_parts.append(f"[{source}]")
        if chunk_id:
            meta_parts.append(f"[{chunk_id}]")
        if url:
            meta_parts.append(f"[{url}]")
        if title:
            meta_parts.append(f"[{title}]")
        meta_parts.append(f"[{score:.4f}]")
        
        # Format as "[Index] [Source] [Chunk metadata] \n Content Text"
        meta_str = " ".join(meta_parts)
        content = f"{meta_str}\n{text}"
        
        docs.append(content)
        used += len(content)
        if used + len(meta_str) > MAX_CONTEXT_CHARS:
            break
        
    context_str = "\n\n".join(docs) if docs else "No relevant content found."
    # 3. Format Conversational History Logs
    # ---------------------------------------------------------
    # Process Explanation:
    # 1. Iterate through history records from MongoDB.
    # 2. Reformat each turn into a standard "User: ... \n Assistant: ..." dialogue format.
    # 3. Continue accumulating length (used) to ensure historical logs do not overflow context limits.
    #    Note: This is cumulative with reference context, meaning extremely long context files 
    #    might truncate historical logs.
    # ---------------------------------------------------------
    
    history_str = ""
    if history:
        for msg in history:
            role = msg.get("role", "")
            text = msg.get("text", "")
            if role == "user" and text:
                history_str += f"User: {text}\n"
            elif role == "assistant" and text:
                history_str += f"Assistant: {text}\n"
            used += len(history_str)
            if used + len(history_str) > MAX_CONTEXT_CHARS:
                break
    else:
        history_str = "No historical dialogue found."
    
    # 4. Format Item Names
    item_names_str = ", ".join(item_names) if item_names else "No item names found."
    
    # 5. Format Prompt Template
    prompt = load_prompt("answer_out", query=query, context=context_str, history=history_str, item_names=item_names_str, question=question)
    return prompt

def _extract_images_from_docs(docs: List[Dict[str, Any]]) -> List[str]:
    """
    Helper Method: Extracts unique image URLs from the retrieval documents list.
    
    Extraction Heuristics:
    1. Iterate through all related documents (including local chunks and web search results).
    2. Check the 'url' field directly (primarily for web documents matching image extension suffixes).
    3. Run a Regex scan over the 'text' field contents (primarily for local Markdown chunks matching standard syntax: ![alt](url)).
    4. Deduplicate matched URLs and return the sorted unique list.
    """
    images = []
    seen = set()  # Deduplication set
    if not docs:
        return []

    # Regex pattern matching standard Markdown image syntax: !\[.*?\]\((.*?)\)
    # Group 1 captures the URL text within the parenthesis.
    md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

    logger.info(f"Extracting image assets from {len(docs)} document sources...")

    for i, doc in enumerate(docs):
        # 1. Check 'url' metadata field directly
        url = (doc.get("url") or "").strip()
        if url:
            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                if url not in seen:
                    logger.debug(f"Document [{i}] - Image URL field match identified: {url}")
                    seen.add(url)
                    images.append(url)

        # 2. Scan Markdown patterns inside document text
        text = (doc.get("text") or "").strip()
        if text:
            matches = md_img_pattern.findall(text)
            for img_url in matches:
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    logger.debug(f"Document [{i}] - Markdown image pattern match identified: {img_url}")
                    seen.add(img_url)
                    images.append(img_url)

    logger.info(f"Image asset extraction finished. Identified {len(images)} unique URL(s): {images}")
    return images

def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
    """
    Phase 3: Generate Response
    Invokes the LLM to generate the answer, supporting streaming outputs.
    """
    logger.info("---Step 3: Starting response generation (LLM Generation)---")
    logger.debug(f"Final Prompt text: {prompt}")
    
    llm = get_llm_client()
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    if is_stream:
        logger.info(f"Streaming output enabled. Session ID: {session_id}")
        final_text = ""
        try:
            # Iterate over the prompt until the response is complete
            for chunk in llm.stream(prompt):
                delta = getattr(chunk, "content","") or ""
                if delta:
                    final_text += delta
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
            logger.info(f"Streaming completed. Total characters: {len(final_text)}")
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
    else:
        logger.info("Streaming disabled. Session ID: {session_id}")
        try:
            response = llm.invoke(prompt)
            final_text = response.content
            state["answer"] = final_text
            set_task_result(session_id, "answer", final_text)
            logger.info(f"Response generated successfully. {final_text}")
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["answer"] = "Sorry, an error occurred during response generation."
    
    return state

def step_4_write_history(state: QueryGraphState,images_urls: List[str]=None) -> QueryGraphState:
    """
    Phase 4: Log and persist this turn's response to MongoDB chat history.
    """
    session_id = state.get("session_id","default")
    answer = state.get("answer","").strip()
    item_names = state.get("item_names",[])
    try:
        if answer:
            logger.info(f"Saving final turn to MongoDB history")
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                item_names=item_names,
                image_urls=images_urls,
                rewritten_query="",
                message_id = None
            )
        else:
            logger.info(f"No answer found, skipping MongoDB history save")
    except Exception as e:
        logger.error(f"Error saving final turn to MongoDB history: {e}")
    return state
        

def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    Main Node Function:
    1. Check if 'answer' already exists in the state. If true, outputs it directly (respecting SSE streaming flag).
    2. If no answer exists, constructs prompt context using query rewrites, history, and retrieved documents.
    3. Invokes LLM to generate the answer (respecting SSE streaming flag).
    4. Saves the completed dialog turn to MongoDB history.
    5. Performs final SSE PUSH signals containing standard image URLs for front-end rendering.
    """
    logger.info("---node_answer_output (Response Generation) starting execution---")
    add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    
    # Phase 1: Direct exit check
    answer_exists = step_1_check_answer(state)
    
    # Phase 2: Construct prompt context if answer is missing
    if not answer_exists:
        prompt = step_2_construct_prompt(state)
        state["prompt"] = prompt
        
        # Phase 3: Generate response via LLM
        step_3_generate_response(state, prompt)

    # Extract images assets from reranked context documents
    images_urls = _extract_images_from_docs(state.get("reranked_docs", []))
    
    # Phase 4: save final record turn to MongoDB history
    if state.get("answer"):
        logger.info(f"Node: Saving final turn to MongoDB history")
        step_4_write_history(state,images_urls)
    
    # Phase 5: Stream termination, dispath FINAL metadata
    logger.info(f"Node: --- Dispatching final event --- Assets: {images_urls}")
    if state.get("is_stream"):
        push_to_session(state["session_id"], SSEEvent.FINAL, {"answer": state.get("answer",""), "status":"completed", "image_urls": images_urls})
        logger.info(f"Node: Streaming terminated, final metadata dispatched")
    return state

if __name__ == "__main__":
    # Local unit testing block
    print("\n" + "="*50)
    print(">>> Starting local test for node_answer_output")
    print("="*50)
     
    # Mock Reranked documents
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK_180_Stamping_Manual_v2.pdf",
            "score": 0.95,
            "text": """
            The control panel of the HAK 180 hot stamping machine is located on the front face.
            After turning on the power, you must first configure the operating temperature, which is recommended to be set around 110C.
            Refer to the diagram below for the control panel layout:
            ![Control Panel Layout](http://local-server/images/panel_view.jpg)
             
            For local hot stamping, please adjust the side dials.
            ![Side Dial Detail](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 Official Troubleshooting Portal",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # Direct image URL targeting Web asset extraction tests
            "text": "If the heater fails to initialize, inspect if the primary fuse is blown..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "Thermal Safety Regulations",
            "score": 0.82,
            "text": "Always wear insulated thermal gloves during machine operation to avoid high-temperature burns."
        }
    ]

    # Mock historical conversation log
    mock_history = [
        {"role": "user", "text": "Hello, how do I operate this machine?"},
        {"role": "assistant", "text": "Hello! Could you please clarify the specific machine model?"},
        {"role": "user", "text": "The HAK 180 hot stamping machine."}
    ]

    # Mock Input State
    mock_state = {
        "session_id": "test_answer_session_002",
        "original_query": "How do I operate the HAK 180 hot stamping machine?",
        "rewritten_query": "What are the specific operational steps and panel configurations for the HAK 180?",
        "item_names": ["HAK 180 hot stamping machine"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # Local test in blocking mode
        "answer": None       # Empty initial answer state
    }

    try:
        # Run node function
        result = node_answer_output(mock_state)
         
        print("\n" + "="*50)
        print(">>> Test Results Summary:")
         
        # 1. Verify Prompt Construction
        if "prompt" in result:
            print(f"[PASS] Prompt constructed successfully (length: {len(result['prompt'])})")
        else:
            print("[FAIL] Prompt construction failed.")

        # 2. Verify Response Generation
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] Response generated successfully (length: {len(answer)})")
            print(f"Response Preview: {answer[:60]}...")
        else:
            print(f"[WARN] Response generation anomaly noticed (Content: {answer})")

        # 3. Verify Image Extraction Logs
        # We expect to identify three distinct image URLs:
        # 1. http://local-server/images/panel_view.jpg (from local_101 via regex text scan)
        # 2. http://local-server/images/knob_detail.png (from local_101 via regex text scan)
        # 3. http://example.com/hak180_troubleshooting.jpeg (from web document's url metadata field)
        print("\n[INFO] Please verify if the following image assets were successfully logged above:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("="*50)

    except Exception as e:
        logger.exception(f"Unhandled exception encountered during test execution: {e}")
    
