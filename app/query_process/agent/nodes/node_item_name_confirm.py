import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
 
from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger
 
load_dotenv(find_dotenv())
 
 
def step_3_extract_info(query: str, history: List[Dict]) -> Dict:
    """
    Extract the primary queried product names (item_names, as a JSON list) from the current question 
    and historical sessions using LLM.
    If the product name is not clear enough, an empty list is returned. Meanwhile, the query 
    is rewritten based on context to ensure it is self-contained and complete.
    :param query: str - User's current raw query (e.g., "How much is this?")
    :param history: List[Dict] - Recent session history
    :return: Dict - Extraction result, format: {"item_names": [], "rewritten_query": ""}
    """
    logger.info("Step 3: Starting information extraction (LLM)")
     
    # 1. Initialization and preparation
    client = get_llm_client(json_mode=True)
     
    # Construct historical dialogue text
    history_text = ""
    for msg in history:
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"
     
    logger.info(f"Step 3: Historical context construction completed, length: {len(history_text)} characters")
 
    # 2. Load the prompt template
    try:
        # Pass variables via keyword arguments to avoid positional argument errors
        prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
        logger.debug(f"Step 3: Prompt loaded successfully, Prompt length: {len(prompt)}")
    except Exception as e:
        logger.error(f"Step 3: Failed to load prompt: {e}")
        return {"item_names": [], "rewritten_query": query}
 
    messages = [
        SystemMessage(content="You are a professional customer service assistant, expert in understanding user intent and extracting key information."),
        HumanMessage(content=prompt)
    ]
 
    try:
        logger.info("Step 3: Calling LLM for extraction...")
        response = client.invoke(messages)
        content = response.content
        logger.debug(f"Step 3: Raw LLM response: {content}")
 
        # Clean up Markdown code block wrappers
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
         
        result = json.loads(content)
         
        # Robustness checks
        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = query
             
        logger.info(f"Step 3: Extraction result parsed successfully - Item Names: {result['item_names']}, Rewritten Query: {result['rewritten_query']}")
        return result
 
    except Exception as e:
        logger.error(f"Step 3: LLM extraction or parsing failed: {e}")
        return {"item_names": [], "rewritten_query": query}
 
 
def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict]:
    """
    Vectorizes the extracted item_names and performs hybrid search inside Milvus.
    """
    logger.info(f"Step 4: Starting vector retrieval, target items: {item_names}")
    results = []
     
    client = get_milvus_client()
    if not client:
        logger.error("Step 4: Unable to connect to Milvus")
        return results
 
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("Step 4: ITEM_NAME_COLLECTION not found in environment variables")
        return results
 
    try:
        logger.info("Step 4: Generating Embeddings (Dense + Sparse)...")
        embeddings = generate_embeddings(item_names)
        logger.info(f"Step 4: Vector generation completed. Starting Milvus search (Collection: {collection_name})")
 
        for i, name in enumerate(item_names):
            try:
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]
 
                # Construct hybrid search request
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )
 
                # Execute hybrid search
                # Adjust weight ratios to 0.8 (Dense) / 0.2 (Sparse) to optimize scores
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2), 
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"]
                )
 
                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        entity = hit.get("entity") or {}
                        item_name = entity.get("item_name")
                        score = hit.get("distance")
                         
                        if item_name:
                            matches.append({
                                "item_name": item_name,
                                "score": score
                            })
                            logger.debug(f"Step 4: Match for '{name}': {item_name} (Score: {score:.4f})")
 
                results.append({
                    "extracted_name": name,
                    "matches": matches
                })
                logger.info(f"Step 4: Item retrieval for '{name}' completed, found {len(matches)} matches")
 
            except Exception as inner_e:
                logger.error(f"Step 4: Error encountered while processing item '{name}': {inner_e}")
                results.append({"extracted_name": name, "matches": []})
 
    except Exception as e:
        logger.error(f"Step 4: Global exception occurred during vectorization or search: {e}")
 
    return results
 
 
def step_5_align_item_names(query_results: List[Dict]) -> Dict:
    """
    Aligns item names based on Milvus search scores, generating 'confirmed item names' and 'candidate item name options'.
    """
    logger.info("Step 5: Starting item name alignment (Score Analysis)")
     
    confirmed_item_names = []
    options = []
 
    for res in query_results:
        extracted_name = res.get("extracted_name", "").strip()
        matches = res.get("matches", []) or []
         
        if not matches:
            logger.info(f"Step 5: No matching results for '{extracted_name}'")
            continue
 
        # Sort in descending order by score
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
         
        # Print detailed score logs to assist debugging
        top_matches_log = ", ".join([f"{m['item_name']}({m['score']:.3f})" for m in matches[:3]])
        logger.info(f"Step 5: '{extracted_name}' Top Matches: {top_matches_log}")
 
        # Filtering thresholds
        high = [m for m in matches if m.get("score", 0) > 0.85]
        mid = [m for m in matches if m.get("score", 0) >= 0.6]
 
        # Rule A: Single high-confidence match
        if len(high) == 1:
            confirmed_name = high[0].get("item_name")
            confirmed_item_names.append(confirmed_name)
            logger.info(f"Step 5: Rule A Hit (Single High) -> Confirmed: {confirmed_name}")
            continue
 
        # Rule B: Multiple high-confidence matches
        if len(high) > 1:
            picked = None
            # Prioritize matching exact same name
            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        logger.info(f"Step 5: Rule B Hit (Exact Match in High) -> Confirmed: {picked.get('item_name')}")
                        break
             
            # Otherwise, pick the highest score
            if not picked:
                picked = high[0]
                logger.info(f"Step 5: Rule B Hit (Highest Score) -> Confirmed: {picked.get('item_name')}")
 
            confirmed_item_names.append(picked.get("item_name"))
            continue
 
        # Rule C: No high-confidence matches, pick from mid-confidence candidates
        if len(mid) > 0:
            current_options = [m.get("item_name") for m in mid[:5]]
            options.extend(current_options)
            logger.info(f"Step 5: Rule C Hit (Mid Confidence) -> Candidates added: {current_options}")
            continue
         
        logger.info("Step 5: Rule D Hit (Low Confidence) -> No matches")
 
    result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options))
    }
    logger.info(f"Step 5: Alignment results: {result}")
    return result
 
 
def step_6_check_confirmation(state: Dict, align_result: Dict, session_id: str, history: List[Dict], rewritten_query: str) -> Dict:
    """
    Checks the alignment results and updates the State object.
    """
    logger.info("Step 6: Checking confirmation status and updating State")
     
    # Robustness handling
    if align_result is None:
        align_result = {}
 
    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])
 
    # Branch A: Confirmed item name exists
    if confirmed:
        logger.info(f"Step 6: [Branch A] Confirmed item name exists: {confirmed}")
         
        # Update item_names in history messages
        ids_to_update = []
        for msg in history:
            if not msg.get("item_names"):
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))
         
        if ids_to_update:
            logger.info(f"Step 6: Updating associated item names for {len(ids_to_update)} history messages")
            update_message_item_names(ids_to_update, confirmed)
 
        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return state
 
    # Branch B: Candidate item name options exist
    if options:
        logger.info(f"Step 6: [Branch B] Candidate item names exist: {options}")
        options_str = " or ".join(options[:3])
        answer = f"Did you mean one of the following products: {options_str}? Please specify the exact model."
        state["answer"] = answer
        state["item_names"] = []
        return state
 
    # Branch C: No matching results
    logger.info("Step 6: [Branch C] No confirmed and no candidate products found")
    state["answer"] = "Sorry, no matching product was found. Please provide the exact model so I can search it for you."
    state["item_names"] = []
    return state
 
 
def step_7_write_history(state: Dict, session_id: str, history: List[Dict], rewritten_query: str, message_id: str) -> Dict:
    """
    Writes final session records to database.
    """
    logger.info("Step 7: Writing session history")
     
    # If there is an assistant reply (Branch B/C), save assistant message
    if state.get("answer"):
        logger.info("Step 7: Saving assistant reply")
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=state["answer"],
            rewritten_query="",
            item_names=[]
        )
 
    # Update user message (linking rewritten_query and item_names)
    logger.info(f"Step 7: Updating user message (ID: {message_id})")
    save_chat_message(
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        item_names=state.get("item_names", []),
        message_id=message_id
    )
 
    return state
 
 
def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    Main node function: Item name confirmation process.
    """
    logger.info(">>> node_item_name_confirm: Processing started")
     
    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)
 
    # Mark task start
    add_running_task(session_id, "node_item_name_confirm", is_stream)
 
    # 1. Retrieve history records
    history = get_recent_messages(session_id, limit=10)
    logger.info(f"Node: Retrieved {len(history)} historical messages")
 
    # 2. Initial save of the user's current message (will be updated later in Step 7)
    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))
    logger.debug(f"Node: User message initially saved, ID: {message_id}")
 
    # 3. Extract information
    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)
     
    # Update rewritten_query in State
    state["rewritten_query"] = rewritten_query
 
    align_result = {}
 
    # 4. & 5. If item names are extracted, perform search and alignment
    if len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: No item names extracted. Skipping vector retrieval.")
 
    # 6. Check confirmation status
    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)
 
    # 7. Write final session history
    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)
 
    # Save history inside state for downstream nodes (e.g., node_answer_output)
    final_state["history"] = history
 
    # Mark task as completed
    add_done_task(session_id, "node_item_name_confirm", is_stream)
     
    logger.info(f"Node: Processing finished, Final State Item Names: {final_state.get('item_names')}")
    return final_state
 
 
if __name__ == "__main__":
    # Local test block
    print("\n" + "="*50)
    print(">>> Starting node_item_name_confirm local test")
    print("="*50)
     
    # Mock input state
    mock_state = {
        "session_id": "test_debug_session_001",
        "original_query": "what is Meraki MS120-8 Compact Switch?",  # Target specific debug case
        "is_stream": False,
        "item_names": []
    }
 
    try:
        # Run node function
        result = node_item_name_confirm(mock_state)
         
        print("\n" + "="*50)
        print(">>> Test Results Summary:")
        print(f"Rewritten Query: {result.get('rewritten_query')}")
        print(f"Item Names: {result.get('item_names')}")
        print(f"Answer: {result.get('answer')}")
        print("="*50)
 
    except Exception as e:
        logger.exception(f"Unhandled exception encountered during test execution: {e}")