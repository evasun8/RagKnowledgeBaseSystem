import os
from tavily import TavilyClient  # pip install tavily-python
from dotenv import load_dotenv

# Import your existing app utilities - adjust path as needed
# from app.core.logger import logger
# from app.utils.task_utils import add_running_task, add_done_task

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def node_web_search_mcp(state):
    """
    LangGraph node: perform web search using Tavily to supplement knowledge base results.
    Replaces the original Bailian MCP web search node.

    Args:
        state: LangGraph state dict containing session_id, rewritten_query, is_stream

    Returns:
        dict: { "web_search_docs": list of search result dicts }
    """
    # add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print("--- node_web_search: starting ---")

    # 1. Get the rewritten query from state
    query = state.get("rewritten_query", "")
    if not query:
        print("--- node_web_search: no query found, skipping ---")
        return {"web_search_docs": []}

    # 2. Call Tavily search
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            search_depth="basic",   # "basic" (faster) or "advanced" (more thorough)
            max_results=5,
            include_answer=True,    # Tavily generates a brief summary answer
            include_raw_content=False,
        )

        # 3. Parse results into the same format as original web_search_docs
        # Original format: [{ "title": ..., "url": ..., "snippet": ... }]
        web_documents = []

        # Add Tavily's summary as the first item if available
        if response.get("answer"):
            web_documents.append({
                "title": "Tavily Summary",
                "url": "",
                "snippet": response["answer"],
                "hostname": "tavily"
            })

        # Add individual search results
        for result in response.get("results", []):
            web_documents.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")[:500],  # truncate to 500 chars
                "hostname": result.get("url", "").split("/")[2] if result.get("url") else ""
            })

        # logger.info(f"Tavily search results: {web_documents}")
        print(f"--- node_web_search: found {len(web_documents)} results ---")

    except Exception as e:
        print(f"--- node_web_search: search failed - {e} ---")
        web_documents = []

    print("--- node_web_search: done ---")
    # add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    # Return partial state update (parallel-safe)
    return {"web_search_docs": web_documents}


# -------------------------------------------------------
# Quick test
# -------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    test_state = {
        "session_id": "tavily_test_01",
        "rewritten_query": "What is RAG (Retrieval Augmented Generation) and how does it improve LLM accuracy?",
        "is_stream": True
    }

    result_state = node_web_search_mcp(test_state)

    print("\nTest Results:")
    print(f"Query: {test_state.get('rewritten_query')}")
    search_results = result_state.get("web_search_docs", [])
    print(f"Number of results: {len(search_results)}")
    for i, doc in enumerate(search_results, 1):
        print(f"\n[{i}] {doc['title']}")
        print(f"    URL: {doc['url']}")
        print(f"    Snippet: {doc['snippet'][:150]}...")