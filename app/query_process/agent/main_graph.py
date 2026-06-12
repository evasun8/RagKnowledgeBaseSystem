from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
# Import all node functions
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

# Initialize the state graph
builder = StateGraph(QueryGraphState)

# Register all nodes
builder.add_node("node_item_name_confirm", node_item_name_confirm)  # Confirm product name
builder.add_node("node_multi_search", lambda x: x)  # Virtual node: branch point for multi-route search
builder.add_node("node_search_embedding", node_search_embedding)  # Dense vector search
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)  # HyDE vector search
builder.add_node("node_query_kg", node_query_kg)  # Knowledge Graph search
builder.add_node("node_web_search_mcp", node_web_search_mcp)  # Web search via MCP
builder.add_node("node_join", lambda x: {})  # Virtual node: merge point for multi-route search
builder.add_node("node_rrf", node_rrf)  # Rank (Reciprocal Rank Fusion)
builder.add_node("node_rerank", node_rerank)  # Rerank
builder.add_node("node_answer_output", node_answer_output)  # Answer generation

# The role of virtual nodes:
# Act as "branching / merging transfer stations" in the flow to solve multi-branch execution structures, containing no actual business logic.
# 'lambda x: x' logic: Receives the state and returns it as-is, serving as the most lightweight method for passing state without logic.
# Equivalent regular function replacement: Defining 'def function_name(state): return state' works exactly the same way, with the advantage of being easier to extend and debug.

# Set entry point
builder.set_entry_point("node_item_name_confirm")


def route_after_item_confirm(state: QueryGraphState):
    # If an answer already exists (Branch B/C), skip directly to the output node
    if state.get("answer"):
        """
        This primarily happens in scenarios where the node_item_name_confirm node cannot directly determine a unique product model,
        thus requiring clarification from the user (disambiguation) or a rejection response (fallback).
        
        Specifically, there are two situations that directly populate 'answer' in the state, thereby bypassing downstream retrieval:
        1. Multi-choice (Asking User for Clarification):
           - Scenario: The user's query is too vague (e.g., "Huawei P60"), and the system detects multiple models in the database 
             such as "Huawei P60 128G" and "Huawei P60 Art", with none of them having a high enough confidence score to be directly confirmed.
           - Action: The node generates a clarifying question as the 'answer' (e.g., "Did you mean one of the following products: Huawei P60 128G or Huawei P60 Art? Please specify the exact model.").
           - Result: Downstream document retrieval is skipped, and this question is sent directly to the user.
           
        2. Product Not Found (Refusal to Answer):
           - Scenario: The user queries a product that is entirely absent from the system (e.g., "Xiaomi 15", but the database only contains Huawei data) 
             or the matching score is too low (< 0.6).
           - Action: The node generates a fallback rejection statement as the 'answer' (e.g., "Sorry, no matching product was found. Please provide the exact model so I can search it for you.").
           - Result: Similarly, downstream retrieval is bypassed and the workflow terminates.
        """
        return "node_answer_output"
    # Otherwise, proceed with the multi-route search process
    return "node_multi_search"

# 1. Intent confirmation -> (Conditional Branch) -> Multi-route search / Direct answer output
builder.add_conditional_edges(
    "node_item_name_confirm",
    route_after_item_confirm
)
# 2. Concurrently execute 4-way search branches
builder.add_edge("node_multi_search", "node_search_embedding")
builder.add_edge("node_multi_search", "node_search_embedding_hyde")
builder.add_edge("node_multi_search", "node_web_search_mcp")
builder.add_edge("node_multi_search", "node_query_kg")

# 3. 4-way search branches -> Merge results at Join node
builder.add_edge("node_search_embedding", "node_join")
builder.add_edge("node_search_embedding_hyde", "node_join")
builder.add_edge("node_web_search_mcp", "node_join")
builder.add_edge("node_query_kg", "node_join")

# 4. Merge -> RRF Rank -> Rerank -> Generate Answer -> End
builder.add_edge("node_join", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

# Compile to generate executable Runnable application
query_app = builder.compile()