
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
 
from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app
 
# The workflow graph instance will be imported downstream
# from app.query_process.main_graph import query_app
 
 
# Define FastAPI application instance
app = FastAPI(title="Query Service", description="Shopkeeper Brain Trust Query Service!")


# CORS middleware configuration to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
async def health():
    """
    Checks if the query service is healthy
    """
    return {"ok": True}

# Returns the chat.html static page
@app.get("/chat.html")
async def chat():
    # Retrieve page path relative to the api -> query_process directory structure
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    # Define location of chat.html
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail=f"Page not found at path: {chat_html_path}!")
    return FileResponse(chat_html_path)

# Define data validation schema for incoming requests
class QueryRequest(BaseModel):
    """Query request payload structure"""
    query: str = Field(..., description="Query content")  # Required parameter
    session_id: str = Field(None, description="Session ID")
    is_stream: bool = Field(False, description="Whether to return a stream")


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    Main Query Processing Endpoint:
    1. Parse incoming parameters
    2. Update task status inside the memory manager
    3. Trigger LangGraph execution
    4. Return task initialization responses
    :param background_tasks: FastAPI BackgroundTasks object for asynchronous runoffs
    :param request: QueryRequest validated payload
    :return: dict status metadata
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    # Check whether streaming output is requested
    is_stream = request.is_stream
    if is_stream:
        # Create an SSE queue mapped to the session_id
        create_sse_queue(session_id)
    # Update global task status as: PROCESSING
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    print(f"Processing workflow started. Is stream: {is_stream}, Parameters: query='{user_query}', session_id='{session_id}'")
 
    if is_stream:
        # For streams, dispatch run_query_graph as a background task and immediately return session metadata
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        print("Result processing initiated (Asynchronous)...")
        return {
            "message": "Result is being processed...",
            "session_id": session_id
        }
    else:
        # Synchronous execution
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        return {
            "message": "Processing completed!",
            "session_id": session_id,
            "answer": answer,
            "done_list": []
        }
    
# Asynchronous graph executor function
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"Starting graph workflow processing... {session_id} {user_query} {is_stream}")
 
    default_state = {"original_query": user_query, "session_id": session_id, "is_stream": is_stream}
    try:
        # Run compiled LangGraph workflow app
        query_app.invoke(default_state)
        # Update task status on complete. State and database modifications are handled inside graph nodes.
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"Workflow execution exception: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    print(f"Streaming endpoint invoked for session: {session_id}")
    """
    Establishes real-time Server-Sent Events (SSE) connection to stream data back to the client
    """
    try:
        return StreamingResponse(
            sse_generator(session_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        print(f"Streaming failed: {e}")
        raise e

@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    Retrieves conversational logs and chat history for the current session
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {e}")
    
@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Deletes the current chat session's dialogue history from Mongo
    """
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}

if __name__ == "__main__":
    # Start the server locally using uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)