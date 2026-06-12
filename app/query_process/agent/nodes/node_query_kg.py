import time 
import sys 
from app.utils.task_utils import add_running_task, add_done_task 

def node_query_kg(state): 
    """ 
    Node Function: Queries entity relationships within the Neo4j knowledge graph. 
    """ 
    print("=== node_query_kg Graph Query Processing ===") 
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream")) 
    time.sleep(1) 
    # ... 
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))