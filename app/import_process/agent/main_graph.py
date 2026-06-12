# Load environment variables: Read configuration from .env file (such as Milvus address, KG service address, BGE model path, etc.)
from dotenv import load_dotenv
# Import core LangGraph dependencies: StateGraph (state graph), START/END (built-in start/end node constants)
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# Import custom state class: Unify and manage all data throughout the workflow (shared/modified across nodes)
from app.import_process.agent.state import ImportGraphState, create_default_state
# Import all custom business nodes: Each node corresponds to a specific step in the knowledge base import process
from app.import_process.agent.nodes.node_entry import node_entry  # Entry node: Initialize parameters, validate inputs
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF to MD: Parse PDF files into markdown format
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD Image Processing: Extract/download images in markdown, fix image paths
from app.import_process.agent.nodes.node_document_split import node_document_split  # Document Splitting: Split long documents into small segments matching model requirements
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # Item Name Recognition: Extract core item names from chunks (business customization)
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE Embedding: Convert text chunks into vector representations (compatible with Milvus vector database)
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # Import Milvus: Write vector data into the Milvus vector database


# Initialize environment variables: Must be executed before reading configurations, ensuring subsequent nodes can obtain config information from environment variables
load_dotenv()

# ===================== 1. Initialize LangGraph State Graph =====================
# Core: StateGraph is the core class of LangGraph, used to build stateful workflows
# Parameter ImportGraphState: Custom TypedDict type, defining the **full state fields** of the workflow
# Function: All nodes take this state object as an input parameter, and the key-value pairs returned by the nodes will be automatically merged back into the state, enabling data sharing between nodes
workflow = StateGraph(ImportGraphState)

# ===================== 2. Register All Business Nodes =====================
# Syntax: add_node("unique_node_identifier", node_function)
# Requirement: Node functions must receive the "state object" as an input parameter and return a dictionary (used to update the state)
# All nodes are registered in the chronological order of the "knowledge base import workflow". Node identifiers are kept consistent with function names for easy maintenance
workflow.add_node("node_entry", node_entry)  # Workflow Entry: Parameter initialization, input validation
workflow.add_node("node_pdf_to_md", node_pdf_to_md)  # PDF to MD: Preprocessing for non-MD format files
workflow.add_node("node_md_img", node_md_img)  # MD Image Processing: Ensure image accessibility in documents
workflow.add_node("node_document_split", node_document_split)  # Document Splitting: Solve the issue of large texts being unable to be embedded/inferred
workflow.add_node("node_item_name_recognition", node_item_name_recognition)  # Item Name Recognition: Customized business step, extract core business identifiers
workflow.add_node("node_bge_embedding", node_bge_embedding)  # BGE Embedding: Text -> Vector, preparing for Milvus storage
workflow.add_node("node_import_milvus", node_import_milvus)  # Vector Storage: Persist vector data into Milvus

# ===================== 3. Set Workflow Entry Point =====================
# Syntax: set_entry_point("node_identifier") -> Recommended approach, directly specifying the starting node of the workflow
# Equivalent syntax: workflow.add_edge(START, "node_entry") (START is a built-in start constant of LangGraph)
# Function: Specify the first node executed in the workflow, replacing manually adding an edge from START to the target node, making the code cleaner
workflow.set_entry_point("node_entry")

# ===================== 4. Define Conditional Routing Function (Branching logic after the entry node) =====================
# Core: Dynamically determine the subsequent execution path based on the configuration items in the state, implementing "PDF import" / "direct MD import" branches
# Requirement: Accepts the state object as input and returns the "target node identifier" or END (built-in end constant)
def route_after_entry(state: ImportGraphState) -> str:
    """
    Conditional routing logic after the entry node
    :param state: Full workflow state object, containing all configurations and intermediate results
    :return: Target node identifier / END, LangGraph will automatically jump to the corresponding node
    """
    # Branch 1: Enable direct MD import -> Skip PDF-to-MD, proceed directly to MD image processing
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    # Branch 2: Enable PDF import -> Execute PDF-to-MD, then follow the remaining workflow steps
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    # Branch 3: No import configuration enabled -> Terminate workflow directly (END is a built-in end constant of LangGraph)
    else:
        return END

# Register conditional edges: Bind the entry node to the routing function
# Syntax: add_conditional_edges("source_node_identifier", routing_function)
# Function: After the source node finishes execution, the routing function is called to dynamically jump to the target node based on the return value
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END
    }
)

# ===================== 5. Register Static Sequential Edges (Unified workflow after branch merging) =====================
# Core: All branches eventually merge into a "fixed-order execution workflow", running all the way from MD image processing to vector storage
# Syntax: add_edge("source_node_identifier", "target_node_identifier/END") -> Static edge, fixed routing relationship, no branching logic
workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF-to-MD completed -> MD image processing
workflow.add_edge("node_md_img", "node_document_split")  # MD processing completed -> Document splitting
workflow.add_edge("node_document_split", "node_item_name_recognition")  # Splitting completed -> Item name recognition
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # Item name recognition completed -> BGE embedding
workflow.add_edge("node_bge_embedding", "node_import_milvus")  # Embedding completed -> Import to Milvus vector database
workflow.add_edge("node_import_milvus", END)  # Milvus import completed -> Workflow execution finished (END is a built-in end node)

# ===================== 6. Compile Workflow into an Executable Object =====================
# Syntax: compile() -> Compile the workflow built by StateGraph into an executable LangGraph application
# Function: Generate a callable kb_import_app, which triggers the workflow execution via the invoke() method
# Features: Once compiled, it can be repeatedly called and supports passing in different initial states to execute multiple tasks
kb_import_app = workflow.compile()

if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
    import os

    # Full workflow test: Validate the entire pipeline from PDF import -> Milvus storage -> KG import
    logger.info("===== Starting Knowledge Graph Import Full Workflow Test =====")
    
    # 1. Construct test file path (reuse the project 'doc' directory, matching the pdf2md test file)
    test_pdf_name = os.path.join("docs", "m-welcome-to-security-cloud-control.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    
    # 2. Construct output directory (to store intermediate files like MD and images)
    test_output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)  # Create if it does not exist

    # 3. Verify whether the test PDF file exists
    if not os.path.exists(test_pdf_path):
        logger.error(f"Full workflow test failed: Test PDF file does not exist, path: {test_pdf_path}")
        logger.info("Please check the file path, or manually place the test file inside the 'doc' folder in the project root directory")
    else:
        # 4. Construct test state (aligned with practical business inputs, enabling PDF parsing)
        test_state = ImportGraphState({
            "task_id": "test_kg_import_workflow_001",  # Test task ID
            "user_id": "test_user",  # Test user ID
            "local_file_path": test_pdf_path,  # Test PDF file path
            "local_dir": test_output_dir,  # Intermediate file output directory
            "is_pdf_read_enabled": False,  # Enable PDF parsing (core switch)
            "is_md_read_enabled": False  # Disable MD parsing
        })
        try:
            logger.info(f"Test task started. PDF file path: {test_pdf_path}")
            logger.info(f"Intermediate file output directory: {test_output_dir}")
            logger.info("Starting execution of all workflow nodes in sequence: entry -> pdf2md -> md_img -> split -> item_name -> embedding -> milvus -> kg")

            # 5. Execute the full LangGraph workflow (streaming execution to print node progress)
            final_state = None
            for step in kb_import_app.stream(test_state, stream_mode="values"):
                # Print the currently completed node (streaming output is more intuitive)
                current_node = list(step.keys())[-1] if step else "Unknown Node"
                logger.info(f"✅ Node execution completed: {current_node}")
                final_state = step  # Save the final state

            # 6. Full workflow execution completed. Preview results and print core metrics
            if final_state:
                logger.info("-" * 80)
                logger.info("===== Full Workflow Test Executed Successfully! Core Results Preview =====")
                # Extract core result metrics
                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = final_state.get("md_content", "")[:150]  # First 150 characters of MD content
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False
                kg_id = final_state.get("kg_id", "Not Generated")  # ID generated by KG import (adjust based on actual business fields)

                # Print core metrics
                logger.info(f"📄 PDF to MD content preview (first 150 chars): {md_content}...")
                logger.info(f"📝 Total chunk count from document splitting: {chunk_count}")
                logger.info(f"🔍 Are all chunks vectorized: {'Yes' if has_embedding else 'No'}")
                logger.info(f"🗄️ Are all chunks stored in Milvus (with chunk_id): {'Yes' if has_chunk_id else 'No'}")
                logger.info(f"🧠 Knowledge Graph Import ID: {kg_id}")
                logger.info(f"📂 Core keys contained in the final state: {list(final_state.keys())}")
                logger.info("-" * 80)
        except Exception as e:
            # 7. Exception handling to print detailed error information
            logger.error(f"===== Full Workflow Test Failed =====", exc_info=True)
            logger.error(f"Failure reason: {str(e)}")
    logger.info("===== Knowledge Graph Import Full Workflow Test Ended =====")