# Import Python built-in modules: used for reading environment variables
import os
# Import logging module: used to log application execution (success/failure/error messages)
import logging
# Import type hinting module: used for function parameters/return value type hints, improving code readability and standardization
from typing import List, Dict, Any, Optional
# Import datetime module: used to generate timestamps for recording dialogue creation times
from datetime import datetime
# Import core pymongo modules: Native Python driver for MongoDB, implementing database connections and operations
# ASCENDING: represents ascending order, used for MongoDB index definitions and query sorting
from pymongo import MongoClient, ASCENDING
# Import ObjectId from bson: MongoDB's default primary key type, used to uniquely identify documents
from bson import ObjectId
# Import dotenv module: used to load environment variables from a .env file, avoiding hardcoded sensitive configs (e.g., MongoDB connection URI)
from dotenv import load_dotenv

# Load environment variables from the .env file, allowing os.getenv to read configurations
load_dotenv()


class HistoryMongoTool:
    """
    MongoDB Dialogue History Read/Write Utility Class (Implemented via native PyMongo).
    Core Functionality: Encapsulates MongoDB connection, collection initialization, and index creation, 
                        providing a unified database operation entryway for upper-layer logic.
    Extended Functionality: Supports format conversion with LangChain message objects (reserved capability from legacy code).
    """

    def __init__(self):
        """
        Class initialization method: completes MongoDB connection, database/collection retrieval, and index creation.
        Throws an exception and logs an error upon initialization failure, ensuring the application is aware of connection issues.
        """
        try:
            # Read MongoDB connection URI from environment variables (sensitive config, not hardcoded)
            self.mongo_url = os.getenv("MONGO_URL")
            # Read the target database name from environment variables
            self.db_name = os.getenv("MONGO_DB_NAME")

            # Create MongoDB client instance, establishing connection with the database
            self.client = MongoClient(self.mongo_url)
            # Retrieve the specified database object
            self.db = self.client[self.db_name]
            # Retrieve the chat history collection (equivalent to a table in relational DBs), collection name: chat_message
            self.chat_message = self.db["chat_message"]

            # Create a compound index for the chat_message collection to optimize query performance
            # Index Rule: session_id ASC + ts DESC, tailored for the core query scenario: "fetch the latest records by session"
            # create_index is idempotent by nature: it will not recreate the index if it already exists, eliminating extra checks
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            # Log success message confirming database connection and initialization completion
            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            # Catch all initialization exceptions and log detailed error messages
            logging.error(f"Failed to connect to MongoDB: {e}")
            # Re-raise the exception so the caller senses the initialization failure, preventing usage of an uninitialized instance
            raise


def clear_history(session_id: str) -> int:
    """
    Clears all historical dialogue records for a specified session.
    :param session_id: Unique session identifier used to filter records for deletion.
    :return: The actual number of deleted documents; returns 0 if deletion fails.
    """
    # Retrieve the global HistoryMongoTool instance, using the singleton pattern to avoid duplicate DB connection creation
    mongo_tool = get_history_mongo_tool()
    try:
        # Execute batch deletion: delete all documents matching the session_id
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        # Log deletion success, including the deleted count and session ID for easier troubleshooting
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        # Return the actual deleted count (delete_many's return object contains the deleted_count attribute)
        return result.deleted_count
    except Exception as e:
        # Catch deletion exceptions and log error messages including the session ID
        logging.error(f"Error clearing history for session {session_id}: {e}")
        # Return 0 on exception, indicating deletion failure
        return 0


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        message_id: str = None
) -> str:
    """
    Writes or updates a single session record in MongoDB.
    Supports two modes: inserts a new record when message_id is absent; updates an existing record when message_id is provided.
    :param session_id: Unique session identifier linking the dialogue to its respective session.
    :param role: Message role, fixed values: 'user' or 'assistant'.
    :param text: Core dialogue content, either the user's query or the assistant's response.
    :param rewritten_query: Rewritten query string (optional, used in scenarios like retrieval-augmented generation, defaults to empty string).
    :param item_names: Associated product name list (optional, supports multiple products, defaults to None).
    :param message_id: Primary key ID of the record (optional, triggers update if provided, inserts if absent).
    :return: Unique identifier of the inserted/updated record (returns ObjectId string for insertion, returns passed message_id for update).
    """
    # Generate the current timestamp (second-level resolution) to record message creation time, used later for sorting and queries
    ts = datetime.now().timestamp()

    # Construct the document data to be inserted/updated (MongoDB's basic data unit is a document, similar to a Python dict)
    document = {
        "session_id": session_id,  # Session ID, relationship dimension
        "role": role,  # Message role
        "text": text,  # Message content
        "rewritten_query": rewritten_query or "",  # Rewritten query, fallbacks empty values to an empty string
        "item_names": item_names,  # Associated product name list
        "ts": ts  # Timestamp, sorting and time-filtering dimension
    }

    # Retrieve the global HistoryMongoTool singleton instance
    mongo_tool = get_history_mongo_tool()
    # Check if a primary key ID is passed to distinguish between update and insert logic
    if message_id:
        # message_id present: execute update operation (matching by primary key)
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},  # Update condition: primary key match (must convert string to ObjectId type)
            {"$set": document}  # Update operation: $set modifies only specified fields, preserving other fields
        )
        # Return the passed message_id for update operations
        return message_id
    else:
        # message_id absent: execute insertion operation
        result = mongo_tool.chat_message.insert_one(document)
        # Convert the inserted ObjectId to a string, easing upper-layer consumption (prevents returning raw ObjectId object)
        return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    Batch updates the associated product names of historical dialogue records.
    Only updates records meeting the criteria: primary key is in the specified list, and item_names is empty/non-existent/None.
    :param ids: List of record primary key IDs to update (string type).
    :param item_names: List of new product names to set.
    :return: The actual number of updated documents; returns 0 if update fails.
    """
    # Retrieve the global HistoryMongoTool singleton instance
    mongo_tool = get_history_mongo_tool()
    try:
        # Convert string primary keys to MongoDB's ObjectId type (primary keys in the DB are stored as ObjectIds)
        object_ids = [ObjectId(i) for i in ids]
        # Execute batch update operation
        result = mongo_tool.chat_message.update_many(
            # Update condition: compound criteria, all must be met simultaneously
            {
                "_id": {"$in": object_ids},  # Primary key is within the specified ID list (batch filtering)
                "$or": [  # Meets any of the following conditions: item_names is unset/empty
                    {"item_names": {"$exists": False}},  # item_names field does not exist
                    {"item_names": []},  # item_names is an empty list
                    {"item_names": None}  # item_names is None
                ]
            },
            {"$set": {"item_names": item_names}}  # Update operation: set the new product name list
        )
        # Log update success, including modified count and the new product names
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        # Return the actual updated count (modified_count: documents truly modified, distinct from matched_count)
        return result.modified_count
    except Exception as e:
        # Catch batch update exceptions and log error messages
        logging.error(f"Error updating history item_names: {e}")
        # Return 0 on exception, indicating update failure
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Queries the recent N dialogue records for a specified session, returning raw dictionary format.
    Results are sorted in chronological ascending order, ready to be fed directly into an LLM context window.
    :param session_id: Unique session identifier used to filter records of the specified session.
    :param limit: Record count constraint, defaults to returning the recent 10 records.
    :return: List of dialogue records (dictionary format); returns an empty list if query fails.
    """
    # Retrieve the global HistoryMongoTool singleton instance
    mongo_tool = get_history_mongo_tool()
    try:
        # Construct query condition: fetch records for specified session_id only
        query = {"session_id": session_id}

        # Execute query: sort by timestamp ascending, constrain return record count
        # find(query): gets a cursor matching the criteria (lazy loading, does not fetch immediately)
        # sort("ts", ASCENDING): sorts by 'ts' field ascending (oldest to newest), matching LLM context sequence requirements
        # limit(limit): constrains the maximum number of returned records
        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        # Convert cursor to a list, triggering actual database query to fetch all matching documents
        messages = list(cursor)

        # Return the list of fetched records
        return messages
    except Exception as e:
        # Catch query exceptions and log error messages
        logging.error(f"Error getting recent messages: {e}")
        # Return an empty list on exception, preventing upper-layer processing from blowing up on NoneType
        return []


# Define global variable: stores the singleton instance of HistoryMongoTool
# Purpose: Prevents multiple instantiations of HistoryMongoTool, thereby avoiding duplicate connection establishments to MongoDB
_history_mongo_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """
    Retrieves the singleton instance of HistoryMongoTool (Lazy Loading Pattern).
    Core Logic: Instantiates the class if the global reference is empty; returns it directly if populated, 
                guaranteeing a single database connection instance across the application lifecycle.
    :return: Singleton instance of HistoryMongoTool.
    """
    # Declare usage of the global variable to prevent scoping it as local within the function
    global _history_mongo_tool
    # Lazy initialization: create new instance only if the global variable is unassigned
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    # Return the singleton instance
    return _history_mongo_tool


# Attempt to initialize the singleton instance upon module loading for pre-loading purposes
# Objective: Advance database connection initialization to the module load phase, avoiding latency spikes 
# during the very first API call (improving cold-start/first-response latency)
try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    # Log a warning message instead of crashing the process if initialization fails on module load
    # Reason: Exceptions raised during module loading can stall the entire application startup; 
    # lazy loading via get_history_mongo_tool acts as a fallback to retry instantiation on-demand.
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")

# Main execution entry: executes simple functional testing only when the script is run directly
if __name__ == "__main__":
    # Simple test suite: validates database write and query capabilities
    # Test session ID used to identify testing dialogue records
    sid = "000015_hybrid"
    # 1. Write user message (manually triggering insertion)
    save_chat_message(sid, "user", "Hello (Hybrid)")
    # 2. Write assistant response (mocking a sequential response following the user query)
    save_chat_message(sid, "assistant", "Hello! I am an assistant powered by native Mongo + LangChain objects.")
    # 3. Write user query with associated product metadata (testing the item_names field logic)
    save_chat_message(sid, "user", "How do I swap the battery for this multimeter?", item_names=["Hybrid Multimeter"])

    # 4. Query the recent 5 records of the specified session to validate query execution
    print("--- Querying LangChain Object Records ---")
    messages = get_recent_messages(sid, limit=5)
    # Print the number of retrieved records
    print(f"Retrieved record count: {len(messages)}")
    # Iterate and print the detailed content of each record
    for m in messages:
        print(f" {m}  ")