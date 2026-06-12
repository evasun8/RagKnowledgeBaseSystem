import os
from neo4j import GraphDatabase

_neo4j_driver = None    
def get_neo4j_driver() -> GraphDatabase:
    """
    Retrieves the Neo4j driver instance.
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    return _neo4j_driver