"""
JSON Formatting Utility Module

Provides unified JSON serialization and formatting functionalities to ensure consistency of JSON outputs across the project.
"""

import json
from typing import Any, Dict


def format_state(state: Dict[str, Any], indent: int = 4) -> str:
    """
    Specifically designed for formatting workflow states (ImportGraphState).

    Args:
        state: ImportGraphState workflow state dictionary.
        indent: Number of spaces for JSON indentation, defaults to 4.

    Returns:
        The formatted JSON string.

    Example:
        >>> state = {"task_id": "001", "pdf_path": "test.pdf"}
        >>> print(format_state(state))
        {
            "task_id": "001",
            "pdf_path": "test.pdf"
        }
    """

    return json.dumps(state, indent=indent, ensure_ascii=False)


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:
    """
    General-purpose JSON formatting function.

    Args:
        data: Data to be formatted (serializable objects such as dicts, lists, etc.).
        indent: Number of spaces for JSON indentation, defaults to 4.
        ensure_ascii: Whether to escape non-ASCII characters, defaults to False.

    Returns:
        The formatted JSON string.

    Example:
        >>> data = {"name": "Test", "value": 123}
        >>> print(format_json(data))
        {
            "name": "Test",
            "value": 123
        }
    """
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)