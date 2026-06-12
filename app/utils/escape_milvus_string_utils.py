# ===================== Core Utility Functions =====================
def escape_milvus_string(value: str) -> str:
    """
    Dedicated string safety escape function for Milvus filter expressions.
    Core Purpose:
        Prevents Milvus from throwing errors when parsing filter_expr due to special characters in the raw string, 
        ensuring normal execution of CRUD operations.
    Escaping Rules:
        1. Backslash (\) -> Double backslash (\\\\): Milvus expression escaping rule.
        2. Double quote (") -> Escaped double quote (\\"): Prevents truncation of string expressions.
        3. Newlines/Carriage returns/Tabs -> Spaces: Prevents parsing failures caused by line breaks in the expression.
    Parameters:
        value: The raw string to be escaped (e.g., product name, file title).
    Returns:
        str: The escaped safe string, which can be directly used in Milvus filter_expr.
    """
    if value is None:
        return ""
    # Ensure the input is of string type to prevent errors from non-string values
    s = str(value)
    # Escape special characters according to Milvus rules
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    # Replace carriage returns/newlines/tabs with spaces to ensure the expression remains valid on a single line
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s