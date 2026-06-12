# app/utils/path_utils.py
from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path

def get_path_dir(ps: int = 0) -> Path:
    """
    pathlib.Path provides the `parents` attribute, which is an ordered sequence of the path's logical ancestors. 
    Accessing it via an index allows quick retrieval of the "N-th parent directory", perfectly resolving the tediousness 
    of chaining multiple `.parent` calls. This is also the officially recommended shorthand approach!
    
    Core Rules: The index of `parents[N]` corresponds to the "number of levels upward"
    parents[0] -> Equivalent to .parent (the immediate parent directory, 1 level up)
    parents[1] -> Equivalent to .parent.parent (2 levels up)
    parents[2] -> Equivalent to .parent.parent.parent (3 levels up)
    And so forth. parents[N] -> Directly retrieves the (N+1)-th parent directory; the larger the index, the higher the hierarchy level.
    :param ps: Index of the parent level to retrieve
    :return: Path object of the target directory
    """
    dir_path = Path(__file__).parents[ps]
    return dir_path


def get_project_root(identifier: str = ".env") -> Path:
    # Step 1: Prioritize reading from environment variables (for production environments)
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    # Step 2: Load the .env file at the root directory (for subsequent logic, can also be omitted)
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    # Step 3: Recursively look up the identifier (Fallback mechanism, for development environments)
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(f"Project root identifier '{identifier}' not found, and environment variable PROJECT_ROOT is not configured.")


PROJECT_ROOT = get_project_root(".env")