# Import core dependencies: dataclasses, environment variable parser, and path utilities
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load the .env configuration file in advance (must be executed before reading env variables to ensure os.getenv works)
# If .env is not in the project root directory, specify the path: load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# Define the service configuration for MinerU (Document Parsing & Extraction Service)
@dataclass
class MineruConfig:
    base_url: str
    api_key : str

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_key=os.getenv("MINERU_API_TOKEN")
)