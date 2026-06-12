# Import core dependencies: dataclasses, environment variable parser, and path utilities
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load the .env configuration file in advance (must be executed before reading env variables to ensure os.getenv works)
# If .env is not in the project root directory, specify the path: load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# Define the service configuration for LLM (Large Language Model)
@dataclass
class LLMConfig:
    base_url: str
    api_key : str
    lv_model: str
    llm_model: str
    llm_temperature: float

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    lv_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
)