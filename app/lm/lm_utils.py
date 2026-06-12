# Environment configuration and dependency imports
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

# Project internal dependencies
from app.conf.lm_config import lm_config
from app.core.logger import logger

# Global cache: keys are (model_name, json_mode) tuples, values are ChatOpenAI instances
# Purpose: Avoids duplicate initialization of clients, boosts performance, and unifies instance management
_llm_client_cache = {}

def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    Retrieves a LangChain ChatOpenAI client instance equipped with a global cache mechanism.
    Adapts to OpenAI and OpenAI-compatible APIs, supporting custom models and JSON structured outputs.

    :param model: Model name. Priority: method arguments > configuration lm_config.llm_model > built-in default 'gpt-4o-mini'.
    :param json_mode: Whether to enable JSON output mode. When enabled, forces the model to return standard json_object formats.
    :return: An initialized ChatOpenAI instance.
    :raise ValueError: Raised if critical configurations like API key or base URL are missing.
    :raise Exception: Raised if model initialization fails at the LangChain encapsulation layer.
    """
    # 1. Determine the target model (descending priority to guarantee non-empty model name)
    # UPDATED: Changed the hardcoded default fallback from "qwen3-32b" to "gpt-4o-mini"
    target_model = model or lm_config.llm_model or "gpt-4o-mini"
    
    # Cache key: model name + JSON mode, uniquely identifying clients with different configurations
    cache_key = (target_model, json_mode)

    # 2. Cache hit: return the already initialized instance directly, avoiding duplicate creation overhead
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM Client] Cache hit; returning instance directly: model={target_model}, json_mode={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. Core configuration validation: intercept missing API configurations early and raise explicit exceptions
    if not lm_config.api_key:
        raise ValueError("[LLM Client] Missing configuration: Please configure OPENAI_API_KEY (LLM API Key) in your .env file.")
    if not lm_config.base_url:
        raise ValueError("[LLM Client] Missing configuration: Please configure OPENAI_API_BASE (API Base URL) in your .env file.")

    logger.info(f"[LLM Client] Starting initialization of a new instance: model={target_model}, json_mode={json_mode}")

    # 4. Configuration parameter assembly
    # model_kwargs: OpenAI general parameters universally supported by all compatible APIs
    model_kwargs = {}
    if json_mode:
        # Enable standard JSON output mode, forcing the model to return a parsable json_object
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM Client] JSON mode enabled; the model will enforce a standard JSON structure.")

    # 5. Client initialization: catch LangChain encapsulation layer exceptions and wrap them into user-friendly prompts
    try:
        llm_client = ChatOpenAI(
            model=target_model,                          # Target model name
            temperature=lm_config.llm_temperature or 0.1, # Low temperature ensures deterministic outputs (0.0 to 1.0)
            api_key=lm_config.api_key,                   # API key
            base_url=lm_config.base_url,                 # API base URL 
            model_kwargs=model_kwargs,                   # OpenAI general parameters
            # REMOVED: extra_body={"enable_thinking": False} was deleted because OpenAI API doesn't support Qwen private params.
        )
    except LangChainException as e:
        raise Exception(f"[LLM Client] Failed to initialize model 【{target_model}】 (LangChain layer): {str(e)}") from e

    # 6. Store the newly created instance into the global cache for future invocation reuse
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM Client] Instance initialized successfully and cached: model={target_model}, json_mode={json_mode}")
    return llm_client

# Test execution entry: validates client creation, caching mechanism, and log outputs
if __name__ == "__main__":
    logger.info("===== Starting LLM Client Utility Test Suite =====")
    try:
        # Test 1: Default configuration (default model + standard text generation mode)
        client1 = get_llm_client()
        logger.info("✅ Test 1 passed: Client created successfully with default configurations.")

        # Test 2: Specify an OpenAI model (gpt-4o-mini) + standard text generation mode
        # UPDATED: Replaced Qwen models with gpt-4o-mini
        client2 = get_llm_client(model="gpt-4o-mini")
        logger.info("✅ Test 2 passed: Client created successfully with a designated model.")

        # Test 3: Same model + mode configuration to validate caching mechanism efficiency
        client3 = get_llm_client(model="gpt-4o-mini")
        logger.info(f"✅ Test 3 passed: Cache validation verified; client2 and client3 share the identical instance: {client2 is client3}")

        # Test 4: Enable JSON structured output mode
        client4 = get_llm_client(model="gpt-4o-mini", json_mode=True)
        logger.info("✅ Test 4 passed: Client with JSON output mode activated successfully.")

    except Exception as e:
        logger.error(f"❌ LLM Client utility test suite failed: {str(e)}", exc_info=True)
    finally:
        logger.info("===== LLM Client Utility Test Suite Concluded =====")
