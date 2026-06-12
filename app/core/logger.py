"""
Project Logging Utility
Based on loguru, supports .env configuration for dual console/file output, automatically generating logs/app_YYYYMMDD.log
Features:
1. Configuration-driven: Toggle output and modify log levels via .env file.
2. Automated Path: File logs are output by default to PROJECT_ROOT/logs/app_YYYYMMDD.log.
3. Auto-cleanup: Retain logs according to configuration and automatically delete expired files.
4. UTF-8 encoding
5. Async & Safe: Thread-safe queueing enabled, supports multi-threaded/asynchronous scenarios to avoid log interleaving.
6. Out-of-the-box: Directly import the logger into any module across the project for immediate use.
7. Ultra-precise Location: Penetrates loguru internals + utility class itself, perfectly displaying the actual calling position of the business module.
"""
import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger
 
 
# -------------------------- Step 1: Load .env Configuration File --------------------------
load_dotenv()
 
# -------------------------- Step 2: Read .env Configs (with default values to prevent missing configs) --------------------------
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")
 
# -------------------------- Step 3: Define Log Path (Automatically deduce project root) --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME
 
# -------------------------- Step 4: Define Log Format (Colored, structured, easy to read) --------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)
 
# -------------------------- Step 5: Initialize Log Configuration (Core Method) --------------------------
def init_logger():
    """
    Initialize global log configuration
    1. Remove loguru default console output (to avoid duplicate printing)
    2. Toggle console output based on .env configuration
    3. Toggle file output based on .env configuration (automatically create logs folder)
    4. Configure log format, level, rotation, and retention policies
    :return: Configured loguru logger instance
    """
    # 1. Remove the default console output of loguru
    logger.remove()
 
    # 2. Configure console output (if enabled in .env)
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True
        )
 
    # 3. Configure file output (if enabled in .env)
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
 
    return logger
 
# -------------------------- Step 6: Initialize and Ultra-fix Global Logger --------------------------
base_logger = init_logger()
 
def fix_log_position(record):
    """Traverse the call stack, skip loguru internal frames + utility class frames, and extract the actual call position of the business code."""
    for frame in inspect.stack():
        # Ultimate filtering: exclude loguru internals + exclude the utility class logger.py itself to locate the business module directly
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # Update log fields to the actual location of the business code
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break
 
# Apply the ultimate fix and export the globally available logger
logger = base_logger.patch(fix_log_position)
 
# -------------------------- Test Code (Verify the fix effect) --------------------------
if __name__ == '__main__':
    logger.info("[Test] Internal call within logger.py (Test only, business module calls will show the correct filename)")
    logger.error("[Test] Program error occurred!!!")
    print(f"Log file output path: {LOG_FILE_PATH}")