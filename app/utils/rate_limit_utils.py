# app/utils/rate_limit_utils.py
import time
from typing import Deque
from app.core.logger import logger  # Reuse the project's global logger


def apply_api_rate_limit(
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 60
) -> None:
    """
    General-purpose sliding window API rate limiter (extracted as a common utility).
    Core Logic: Maintains a double-ended queue (deque) of request timestamps. If the number of requests 
    within the window exceeds the limit, it automatically blocks and waits, preventing triggering third-party API rate limits.
    :param request_times: A double-ended queue storing request timestamps, initialized externally (global/singleton) and reused across calls.
    :param max_requests: The maximum allowed number of requests within the rate limit window.
    :param window_seconds: The duration of the sliding window for rate limiting, defaults to 60 seconds (1 minute).
    :return: None. It blocks and waits when the limit is exceeded.
    """
    current_time = time.time()

    # 1. Clean up expired request timestamps outside the sliding window to ensure the queue only contains requests within the current window.
    while request_times and current_time - request_times[0] >= window_seconds:
        request_times.popleft()

    # 2. If the request count within the window hits the upper limit, calculate and block-wait for the remaining time.
    if len(request_times) >= max_requests:
        # Calculate the duration needed to sleep (total window duration - age of the earliest request)
        sleep_duration = window_seconds - (current_time - request_times[0])
        if sleep_duration > 0:
            logger.debug(f"API rate limit triggered. Max {max_requests} requests per {window_seconds}s window. Waiting for: {sleep_duration:.2f} seconds.")
            time.sleep(sleep_duration)
            # Update the current time after waiting, and re-clean expired requests (in case any request expires during the wait period)
            current_time = time.time()
            while request_times and current_time - request_times[0] >= window_seconds:
                request_times.popleft()

    # 3. Record the current request timestamp and append it to the sliding window queue.
    request_times.append(current_time)
    logger.debug(f"API request timestamp recorded. Current request count within the {window_seconds}s window: {len(request_times)}")
    