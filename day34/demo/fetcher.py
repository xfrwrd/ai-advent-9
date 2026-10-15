"""Demo HTTP fetcher used by the Day 34 documentation scenario."""

import time

import requests

DEFAULT_TIMEOUT = 180
MAX_RETRIES = 3


def fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.get(url, timeout=timeout).text
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(attempt)
    return ""
