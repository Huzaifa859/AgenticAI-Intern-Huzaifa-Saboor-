"""
CONSTANTS.py
============

Centralized configuration and constants for the OpenRouter CLI Chatbot.

Keeping constants (model IDs, API settings, ANSI color codes, etc.) in one
dedicated module makes them easy to find and update without digging through
the main application logic
"""

from typing import Dict

# --------------------------------------------------------------------------- #
# OpenRouter model configuration
# --------------------------------------------------------------------------- #

# The dictionary key is the number the user types at the selection menu;
# the value is the exact OpenRouter model identifier used in API requests.
MODELS: Dict[str, str] = {
    "1": "meta-llama/llama-3.1-8b-instruct",
    "2": "google/gemma-3-27b-it",
    "3": "nvidia/nemotron-nano-9b-v2:free",
}

# --------------------------------------------------------------------------- #
# API configuration
# --------------------------------------------------------------------------- #

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0

# Optional headers that let your app show up on the OpenRouter leaderboards.
# They are not required for the API to function and can be safely edited.
APP_REFERER = "https://github.com/Huzaifa859/AgenticAI-Intern-Huzaifa-Saboor-"
APP_TITLE = "Huzaifa Week 2 OpenRouter CLI Chatbot"


# --------------------------------------------------------------------------- #
# Terminal colors
# --------------------------------------------------------------------------- #

class Colors:
    """ANSI escape codes for simple, dependency-free terminal colours."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"