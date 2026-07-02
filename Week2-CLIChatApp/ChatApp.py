#!/usr/bin/env python3
"""
ChatApp.py
===================

A command-line chat application that talks to Large Language Models
through the OpenRouter API (https://openrouter.ai).

Features
--------
* Choose from 3 preconfigured OpenRouter models at startup.
* Switch models at any time during the session with ``/model``.
* Maintains full conversation history/context for the session.
* Displays response time and token usage for every reply.
* Gracefully handles network errors, invalid API keys, and rate limits
  (HTTP 429) using exponential backoff retries.
* Simple built-in commands: help, clear, history, exit.
* Lightweight ANSI colour output (no extra dependency required).

Author: Internship Project - Week 2
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv


# --------------------------------------------------------------------------- #
# Constants & configuration
# --------------------------------------------------------------------------- #

# The three models this internship task requires. The dictionary key is the
# number the user types at the selection menu; the value is the exact
# OpenRouter model identifier used in API requests.
MODELS: Dict[str, str] = {
    "1": "meta-llama/llama-3.1-8b-instruct",
    "2": "google/gemma-3-27b-it",
    "3": "nvidia/nemotron-nano-9b-v2:free",
}

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0

# Optional headers that let your app show up on the OpenRouter leaderboards.
# They are not required for the API to function and can be safely edited.
APP_REFERER = "https://github.com/Huzaifa859/AgenticAI-Intern-Huzaifa-Saboor-"
APP_TITLE = "Huzaifa Week 2 OpenRouter CLI Chatbot"


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


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class OpenRouterAPIError(Exception):
    """Raised for any non-recoverable error returned by the OpenRouter API."""


# --------------------------------------------------------------------------- #
# API key / environment handling
# --------------------------------------------------------------------------- #

def load_api_key() -> str:
    """
    Load the OpenRouter API key from a local ``.env`` file.

    The key is never hardcoded in source code. If it is missing, the
    program prints a clear, actionable error message and exits instead of
    crashing with a raw traceback.

    Returns:
        The API key as a string.
    """
    load_dotenv()  # Reads variables from a .env file in the working directory
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print(f"{Colors.RED}{Colors.BOLD}ERROR:{Colors.RESET} "
              f"{Colors.RED}OPENROUTER_API_KEY not found.{Colors.RESET}")
        print("Please create a '.env' file in the project root "
              "(see '.env.example') and set:")
        print(f"  {Colors.DIM}OPENROUTER_API_KEY=sk-or-v1-your_api_key_here"
              f"{Colors.RESET}")
        sys.exit(1)

    return api_key


# --------------------------------------------------------------------------- #
# OpenRouter API client
# --------------------------------------------------------------------------- #

class OpenRouterClient:
    """Thin wrapper around the OpenRouter chat completions endpoint."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": APP_REFERER,
            "X-Title": APP_TITLE,
        }

    def send_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_retries: int = MAX_RETRIES,
    ) -> Tuple[dict, float]:
        """
        Send a chat completion request to OpenRouter, retrying on rate limits.

        Args:
            model: OpenRouter model identifier (e.g. "meta-llama/llama-3.1-8b-instruct").
            messages: Full conversation history as a list of
                ``{"role": ..., "content": ...}`` dictionaries.
            max_retries: Number of retries on HTTP 429 (rate limit) before
                giving up.

        Returns:
            A tuple of ``(response_json, elapsed_seconds)``.

        Raises:
            OpenRouterAPIError: On network failures or non-recoverable
                HTTP error responses (invalid key, missing model, etc.).
        """
        payload = {"model": model, "messages": messages}
        backoff = INITIAL_BACKOFF_SECONDS
        attempt = 0

        while True:
            start = time.time()
            try:
                response = requests.post(
                    OPENROUTER_API_URL,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.exceptions.Timeout as exc:
                raise OpenRouterAPIError(
                    "The request timed out. Check your internet connection "
                    "and try again."
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                raise OpenRouterAPIError(
                    "Could not connect to OpenRouter. Check your internet "
                    "connection."
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise OpenRouterAPIError(
                    f"An unexpected network error occurred: {exc}"
                ) from exc

            elapsed = time.time() - start

            if response.status_code == 200:
                return response.json(), elapsed

            if response.status_code == 429:
                # Rate limited - back off and retry, honouring Retry-After
                # if the server provides one.
                attempt += 1
                if attempt > max_retries:
                    raise OpenRouterAPIError(
                        "Rate limit exceeded and the maximum number of "
                        "retries was reached. Please wait a moment before "
                        "trying again."
                    )
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else backoff
                print(
                    f"{Colors.YELLOW}Rate limited by OpenRouter. Retrying "
                    f"in {wait_time:.0f}s (attempt {attempt}/"
                    f"{max_retries})...{Colors.RESET}"
                )
                time.sleep(wait_time)
                backoff *= 2  # Exponential backoff
                continue

            # Any other non-200 status is treated as a hard failure.
            message = self._extract_error_message(response)
            if response.status_code == 401:
                raise OpenRouterAPIError(
                    "Invalid API key (401 Unauthorized). Double-check "
                    "OPENROUTER_API_KEY in your .env file."
                )
            if response.status_code == 402:
                raise OpenRouterAPIError(
                    "Payment required (402). Your OpenRouter account may "
                    "be out of credits."
                )
            if response.status_code == 404:
                raise OpenRouterAPIError(
                    f"Model '{model}' was not found on OpenRouter (404)."
                )
            raise OpenRouterAPIError(
                f"OpenRouter API returned status {response.status_code}: "
                f"{message}"
            )

    @staticmethod
    def _extract_error_message(response: requests.Response) -> str:
        """Pull a human-readable error message out of an API response."""
        try:
            data = response.json()
            return data.get("error", {}).get("message", response.text)
        except ValueError:
            return response.text or "No additional error details provided."


# --------------------------------------------------------------------------- #
# Conversation state
# --------------------------------------------------------------------------- #

class ChatSession:
    """Holds the conversation history and current model for one session."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.history: List[Dict[str, str]] = []
        self.total_tokens_used = 0
        self.message_count = 0

    def add_user_message(self, content: str) -> None:
        self.history.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def add_assistant_message(self, content: str) -> None:
        self.history.append(
            {
                "role": "assistant",
                "content": content,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def remove_last_message(self) -> None:
        """Roll back the last message (used when a request fails)."""
        if self.history:
            self.history.pop()

    def clear(self) -> None:
        self.history = []
        self.message_count = 0

    def switch_model(self, model: str) -> None:
        self.model = model

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Return history stripped of the local-only 'timestamp' field."""
        return [
            {"role": m["role"], "content": m["content"]} for m in self.history
        ]


# --------------------------------------------------------------------------- #
# CLI presentation helpers
# --------------------------------------------------------------------------- #

def print_banner() -> None:
    """Display a welcome banner when the application starts."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════╗
║             OpenRouter CLI Chatbot  v1.0              ║
╚══════════════════════════════════════════════════════╝{Colors.RESET}
{Colors.DIM}A lightweight terminal chat client for OpenRouter models.
Type '/help' at any time to see available commands.{Colors.RESET}
"""
    print(banner)


def print_help() -> None:
    """Display the list of available in-chat commands."""
    y, r, b = Colors.YELLOW, Colors.RESET, Colors.BOLD
    help_text = f"""
{b}Available Commands:{r}
  {y}help{r} / {y}/help{r}        Show this help message
  {y}exit{r} / {y}quit{r}        Exit the application
  {y}clear{r} / {y}/clear{r}      Clear the conversation history
  {y}history{r} / {y}/history{r}    Show full conversation history
  {y}/model{r}            Switch to a different model

Anything else you type is sent to the model as a chat message.
"""
    print(help_text)


def print_history(session: ChatSession) -> None:
    """Print every message exchanged so far in the current session."""
    if not session.history:
        print(f"{Colors.DIM}No conversation history yet.{Colors.RESET}\n")
        return

    print(f"\n{Colors.BOLD}--- Conversation History "
          f"({session.model}) ---{Colors.RESET}")
    for msg in session.history:
        if msg["role"] == "user":
            label, color = "You", Colors.CYAN
        else:
            label, color = "Assistant", Colors.BLUE
        print(f"{color}[{msg['timestamp']}] {label}:{Colors.RESET} "
              f"{msg['content']}")
    print(f"{Colors.BOLD}--- End of History ---{Colors.RESET}\n")


def prompt_model_selection(current: Optional[str] = None) -> str:
    """
    Display the model menu and prompt the user until a valid choice is made.

    Args:
        current: The currently active model identifier, if any. Used to
            mark it in the menu when switching models mid-session.

    Returns:
        The dictionary key ("1", "2", or "3") of the chosen model.
    """
    print(f"\n{Colors.BOLD}Available Models:{Colors.RESET}")
    for key, model_name in MODELS.items():
        marker = ""
        if model_name == current:
            marker = f" {Colors.GREEN}(current){Colors.RESET}"
        print(f"  {Colors.YELLOW}{key}{Colors.RESET}. {model_name}{marker}")

    while True:
        choice = input(
            f"\n{Colors.CYAN}Select a model (1-{len(MODELS)}): "
            f"{Colors.RESET}"
        ).strip()
        if choice in MODELS:
            return choice
        print(f"{Colors.RED}Invalid choice. Please enter one of: "
              f"{', '.join(MODELS.keys())}{Colors.RESET}")


# --------------------------------------------------------------------------- #
# Main application loop
# --------------------------------------------------------------------------- #

def handle_chat_turn(client: OpenRouterClient, session: ChatSession, user_input: str) -> None:
    """Send one user message to the model and print the reply, timing, and usage."""
    session.add_user_message(user_input)
    print(f"{Colors.DIM}Waiting for response from {session.model}...{Colors.RESET}")

    try:
        result, elapsed = client.send_chat(session.model, session.get_messages_for_api())
    except OpenRouterAPIError as exc:
        print(f"{Colors.RED}Error: {exc}{Colors.RESET}\n")
        session.remove_last_message()  # Don't keep a message that never got a reply
        return

    try:
        reply = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print(f"{Colors.RED}Error: Received an unexpected response format "
              f"from the API.{Colors.RESET}\n")
        session.remove_last_message()
        return

    session.add_assistant_message(reply)
    session.message_count += 1

    print(f"\n{Colors.BLUE}{Colors.BOLD}{session.model}{Colors.RESET} "
          f"{Colors.DIM}({elapsed:.2f}s){Colors.RESET}:")
    print(f"{reply}\n")

    usage = result.get("usage", {})
    if usage:
        prompt_tokens = usage.get("prompt_tokens", "N/A")
        completion_tokens = usage.get("completion_tokens", "N/A")
        total_tokens = usage.get("total_tokens", "N/A")
        print(f"{Colors.DIM}Tokens -> prompt: {prompt_tokens}, "
              f"completion: {completion_tokens}, "
              f"total: {total_tokens}{Colors.RESET}\n")
        if isinstance(total_tokens, int):
            session.total_tokens_used += total_tokens


def main() -> None:
    """Entry point: sets up the session and runs the interactive REPL loop."""
    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    print_banner()
    model_key = prompt_model_selection()
    session = ChatSession(MODELS[model_key])

    print(f"\n{Colors.GREEN}Chat session started with model: "
          f"{Colors.BOLD}{session.model}{Colors.RESET}")
    print(f"{Colors.DIM}Type /help for available commands.{Colors.RESET}\n")

    while True:
        try:
            user_input = input(f"{Colors.CYAN}You:{Colors.RESET} ").strip()
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted. Type 'exit' to quit, or "
                  f"keep chatting.{Colors.RESET}")
            continue
        except EOFError:
            print(f"\n{Colors.MAGENTA}Goodbye!{Colors.RESET}")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in ("exit", "quit", "/exit", "/quit"):
            print(f"{Colors.MAGENTA}Goodbye! Thanks for chatting. "
                  f"({session.message_count} messages sent, "
                  f"{session.total_tokens_used} tokens used){Colors.RESET}")
            break
        if command in ("help", "/help"):
            print_help()
            continue
        if command in ("clear", "/clear"):
            session.clear()
            print(f"{Colors.YELLOW}Conversation history cleared.{Colors.RESET}\n")
            continue
        if command in ("history", "/history"):
            print_history(session)
            continue
        if command.startswith("/model"):
            new_key = prompt_model_selection(current=session.model)
            session.switch_model(MODELS[new_key])
            print(f"{Colors.GREEN}Switched to model: "
                  f"{session.model}{Colors.RESET}\n")
            continue

        # Anything else is treated as a normal chat message.
        handle_chat_turn(client, session, user_input)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Catches Ctrl+C at any point outside the main input loop
        print(f"\n{Colors.MAGENTA}Goodbye!{Colors.RESET}")
        sys.exit(0)
