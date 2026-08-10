"""
events.py
=========

Enumerates the lifecycle points at which hooks can fire.

TODO: Emit these events from the Supervisor, agents, tools, and the
ingestion pipeline once hook dispatch is implemented. Extend the
enum as further instrumentation points are identified.
"""

from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    """
    Points in a run where hooks may be triggered.

    Attributes:
        BEFORE_INGEST: Immediately before repository ingestion starts.
        AFTER_INGEST: After ingestion completes.
        BEFORE_AGENT_RUN: Before the Supervisor dispatches to an agent.
        AFTER_AGENT_RUN: After an agent returns its response.
        BEFORE_TOOL_CALL: Before a tool is invoked.
        AFTER_TOOL_CALL: After a tool returns.
        BEFORE_MODEL_CALL: Before a request is sent to the LLM.
        AFTER_MODEL_CALL: After an LLM response is received.
        ON_ERROR: When any stage raises an error.
    """

    BEFORE_INGEST = "before_ingest"
    AFTER_INGEST = "after_ingest"
    BEFORE_AGENT_RUN = "before_agent_run"
    AFTER_AGENT_RUN = "after_agent_run"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_MODEL_CALL = "before_model_call"
    AFTER_MODEL_CALL = "after_model_call"
    ON_ERROR = "on_error"
