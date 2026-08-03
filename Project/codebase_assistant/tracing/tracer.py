"""
tracer.py
=========

Defines Tracer, which collects TraceEvent records during a run and
exports them for reporting.

Intended to be constructed once by the Supervisor and shared, so a
single run produces one coherent trace across ingestion, retrieval,
agent dispatch, tool calls, and model calls.

All public methods swallow unexpected errors and log warnings so
tracing can never crash the application.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .events import TraceEvent, TraceEventType

logger = logging.getLogger(__name__)


class Tracer:
    """
    Collects and exports trace events for a single run.
    """

    def __init__(
        self,
        enabled: bool = True,
        run_id: str = "",
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        """
        Initialize the Tracer.

        Args:
            enabled: When False, all recording becomes a no-op.
            run_id: Identifier grouping every event from one run.
            time_fn: Optional clock (defaults to ``time.time``). Tests
                may inject a deterministic counter.
        """
        self.enabled = enabled
        self.run_id = run_id or str(uuid.uuid4())
        self._time: Callable[[], float] = time_fn or time.time
        self._events: List[TraceEvent] = []
        self._open_spans: Dict[str, Dict[str, Any]] = {}
        self._sequence: int = 0

    def start_span(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Begin a timed stage.

        Args:
            name: Identifier of the stage being timed.
            metadata: Extra detail to attach to the span.

        Returns:
            A span id used to close the span, or "" when disabled /
            on failure.
        """
        try:
            if not self.enabled:
                return ""
            span_id = str(uuid.uuid4())
            meta = dict(metadata or {})
            self._open_spans[span_id] = {
                "name": name,
                "start": float(self._time()),
                "metadata": meta,
            }
            self.record(
                TraceEventType.SPAN_START,
                name,
                component=str(meta.get("component") or ""),
                success=True,
                **{k: v for k, v in meta.items() if k != "component"},
            )
            return span_id
        except Exception as exc:
            logger.warning("Tracer.start_span failed: %s", exc)
            return ""

    def end_span(
        self, span_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Close a timed stage and record its duration.

        Args:
            span_id: Id returned by the matching ``start_span`` call.
            metadata: Extra detail to merge into the span.
        """
        try:
            if not self.enabled or not span_id:
                return
            span = self._open_spans.pop(span_id, None)
            if span is None:
                logger.warning("Tracer.end_span: unknown span_id %r", span_id)
                return
            ended = float(self._time())
            duration_ms = max(0.0, (ended - float(span["start"])) * 1000.0)
            merged = dict(span.get("metadata") or {})
            merged.update(dict(metadata or {}))
            success = merged.pop("success", True)
            error = merged.pop("error", None)
            component = str(merged.pop("component", "") or "")
            self.record(
                TraceEventType.SPAN_END,
                str(span["name"]),
                component=component,
                success=bool(success) if success is not None else None,
                duration_ms=duration_ms,
                error=str(error) if error else None,
                **merged,
            )
        except Exception as exc:
            logger.warning("Tracer.end_span failed: %s", exc)

    def record(
        self,
        event_type: TraceEventType,
        name: str,
        **metadata: Any,
    ) -> None:
        """
        Record a point-in-time event.

        Reserved metadata keys (``component``, ``success``,
        ``duration_ms``, ``error``) are lifted onto TraceEvent fields;
        remaining keys stay in ``metadata``.

        Args:
            event_type: Category of measurement.
            name: Identifier of what is being measured.
            **metadata: Extra detail (token counts, scores, etc).
        """
        try:
            if not self.enabled:
                return

            meta = dict(metadata)
            component = str(meta.pop("component", "") or "")
            success = meta.pop("success", None)
            if success is not None:
                success = bool(success)
            duration_ms = meta.pop("duration_ms", None)
            if duration_ms is not None:
                duration_ms = float(duration_ms)
            error = meta.pop("error", None)
            if error is not None:
                error = str(error)

            if isinstance(event_type, str):
                event_type = TraceEventType(event_type)

            self._sequence += 1
            event = TraceEvent(
                event_type=event_type,
                name=str(name or ""),
                timestamp=float(self._time()),
                duration_ms=duration_ms,
                metadata=meta,
                error=error,
                component=component,
                success=success,
                sequence=self._sequence,
            )
            self._events.append(event)
        except Exception as exc:
            logger.warning("Tracer.record failed: %s", exc)

    def get_events(
        self, event_type: Optional[TraceEventType] = None
    ) -> List[TraceEvent]:
        """
        Retrieve recorded events, optionally filtered by type.

        Args:
            event_type: When given, only events of this type are
                returned.

        Returns:
            Matching events in recording order (a shallow copy).
        """
        try:
            if event_type is None:
                return list(self._events)
            return [
                event
                for event in self._events
                if event.event_type == event_type
            ]
        except Exception as exc:
            logger.warning("Tracer.get_events failed: %s", exc)
            return []

    def event_names(self) -> List[str]:
        """
        Return recorded event names in order.

        Returns:
            Event name strings; empty on failure.
        """
        try:
            return [event.name for event in self._events]
        except Exception as exc:
            logger.warning("Tracer.event_names failed: %s", exc)
            return []

    def summarize(self) -> Dict[str, Any]:
        """
        Summarize the run as aggregate metrics.

        Returns:
            Counts by event type, component, and success/failure.
        """
        try:
            by_type: Dict[str, int] = {}
            by_component: Dict[str, int] = {}
            succeeded = 0
            failed = 0
            durations: List[float] = []
            for event in self._events:
                by_type[event.event_type.value] = (
                    by_type.get(event.event_type.value, 0) + 1
                )
                if event.component:
                    by_component[event.component] = (
                        by_component.get(event.component, 0) + 1
                    )
                if event.success is True:
                    succeeded += 1
                elif event.success is False:
                    failed += 1
                if event.duration_ms is not None:
                    durations.append(float(event.duration_ms))
            return {
                "run_id": self.run_id,
                "event_count": len(self._events),
                "by_type": dict(sorted(by_type.items())),
                "by_component": dict(sorted(by_component.items())),
                "succeeded": succeeded,
                "failed": failed,
                "total_duration_ms": round(sum(durations), 3) if durations else 0.0,
            }
        except Exception as exc:
            logger.warning("Tracer.summarize failed: %s", exc)
            return {}

    def export(self, path: str) -> bool:
        """
        Write the collected trace to disk as deterministic JSON.

        Args:
            path: Destination file path.

        Returns:
            True if the export succeeded, False otherwise.
        """
        try:
            payload = {
                "run_id": self.run_id,
                "events": [event.to_dict() for event in self._events],
                "summary": self.summarize(),
            }
            text = json.dumps(payload, indent=2, sort_keys=True, default=str)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.write("\n")
            return True
        except Exception as exc:
            logger.warning("Tracer.export failed for %r: %s", path, exc)
            return False

    def clear(self) -> None:
        """Discard all collected events and open spans."""
        try:
            self._events.clear()
            self._open_spans.clear()
            self._sequence = 0
        except Exception as exc:
            logger.warning("Tracer.clear failed: %s", exc)
