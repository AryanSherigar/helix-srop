"""
Structured logging setup.

All log lines must include session_id, trace_id, user_id when available.
Use structlog's context vars for request-scoped fields.
"""
import logging
import re
import sys
from typing import Any

import structlog


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{8,}\b")


def _redact_text(value: str) -> str:
    value = _EMAIL_RE.sub("***@***.***", value)
    value = _API_KEY_RE.sub("sk-****", value)
    return value


def _redact_pii(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {k: _redact_pii(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_pii(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_pii(v) for v in value)
    return value


def redact_pii(logger, method_name, event_dict):
    return _redact_pii(event_dict)


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            redact_pii,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )


# Usage in request handlers:
#   import structlog
#   log = structlog.get_logger()
#   structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)
#   log.info("pipeline_started", user_message_len=len(message))
