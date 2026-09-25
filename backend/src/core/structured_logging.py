import datetime
import json
import logging
from contextvars import ContextVar

correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(cid: str):
    return correlation_id.set(cid)


def get_correlation_id() -> str | None:
    return correlation_id.get()


class JSONStructuredLoggingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "correlation_id": correlation_id.get(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONStructuredLoggingFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.WARNING)
    root_logger.addHandler(handler)
