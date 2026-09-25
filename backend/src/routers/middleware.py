import uuid

from core.structured_logging import set_correlation_id
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        corr_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        set_correlation_id(cid=corr_id)

        response = await call_next(request)

        response.headers["X-Request-ID"] = corr_id
        return response
