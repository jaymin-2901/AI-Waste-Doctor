"""HTTP client for the AI Waste Doctor API (used by streamlit/app.py).

The client never calls ``response.json()`` blindly. Every response is checked for
status code and content type first, so proxy/HTML error pages, empty bodies and
malformed JSON become clear ``ClassificationError`` messages instead of parser errors.
"""

import time
from typing import Callable, Optional

import requests

# HTTP statuses that mean "try again shortly" (cold start, restart, gateway trouble).
RETRYABLE_STATUSES = {408, 425, 429, 502, 503, 504}
REQUIRED_RESULT_KEYS = ("top_class", "top_confidence", "is_confident", "predictions")

MAX_ATTEMPTS = 4
BACKOFF_SECONDS = (5, 10, 20)  # waits between attempts; bounded, never infinite
HEALTH_TIMEOUT = (10, 75)  # (connect, read) - a cold Render instance can hold the request ~60 s
PREDICT_TIMEOUT = (10, 120)


class ClassificationError(Exception):
    """A failure the user should see. ``technical`` is safe to show in a details box."""

    def __init__(self, message: str, technical: str = "", status: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.user_message = message
        self.technical = technical or message
        self.status = status
        self.retryable = retryable


def _describe(response: requests.Response) -> str:
    content_type = response.headers.get("Content-Type", "unknown")
    snippet = (response.text or "").strip().replace("\n", " ")[:160]
    return f"HTTP {response.status_code}, content-type {content_type}, body: {snippet or '<empty>'}"


def _json_or_none(response: requests.Response) -> Optional[dict]:
    """Return a JSON object body, or None for HTML, empty, malformed or non-object bodies."""
    if "json" not in response.headers.get("Content-Type", "").lower():
        return None
    if not (response.content or b"").strip():
        return None
    try:
        data = response.json()
    except ValueError:  # includes requests' JSONDecodeError
        return None
    return data if isinstance(data, dict) else None


def _api_detail(data: Optional[dict]) -> str:
    if not data:
        return ""
    return str(data.get("detail") or data.get("model_error") or data.get("model_status") or "")


def _check_health(session, api_url: str) -> Optional[ClassificationError]:
    """Return None when the model is ready, otherwise a ClassificationError."""
    try:
        response = session.get(f"{api_url}/health", timeout=HEALTH_TIMEOUT)
    except requests.Timeout as error:
        return ClassificationError(
            "The classification service did not answer in time (it may be starting up).",
            technical=f"Health check timed out: {error}", retryable=True,
        )
    except requests.RequestException as error:
        return ClassificationError(
            "Could not reach the classification service.",
            technical=f"Health check failed: {error}", retryable=True,
        )

    data = _json_or_none(response)
    if response.status_code == 200 and data is not None:
        if data.get("ready", data.get("model_loaded", False)):
            return None
        return ClassificationError(
            "The classification model is not ready yet.",
            technical=f"Health JSON reported not ready: {_api_detail(data) or data}",
            status=200, retryable=True,
        )
    if response.status_code == 200:  # 200 but HTML/empty/malformed: a proxy or wake-up page
        return ClassificationError(
            "The service returned an unexpected page (it is probably waking up).",
            technical=_describe(response), status=200, retryable=True,
        )
    detail = _api_detail(data)
    return ClassificationError(
        "The classification service is not ready" + (f": {detail}" if detail else "."),
        technical=_describe(response), status=response.status_code,
        retryable=response.status_code in RETRYABLE_STATUSES,
    )


def _post_predict(session, api_url: str, filename: str, data: bytes, content_type: str) -> dict:
    """Return the validated prediction dict or raise ClassificationError."""
    try:
        response = session.post(
            f"{api_url}/predict",
            files={"file": (filename, data, content_type)},
            timeout=PREDICT_TIMEOUT,
        )
    except requests.Timeout as error:
        raise ClassificationError(
            "The prediction took too long. Please try again.",
            technical=f"Predict timed out: {error}", retryable=True,
        ) from error
    except requests.RequestException as error:
        raise ClassificationError(
            "Lost connection to the classification service.",
            technical=f"Predict request failed: {error}", retryable=True,
        ) from error

    body = _json_or_none(response)
    status = response.status_code

    if status == 200:
        if body is None:
            raise ClassificationError(
                "The service returned an unexpected page instead of a result.",
                technical=_describe(response), status=status, retryable=True,
            )
        missing = [key for key in REQUIRED_RESULT_KEYS if key not in body]
        if missing or not isinstance(body["predictions"], dict):
            raise ClassificationError(
                "The service returned an incomplete result.",
                technical=f"Missing/invalid fields {missing}: {body}", status=status,
            )
        return body

    detail = _api_detail(body)
    if status in RETRYABLE_STATUSES:
        raise ClassificationError(
            "The classification service is busy or restarting" + (f": {detail}" if detail else "."),
            technical=_describe(response), status=status, retryable=True,
        )
    if 400 <= status < 500:  # our request is wrong (bad type, too large): retrying cannot help
        raise ClassificationError(
            detail or f"The service rejected the image (HTTP {status}).",
            technical=_describe(response), status=status,
        )
    raise ClassificationError(  # 500 etc.: preserve the API's detail, do not retry blindly
        detail or f"The classification service reported an error (HTTP {status}).",
        technical=_describe(response), status=status,
    )


def classify_image(
    api_url: str,
    filename: str,
    data: bytes,
    content_type: str,
    *,
    session=None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
    on_status: Optional[Callable[[str], None]] = None,
) -> dict:
    """Check /health, then POST /predict, retrying transient failures with bounded backoff."""
    session = session or requests
    api_url = api_url.rstrip("/")
    last_error: Optional[ClassificationError] = None

    for attempt in range(1, max_attempts + 1):
        if on_status:
            on_status(f"Contacting the classification service (attempt {attempt} of {max_attempts})...")
        try:
            not_ready = _check_health(session, api_url)
            if not_ready is None:
                return _post_predict(session, api_url, filename, data, content_type)
            last_error = not_ready
        except ClassificationError as error:
            last_error = error

        if not last_error.retryable:
            raise last_error
        if attempt < max_attempts:
            sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])

    assert last_error is not None
    raise ClassificationError(
        f"{last_error.user_message} Tried {max_attempts} times. Wait about a minute and try again; "
        "if it keeps failing, check the Render logs (see the recovery runbook in the README).",
        technical=last_error.technical, status=last_error.status,
    )
