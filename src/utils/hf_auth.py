from __future__ import annotations

"""Hugging Face authentication helpers.

Policy (from the hardening spec):
  * The HF token NEVER appears in logs. Only ``HF authentication: available``
    or ``HF authentication: unavailable (<reason>)`` is ever logged.
  * Startup status is classified into three failure families:
      AUTHENTICATION MISSING            — no token configured
      AUTHENTICATED BUT ACCESS DENIED   — token present but gated/401/403
      DATASET RESOLUTION FAILURE        — dataset could not be resolved
    plus a plain AUTHENTICATED success state.
  * The token is applied to every HF hub/datasets call via the HF_TOKEN env
    (set once at startup) so gated datasets (e.g. the-stack-v2-dedup) load.
"""

import logging
import os
import traceback

logger = logging.getLogger(__name__)

_AUTH_MISSING = "AUTHENTICATION MISSING"
_AUTH_ACCESS_DENIED = "AUTHENTICATED BUT ACCESS DENIED"
_AUTH_RESOLUTION = "DATASET RESOLUTION FAILURE"
_AUTH_OK = "AUTHENTICATED"
_AUTH_UNKNOWN = "UNKNOWN AUTH STATE"

_CLASSIFICATIONS = (_AUTH_MISSING, _AUTH_ACCESS_DENIED, _AUTH_RESOLUTION, _AUTH_OK, _AUTH_UNKNOWN)


def get_hf_token(cfg=None) -> str:
    """Return the effective HF token (config override, else env), never logged."""
    token = ""
    if cfg is not None:
        token = getattr(getattr(cfg, "data", None), "hf_token", None) or ""
    return token or os.environ.get("HF_TOKEN", "")


def redact_text(text: str) -> str:
    """Replace any configured HF token with ``<REDACTED>`` so tracebacks and
    error messages can be logged without leaking credentials."""
    if not text:
        return text
    candidates = []
    token = get_hf_token(None)
    if token:
        candidates.append(token)
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            candidates.append(cached)
    except Exception:
        pass
    for secret in dict.fromkeys(c for c in candidates if c):
        if secret in text:
            text = text.replace(secret, "<REDACTED>")
    return text


def format_sanitized_traceback(exc: BaseException) -> str:
    """Full, token-redacted traceback for one exception (for the error log)."""
    raw = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return redact_text(raw)


def _status_code_of(exc: BaseException):
    """Defensively extract an HTTP status code from hubs exceptions."""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    code = getattr(resp, "status_code", None)
    if code is None:
        status = getattr(resp, "status", None)
        code = getattr(status, "status_code", None)
    return code


def classify_hf_error(exc: BaseException, token_present: bool = False) -> str:
    """Map an exception raised by hub/datasets calls to a classification."""
    status_code = _status_code_of(exc)
    if status_code in (401, 403):
        return _AUTH_ACCESS_DENIED
    err = str(exc).lower()
    if any(kw in err for kw in ["gated", "permission", "access denied", "forbidden"]):
        return _AUTH_ACCESS_DENIED if token_present else _AUTH_MISSING
    if any(kw in err for kw in ["authentication token", "bad credentials", "401"]):
        return _AUTH_ACCESS_DENIED if token_present else _AUTH_MISSING
    if any(kw in err for kw in ["not found", "doesn't exist", "404", "cannot resolve",
                                "failed to resolve", "couldn't find"]):
        return _AUTH_RESOLUTION
    return _AUTH_UNKNOWN


def whoami(token: str):
    """Thin wrapper for testability: hub identity check."""
    from huggingface_hub import HfApi
    return HfApi(token=token).whoami()


def detect_hf_auth(cfg=None) -> dict:
    """Assess HF authentication state without ever logging the token.

    Returns a dict::

        {
          "authenticated": bool,
          "classification": one of _CLASSIFICATIONS,
          "message": str,          # human summary (token-free)
          "token_present": bool,
          "checked": bool,         # False when no token → skipped network call
        }

    Never raises: network failures degrade to a non-fatal classification.
    """
    token = get_hf_token(cfg)
    if not token:
        return {
            "authenticated": False,
            "classification": _AUTH_MISSING,
            "message": "no HF token configured",
            "token_present": False,
            "checked": False,
        }
    try:
        whoami(token)
        return {
            "authenticated": True,
            "classification": _AUTH_OK,
            "message": "token accepted",
            "token_present": True,
            "checked": True,
        }
    except Exception as e:
        classification = classify_hf_error(e, token_present=True)
        return {
            "authenticated": False,
            "classification": classification,
            "message": redact_text(f"{type(e).__name__}: {e}"),
            "token_present": True,
            "checked": True,
        }


def report_hf_auth(cfg=None) -> dict:
    """Log the one allowed auth line: ``HF authentication: available`` /
    ``HF authentication: unavailable (<reason>)``. Returns the status dict."""
    status = detect_hf_auth(cfg)
    if status["authenticated"]:
        logger.info("HF authentication: available")
    elif status["classification"] == _AUTH_MISSING:
        logger.warning("HF authentication: unavailable (AUTHENTICATION MISSING — "
                       "set HF_TOKEN or data.hf_token to access gated datasets)")
    elif status["classification"] == _AUTH_ACCESS_DENIED:
        logger.warning("HF authentication: unavailable (AUTHENTICATED BUT ACCESS "
                       "DENIED — token present but an API accepted it? request "
                       "dataset access and use a valid token)")
    elif status["classification"] == _AUTH_RESOLUTION:
        logger.warning("HF authentication: unavailable (DATASET RESOLUTION FAILURE)")
    else:
        logger.warning("HF authentication: unavailable (%s)", status["classification"])
    return status