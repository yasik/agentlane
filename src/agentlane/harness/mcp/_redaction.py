"""Redact credentials from server-controlled schemas and tool results."""

import re
from collections.abc import Collection, Mapping, Sequence
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ._validation import is_credential_query_key

_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "authorization_code",
        "aws_access_key_id",
        "aws_secret_access_key",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "private_key_id",
        "proxy_authorization",
        "secret",
        "set_cookie",
        "token",
        "vertex_credentials",
        "x_api_key",
        "x_goog_api_key",
    }
)
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")
_KEY_VALUE_PATTERN = re.compile(
    r'"(?P<key>[^"\\]{1,128})"\s*:\s*(?P<value>"(?:[^"\\]|\\.)*"?|[^\s,}\]]+)'
)
"""Matches a quoted key and its value in text, including a cut-off string value."""


def is_secret_key(key: str) -> bool:
    """Return whether a JSON key identifies a credential value."""
    normalized = key.lower().replace("-", "_")
    return normalized in _SECRET_KEYS or normalized.endswith(
        ("_token", "_secret", "_password", "_credentials", "_api_key", "_private_key")
    )


def _redact_url(match: re.Match[str]) -> str:
    try:
        parsed = urlsplit(match.group())
        query = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=1000)
        safe_query = [
            (key, "[redacted]" if is_credential_query_key(key) else value)
            for key, value in query
        ]
        authority = parsed.netloc.rsplit("@", 1)[-1]
        fragment = "[redacted]" if parsed.fragment else ""
        if query == safe_query and authority == parsed.netloc and not parsed.fragment:
            return match.group()

        return urlunsplit(
            (parsed.scheme, authority, parsed.path, urlencode(safe_query), fragment)
        )
    except ValueError:
        return "[redacted URL]"


def redact_known_secrets(value: object, secrets: Collection[str]) -> object:
    """Copy catalog values without changing schema property names."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            key: redact_known_secrets(item, secrets) for key, item in mapping.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            redact_known_secrets(item, secrets)
            for item in cast(Sequence[object], value)
        ]

    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[redacted]")

        return _URL_PATTERN.sub(_redact_url, value)

    return value


def redact_secret_pairs(text: str) -> str:
    """Redact credential values in key-value pairs of text that is not JSON."""
    return _KEY_VALUE_PATTERN.sub(_redact_pair, text)


def _redact_pair(match: re.Match[str]) -> str:
    key = match.group("key")
    if not is_secret_key(key):
        return match.group()

    return f'"{key}":"[redacted]"'


def contains_secret_key(value: object, secrets: Collection[str]) -> bool:
    """Detect unsafe schema keys that cannot be redacted without changing calls."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return any(
            any(secret in str(key) for secret in secrets if secret)
            or contains_secret_key(item, secrets)
            for key, item in mapping.items()
        )

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(
            contains_secret_key(item, secrets) for item in cast(Sequence[object], value)
        )

    return False
