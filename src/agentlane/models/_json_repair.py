# pylint: disable=R0912


"""This module provides utilities for parsing JSON strings into dictionaries."""

import re
from typing import Any, cast

import demjson3
import json_repair

type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]


def _decode_relaxed_json(input_str: str) -> object | None:
    """Decode JSON-like content using demjson3's relaxed parser."""
    decode: Any = getattr(demjson3, "decode")  # noqa: B009
    return cast(object | None, decode(input_str, strict=False))


def _repair_json(input_str: str) -> object:
    """Repair malformed JSON-like content and return the parsed object."""
    return cast(
        object, json_repair.repair_json(json_str=input_str, return_objects=True)
    )


def parse_json_dict(maybe_json_str: str) -> dict[str, Any] | None:
    """Parse the input string into a dictionary. If the input string is not a valid JSON string,
    we will try to repair it. If repair fails, we will return None.

    Args:
        maybe_json_str: The input string to parse.

    Returns:
        The parsed dictionary, or None if the input string is not a valid JSON string
        or None if repair fails.
    """
    # Extract JSON from code blocks if present, but only if the input
    # doesn't already look like JSON (to avoid matching backticks inside JSON strings)
    stripped = maybe_json_str.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        code_match = re.search(code_block_pattern, maybe_json_str)
        if code_match:
            maybe_json_str = code_match.group(1)

    # If the input string does not look like JSON, we cannot parse it
    if not _looks_like_json(maybe_json_str):
        return None

    # Sanitize the JSON string (order matters!):
    # 1. First escape raw newlines inside strings (converts literal \n to \\n)
    # 2. Then fix double-escaped newlines (converts \\\\n to \\n)
    sanitized = _escape_raw_newlines_in_strings(maybe_json_str)
    sanitized = _fix_double_escaped_newlines(sanitized)

    # Try to parse the content as JSON first, assuming it's a string. If it's
    # a full valid JSON object or array, we'll return it as is.
    parsed_dict: dict[str, Any] | None = _parse_from_json_str(sanitized)
    if parsed_dict is not None:
        return parsed_dict

    # Use json_repair if all else fails
    result = _repair_json(sanitized)

    # json_repair can return various types, ensure we return dict or None
    return cast(dict[str, Any], result) if isinstance(result, dict) else None


def _maybe_unescape_json(input_str: str) -> str:
    """Unescape the input string if it is strongly escaped."""
    s = input_str.strip()
    # Heuristic: strongly escaped content (e.g. {\"...\", \\n)
    if s.startswith("{\\") or s.startswith("[\\") or '\\"' in s or "\\n" in s:
        try:
            return s.encode("utf-8").decode("unicode_escape")
        except Exception:
            return input_str
    return input_str


def _parse_from_json_str(input_str: str) -> dict[str, Any] | None:
    """Parse the input string assuming it's a valid JSON string."""

    def _decode_and_post_process(s: str) -> dict[str, Any] | None:
        """Decode the input string and post-process the result."""
        parsed_value = _decode_relaxed_json(s)
        if not isinstance(parsed_value, dict):
            return None
        parsed: dict[str, Any] = cast(dict[str, Any], parsed_value)

        # If the value is a string that is a valid JSON object or array,
        # try to parse it as JSON. We process only the top level keys for now.
        for key, value in parsed.items():
            if isinstance(value, str):
                # Clean the value by stripping whitespace and trailing commas/semicolons
                # to handle cases like '[{"key": "value"}],'
                cleaned_value = value.strip().rstrip(",;").strip()
                if (cleaned_value.startswith("{") and cleaned_value.endswith("}")) or (
                    cleaned_value.startswith("[")
                    and cleaned_value.endswith("]")
                    and _looks_like_json_array(cleaned_value)
                ):
                    try:
                        # Try parsing the cleaned value
                        parsed[key] = _decode_relaxed_json(cleaned_value)
                    except Exception:
                        fixed_value = cleaned_value  # Initialize before try
                        try:
                            # Fix string values inside the inner JSON
                            fixed_value = _double_escape_string_values_only(
                                cleaned_value
                            )
                            parsed[key] = _decode_relaxed_json(fixed_value)
                        except Exception:
                            # Final fallback: use json_repair on cleaned value
                            repaired = _repair_json(cleaned_value)
                            # Guard: if json_repair produced a single-element
                            # list with a primitive, it likely just bracket-
                            # wrapped a plain string (e.g. "[<=3.00]" -> [3.0]).
                            # Legitimate nested JSON arrays produce multi-
                            # element lists or lists containing dicts/lists.
                            repaired_list: list[Any] = repaired  # type: ignore[assignment]
                            if not (
                                isinstance(repaired, list)
                                and len(repaired_list) == 1
                                and not isinstance(repaired_list[0], (dict, list))
                            ):
                                parsed[key] = repaired

        return parsed

    try:
        return _decode_and_post_process(input_str)
    except (demjson3.JSONDecodeError, demjson3.JSONException):
        unescaped = _maybe_unescape_json(input_str)
        if unescaped != input_str:
            try:
                return _decode_and_post_process(unescaped)
            except (demjson3.JSONDecodeError, demjson3.JSONException):
                return None
        return None


def _looks_like_json(maybe_json_str: str) -> bool:
    """
    Check if the input string has basic JSON-like structure.

    Args:
        maybe_json_str: The content string to check.

    Returns:
        True if the input string appears to have JSON-like structure, False otherwise.
    """
    if not maybe_json_str.strip():
        return False

    # Check for JSON structural elements that indicate object/array structure
    # We need both structural chars AND proper pairing to consider it JSON-like
    stripped_str = maybe_json_str.strip()

    # Must start and end with JSON delimiters
    has_object_structure = (
        stripped_str.startswith("{") and stripped_str.endswith("}")
    ) or (stripped_str.startswith("[") and stripped_str.endswith("]"))

    # Or contain quoted strings with colons (key-value pairs)
    has_key_value_pairs = (
        '"' in maybe_json_str
        and ":" in maybe_json_str
        and ("{" in maybe_json_str or "}" in maybe_json_str)
    )

    return has_object_structure or has_key_value_pairs


def _looks_like_json_array(value: str) -> bool:
    """Check if a bracket-enclosed string contains actual JSON array content.

    Prevents plain bracketed strings like "[3.8-11.8]" or "[<=3.00]" from
    being mistakenly parsed as JSON arrays.

    Args:
        value: A string that starts with '[' and ends with ']'.

    Returns:
        True if the inner content contains JSON structural characters.
    """
    inner = value[1:-1].strip()
    if not inner:
        return True
    return '"' in inner or "{" in inner or "[" in inner or "," in inner


def _escape_raw_newlines_in_strings(input_str: str) -> str:
    r"""Escape raw newline characters inside JSON string values.

    Models sometimes output raw newlines inside JSON strings which is invalid JSON.
    This function converts them to proper \\n escape sequences.

    Args:
        input_str: The JSON string to process.

    Returns:
        The JSON string with raw newlines escaped inside string values.
    """
    result: list[str] = []
    in_string = False
    escape = False

    for char in input_str:
        # Handle escape sequences
        if escape:
            result.append(char)
            escape = False
            continue

        if char == "\\":
            result.append(char)
            escape = True
            continue

        # Track string boundaries
        if char == '"' and not escape:
            in_string = not in_string
            result.append(char)
        elif char == "\n" and in_string:
            # Raw newline inside string - escape it
            result.append("\\n")
        elif char == "\r" and in_string:
            # Raw carriage return inside string - escape it
            result.append("\\r")
        else:
            result.append(char)

    return "".join(result)


def _fix_double_escaped_newlines(input_str: str) -> str:
    r"""Convert double-escaped newlines to single-escaped in JSON string values.

    Models sometimes output \\\\n (which decodes to literal \\n text) when they
    should output \\n (which decodes to an actual newline). This function fixes that.

    Args:
        input_str: The JSON string to process.

    Returns:
        The JSON string with double-escaped newlines converted to single-escaped.
    """
    result: list[str] = []
    in_string = False
    i = 0
    n = len(input_str)

    while i < n:
        char = input_str[i]

        # Track string boundaries (handle escaped quotes)
        if char == "\\" and i + 1 < n:
            next_char = input_str[i + 1]
            if in_string and next_char == "\\" and i + 2 < n:
                # Check for double-escaped sequences: \\n, \\t, \\r
                third_char = input_str[i + 2]
                if third_char in "ntr":
                    # Convert \\n to \n (single escape)
                    result.append("\\")
                    result.append(third_char)
                    i += 3
                    continue
            # Handle regular escape sequences
            result.append(char)
            result.append(next_char)
            i += 2
            continue

        if char == '"':
            in_string = not in_string

        result.append(char)
        i += 1

    return "".join(result)


def _double_escape_string_values_only(input_str: str) -> str:
    """Fix unescaped quotes inside string values in a JSON string."""
    # Parse character by character, tracking state
    result: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False

    for i, char in enumerate(input_str):
        # Handle escape sequences
        if escape:
            result.append(char)
            escape = False
            continue

        if char == "\\":
            result.append(char)
            escape = True
            continue

        # Track JSON structure
        if not in_string:
            if char in ["{", "["]:
                stack.append(char)
            elif char == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif char == "]" and stack and stack[-1] == "[":
                stack.pop()
            elif char == '"':
                in_string = True
            result.append(char)
        else:  # We're inside a string
            if char == '"':
                # Is this quote ending the string or is it internal?
                # Look ahead to see if what follows looks like valid JSON structure
                j = i + 1
                while j < len(input_str) and input_str[j].isspace():
                    j += 1

                # If next non-space is a comma, colon, or closing bracket,
                # it's likely an ending quote
                is_ending = j < len(input_str) and input_str[j] in ",:}]"

                if is_ending:
                    in_string = False
                    result.append(char)
                else:
                    # This is an internal quote - escape it
                    result.append("\\")
                    result.append(char)
            else:
                result.append(char)

    return "".join(result)
