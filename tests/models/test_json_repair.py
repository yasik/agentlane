"""Tests for JSON repair helpers."""

from typing import Any, cast

from agentlane.models import parse_json_dict


def test_json_in_code_block() -> None:
    """JSON should parse from fenced code blocks with or without a language tag."""
    json_in_code = '```json\n{"key": "value"}\n```'
    result = parse_json_dict(json_in_code)

    assert isinstance(result, dict)
    assert result == {"key": "value"}

    json_in_code = '```\n{"key": "value"}\n```'
    result = parse_json_dict(json_in_code)

    assert isinstance(result, dict)
    assert result == {"key": "value"}


def test_unquoted_keys() -> None:
    """Unquoted object keys should be repaired."""
    unquoted_keys = '{key: "value", another_key: 123}'
    result = parse_json_dict(unquoted_keys)
    assert isinstance(result, dict)
    assert result == {"key": "value", "another_key": 123}

    unquoted_keys = '{ key : "value",  another_key : 123 }'
    result = parse_json_dict(unquoted_keys)
    assert isinstance(result, dict)
    assert result == {"key": "value", "another_key": 123}


def test_internal_quotes() -> None:
    """Internal quotes inside string values should be repaired."""
    internal_quotes = '{"text": "This contains "quoted text" inside"}'
    result = parse_json_dict(internal_quotes)

    assert isinstance(result, dict)
    assert result == {"text": 'This contains "quoted text" inside'}


def test_multiple_internal_quotes() -> None:
    """Multiple internal quotes inside a string should be repaired."""
    internal_quotes = '{"text": "This contains "multiple" "quoted" parts"}'
    result = parse_json_dict(internal_quotes)

    assert isinstance(result, dict)
    assert result == {"text": 'This contains "multiple" "quoted" parts'}


def test_newline_escapes() -> None:
    """Escaped newlines should decode to real newlines."""
    escaped_newlines = '{"text": "Line 1\\nLine 2"}'
    result = parse_json_dict(escaped_newlines)

    assert isinstance(result, dict)
    assert result == {"text": "Line 1\nLine 2"}


def test_complex_example() -> None:
    """A mixed invalid JSON example should still repair successfully."""
    complex_json = """```json
    {
        key: "value",
        "description": "This text has "quotes" inside and \\n newlines",
        nested: {
            "array": [1, 2, 3],
            "text": \"Some more \"quoted\" text\"
        }
    }
    ```"""

    result = parse_json_dict(complex_json)

    assert isinstance(result, dict)
    assert result == {
        "key": "value",
        "description": 'This text has "quotes" inside and \n newlines',
        "nested": {"array": [1, 2, 3], "text": 'Some more "quoted" text'},
    }


def test_real_world_examples() -> None:
    """Representative malformed LLM JSON should repair correctly."""
    real_example = (
        '{"response": "I\'ll help you analyze this text where the author discusses '
        '"tariffs" and its implications."}'
    )
    result = parse_json_dict(real_example)

    assert isinstance(result, dict)
    assert result == {
        "response": (
            'I\'ll help you analyze this text where the author discusses "tariffs" '
            "and its implications."
        )
    }

    real_example2 = """{
        "analysis": "The customer is expressing frustration about the \\"premium\\" service they purchased.",
        "sentiment": "negative",
        "action_items": ["Refund request", "Review premium features"]
    }"""
    result = parse_json_dict(real_example2)

    assert isinstance(result, dict)
    assert result == {
        "analysis": (
            'The customer is expressing frustration about the "premium" service '
            "they purchased."
        ),
        "sentiment": "negative",
        "action_items": ["Refund request", "Review premium features"],
    }


def test_trailing_comma_edge_case() -> None:
    """Nested JSON strings with trailing commas should still parse."""
    trailing_comma_json = """{
   "some_key": '[{"some_sub_key": "value"}],',
   "some_other_key": '[{"some_other_sub_key": "value"}],'
}"""
    result = parse_json_dict(trailing_comma_json)

    assert isinstance(result, dict)
    some_key = cast(list[dict[str, Any]], result["some_key"])
    some_other_key = cast(list[dict[str, Any]], result["some_other_key"])
    assert len(some_key) == 1
    assert len(some_other_key) == 1
    assert some_key[0] == {"some_sub_key": "value"}
    assert some_other_key[0] == {"some_other_sub_key": "value"}


def test_trailing_semicolon_edge_case() -> None:
    """Nested JSON strings with trailing semicolons should still parse."""
    trailing_semicolon_json = """{
   "some_key": '[{"some_sub_key": "value"}];',
   "some_other_key": '{"nested_key": "nested_value"};'
}"""
    result = parse_json_dict(trailing_semicolon_json)

    assert isinstance(result, dict)
    some_key = cast(list[dict[str, Any]], result["some_key"])
    some_other_key = cast(dict[str, Any], result["some_other_key"])
    assert some_key[0] == {"some_sub_key": "value"}
    assert some_other_key == {"nested_key": "nested_value"}


def test_plain_text_non_json_content() -> None:
    """Plain non-JSON text should return None."""
    plain_text_examples = [
        "This is just a regular sentence with no JSON structure.",
        "Hello world! How are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "Error: Something went wrong in the system.",
        "Processing complete. No further action required.",
        "   ",
        "",
        "Simple text without any special characters",
        "Text with numbers 123 and symbols !@# but no JSON structure",
    ]

    for plain_text in plain_text_examples:
        result = parse_json_dict(plain_text)
        assert result is None, f"Expected None for plain text: '{plain_text}'"


def test_edge_case_text_with_some_json_chars() -> None:
    """Text with a few JSON-like characters should not crash the repair path."""
    edge_case_examples = [
        'The user said: "Hello world" and then left.',
        "Here's a list: [item1, item2, item3] but it's not JSON.",
        "The config has {key: value} pairs in it.",
        'Error message: "Invalid input" received.',
        "Array notation [0] is used in programming.",
        "Object notation {obj} is common in code.",
    ]

    for edge_text in edge_case_examples:
        result = parse_json_dict(edge_text)
        assert result is None or isinstance(result, dict | list), (
            f"Unexpected result type for edge case: '{edge_text}'"
        )


def test_valid_json() -> None:
    """Valid JSON should continue to parse unchanged."""
    valid_json = '{"name": "John", "age": 30, "city": "New York"}'
    result = parse_json_dict(valid_json)
    assert result == {"name": "John", "age": 30, "city": "New York"}

    nested_json = '{"user": {"name": "Jane", "settings": {"theme": "dark"}}}'
    result = parse_json_dict(nested_json)
    assert result == {"user": {"name": "Jane", "settings": {"theme": "dark"}}}

    array_json = '{"items": [1, 2, 3], "tags": ["red", "blue"]}'
    result = parse_json_dict(array_json)
    assert result == {"items": [1, 2, 3], "tags": ["red", "blue"]}


def test_none_input() -> None:
    """Empty and whitespace-only strings should return None."""
    assert parse_json_dict("") is None
    assert parse_json_dict("   ") is None
    assert parse_json_dict("\n\n") is None


def test_double_escaped_newlines() -> None:
    r"""Double-escaped newlines should normalize to actual newlines."""
    json_str = '{"text": "line1\\\\nline2"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result == {"text": "line1\nline2"}

    json_str = '{"text": "line1\\\\nline2\\\\nline3"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result == {"text": "line1\nline2\nline3"}

    json_str = '{"text": "Hello\\\\n\\\\nWorld"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result == {"text": "Hello\n\nWorld"}


def test_double_escaped_tabs() -> None:
    r"""Double-escaped tabs should normalize to actual tabs."""
    json_str = '{"text": "col1\\\\tcol2\\\\tcol3"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result == {"text": "col1\tcol2\tcol3"}


def test_raw_newlines_in_strings() -> None:
    """Raw newlines inside string values should be tolerated."""
    json_str = '{"text": "line1\nline2"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result == {"text": "line1\nline2"}

    json_str = '{"text": "line1\nline2\nline3"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result == {"text": "line1\nline2\nline3"}


def test_raw_carriage_returns_in_strings() -> None:
    """Raw carriage returns inside string values should be tolerated."""
    json_str = '{"text": "line1\rline2"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result == {"text": "line1\rline2"}


def test_code_blocks_with_newlines_in_strings() -> None:
    """Markdown code blocks inside JSON string values should preserve newlines."""
    json_str = '{"text": "```python\\\\nprint(\'hello\')\\\\n```"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert "```python" in result["text"]
    assert "print('hello')" in result["text"]
    assert "\n" in result["text"]

    json_str = '{"text": "```\ncode here\n```"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert "```" in result["text"]
    assert "\n" in result["text"]


def test_real_world_llm_output_double_escaped() -> None:
    """Real-world double-escaped LLM output should normalize correctly."""
    json_str = (
        '{"text": "consider targeted nutritional strategies.\\\\n\\\\n* '
        '**Reduced albumin synthesis (moderate strength)**: Albumin below '
        'optimal suggests a mild reduction."}'
    )
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert "nutritional strategies." in result["text"]
    assert "\n\n* **Reduced albumin" in result["text"]


def test_real_world_llm_output_code_blocks() -> None:
    """Code blocks embedded in LLM JSON should retain their structure."""
    json_str = '{"analysis": "```\\\\nA mild hepatic synthetic shift\\\\n```"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert "```" in result["analysis"]
    assert "\n" in result["analysis"]


def test_mixed_escape_issues() -> None:
    """Mixed raw and double-escaped newline issues should both normalize."""
    json_str = '{"text": "line1\nline2\\\\nline3"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result["text"].count("\n") == 2


def test_range_string_not_converted_to_array() -> None:
    """Bracketed reference ranges should remain strings."""
    json_str = '{"reference_range": "[3.8-11.8]", "name": "T4 Free"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result["reference_range"] == "[3.8-11.8]"
    assert result["name"] == "T4 Free"


def test_inequality_range_not_converted_to_array() -> None:
    """Bracketed inequality ranges should remain strings."""
    json_str = '{"reference_range": "[<=3.00]", "name": "hs-CRP"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result["reference_range"] == "[<=3.00]"
    assert result["name"] == "hs-CRP"


def test_bracketed_word_not_converted_to_array() -> None:
    """Bracketed words should remain strings."""
    json_str = '{"status": "[normal]"}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result["status"] == "[normal]"


def test_structural_newlines_preserved() -> None:
    """Structural JSON whitespace should not change the parsed payload."""
    json_str = '{\n  "key1": "value1",\n  "key2": "value2"\n}'
    result = parse_json_dict(json_str)
    assert isinstance(result, dict)
    assert result == {"key1": "value1", "key2": "value2"}
