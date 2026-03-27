# pylint: disable=C0103

"""This module defines the base class for output schemas and a concrete implementation
for JSON-based output types.

The implementation is largely borrowed from OpenAI Agents SDK with a few modifications.
(https://github.com/openai/openai-agents-python).
"""

import abc
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from ._json_repair import parse_json_dict
from ._response_utils import get_content_or_none
from ._strict_schema import ensure_strict_json_schema
from ._types import ModelResponse

_WRAPPER_DICT_KEY = "response"
type JsonObject = dict[str, Any]


@dataclass
class SchemaValidationResult:
    """Result of schema validation attempt."""

    is_valid: bool
    """Whether the response was valid."""

    should_retry: bool
    """Whether to retry with guidance."""

    error_message: str | None
    """Error message text for retry guidance (None if valid or no more retries)."""


# Covariant so OutputSchema[SubModel] is compatible with OutputSchema[BaseModel]
OutT_co = TypeVar("OutT_co", covariant=True)


class OutputSchemaBase[OutT_co](abc.ABC):
    """An object that captures the JSON schema of the output, as well as validating/parsing JSON
    produced by the LLM into the output type.
    """

    @abc.abstractmethod
    def is_plain_text(self) -> bool:
        """Whether the output type is plain text (versus a JSON object)."""

    @abc.abstractmethod
    def name(self) -> str:
        """The name of the output type."""

    @abc.abstractmethod
    def json_schema(self) -> dict[str, Any]:
        """Returns the JSON schema of the output. Will only be called if the output type is not
        plain text.
        """

    @abc.abstractmethod
    def is_strict_json_schema(self) -> bool:
        """Whether the JSON schema is in strict mode. Strict mode constrains the JSON schema
        features, but guarantees valid JSON.
        """

    @abc.abstractmethod
    def validate_json(self, json_str: str) -> OutT_co | None:
        """Validate a JSON string against the output type. You must return the validated object,
        or raise a `ModelBehaviorError` if the JSON is invalid.
        """

    @abc.abstractmethod
    def response_format(self) -> dict[str, Any] | None:
        """Returns the OpenAI compatible response format for the output type."""


@dataclass(init=False)
class OutputSchema(OutputSchemaBase[OutT_co]):
    """An object that captures the JSON schema of the output, as well as
    validating/parsing JSON produced by the LLM into the output type.
    """

    output_type: type[OutT_co]
    """The type of the output."""

    _type_adapter: TypeAdapter[OutT_co]
    """A type adapter that wraps the output type, so that we can validate JSON."""

    _is_wrapped: bool
    """Whether the output type is wrapped in a dictionary. This is generally done if the base
    output type cannot be represented as a JSON Schema object.
    """

    _output_schema: dict[str, Any]
    """The JSON schema of the output."""

    _strict_json_schema: bool
    """Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
    as it increases the likelihood of correct JSON input.
    """

    def __init__(self, output_type: type[OutT_co], strict_json_schema: bool = True):
        """Initialize the output schema.

        Args:
            output_type: The type of the output.
        """
        self.output_type = output_type
        self._strict_json_schema = strict_json_schema

        if output_type is str:
            self._is_wrapped = False
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()
            return

        # We should wrap for things that are not plain text, and for things that would definitely
        # not be a JSON Schema object.
        self._is_wrapped = not _is_subclass_of_base_model_or_dict(output_type)

        if self._is_wrapped:
            OutputType = TypedDict(
                "OutputType",
                {
                    _WRAPPER_DICT_KEY: output_type,  # type: ignore
                },
            )
            self._type_adapter = TypeAdapter(OutputType)  # type: ignore
            self._output_schema = self._type_adapter.json_schema()
        else:
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()

        if self._strict_json_schema:
            try:
                self._output_schema = ensure_strict_json_schema(self._output_schema)
            except ValueError as e:
                raise ValueError(
                    "Either make the output type strict, "
                    "or wrap your type with OutputSchema(YourType, strict_json_schema=False)"
                ) from e

    def is_plain_text(self) -> bool:
        """Whether the output type is plain text (versus a JSON object)."""
        return self.output_type is str

    def name(self) -> str:
        """The name of the output type."""
        return _type_to_str(self.output_type)

    def json_schema(self) -> dict[str, Any]:
        """Returns the JSON schema of the output."""
        if self.is_plain_text():
            raise ValueError(
                "Output type is plain text, so no JSON schema is available."
            )
        return self._output_schema

    def is_strict_json_schema(self) -> bool:
        """Whether the JSON schema is in strict mode."""
        return self._strict_json_schema

    def validate_json(
        self, json_str: str | None, strict: bool = False
    ) -> OutT_co | None:
        """Validate a JSON string against the output type.

        Args:
            json_str: The JSON string to validate.
            strict: Whether to validate the JSON string in strict mode.
                - strict=False (default, recommended): Uses JSON repair to handle common LLM output
                  issues like unquoted keys, internal quotes, raw/over-escaped newlines, and
                  markdown code blocks. Also searches for nested matching objects when the
                  top-level doesn't match the schema.
                - strict=True: Uses Pydantic's validate_json() directly. Expects syntactically
                  valid JSON. Use for testing or when you're confident the input is well-formed.

        Returns:
            The validated object, or None if the JSON string is invalid.
        """
        if not json_str:
            return None

        if strict:
            return self._validate_json_strict(json_str)
        return self._validate_json_partial(json_str)

    def _validate_json_strict(self, json_str: str, partial: bool = False) -> OutT_co:
        """Validate a JSON string against the output type in strict mode."""
        partial_setting: bool | Literal["off", "on", "trailing-strings"] = (
            "trailing-strings" if partial else False
        )
        try:
            validated = self._type_adapter.validate_json(
                json_str,
                experimental_allow_partial=partial_setting,
            )
            if self._is_wrapped:
                if not isinstance(validated, dict):
                    raise ValueError(
                        f"Expected a dict, got {type(validated)} for JSON: {json_str}"
                    )
                if _WRAPPER_DICT_KEY not in validated:
                    raise ValueError(
                        f"Expected a dict with key {_WRAPPER_DICT_KEY} in JSON: {json_str}"
                    )
                return cast(OutT_co, validated[_WRAPPER_DICT_KEY])
            return validated
        except ValidationError as e:
            raise ValueError(
                f"Invalid JSON when parsing {json_str} for {self._type_adapter}"
            ) from e

    def _validate_json_partial(self, json_str: str) -> OutT_co | None:
        """Validate a JSON string against the output type in partial mode."""
        json_dict = parse_json_dict(json_str)
        if json_dict is None:
            return None

        validated: OutT_co | None = None

        try:
            validated = self._type_adapter.validate_python(json_dict)
        except ValidationError:
            schema_properties = self._output_schema.get("properties", {})
            required_props = self._output_schema.get("required", [])
            matching_obj = _find_matching_object(
                json_dict, schema_properties, required_props
            )
            if matching_obj:
                try:
                    validated = self._type_adapter.validate_python(matching_obj)
                except ValidationError:
                    pass

        if validated is None:
            raise ValueError(
                f"Invalid JSON when parsing {json_str} for {self._type_adapter}"
            )

        if self._is_wrapped:
            if not isinstance(validated, dict):
                raise ValueError(
                    f"Expected wrapped JSON object for {self._type_adapter}"
                )
            return cast(OutT_co, validated[_WRAPPER_DICT_KEY])

        return validated

    def response_format(self) -> dict[str, Any] | None:
        """Returns the OpenAI compatible response format for the output type."""
        if self.is_plain_text():
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "schema": self.json_schema(),
                "name": self.name(),
                "strict": self.is_strict_json_schema(),
            },
        }

    def parse_response(self, response: Any, strict: bool = False) -> OutT_co | None:
        """Extract content from response and validate against the output type.

        Convenience method that combines get_content_or_none() and validate_json().
        This encapsulates the common pattern of extracting content from an LLM
        response and validating it against the schema.

        Args:
            response: The model response to extract and validate content from.
                Expected to be a ModelResponse or compatible type with choices.
            strict: Whether to validate in strict mode.

        Returns:
            The validated object, or None if extraction or validation failed.
        """
        content = get_content_or_none(response)
        if content is None:
            return None
        return self.validate_json(content, strict=strict)

    def validate_response_with_retry(
        self,
        response: "ModelResponse | None",
        retry_count: int,
        max_retries: int,
        enforce_structured_output: bool = True,
        strict: bool = False,
    ) -> SchemaValidationResult:
        """Validate response and determine if retry is needed.

        This method encapsulates the common pattern of:
        1. Validating a response against the schema
        2. Returning whether to retry with an error message

        Args:
            response: The model response to validate.
            retry_count: Current retry attempt count (before this validation).
            max_retries: Maximum allowed retries.
            enforce_structured_output: Whether structured output enforcement is enabled.
                When False, validation is skipped.
            strict: Whether to use strict validation mode.

        Returns:
            SchemaValidationResult indicating validity and retry status.
        """
        # Skip validation if enforcement disabled or plain text schema
        if not enforce_structured_output or self.is_plain_text():
            return SchemaValidationResult(
                is_valid=True,
                should_retry=False,
                error_message=None,
            )

        validated = self.parse_response(response, strict=strict)

        if validated is not None:
            return SchemaValidationResult(
                is_valid=True,
                should_retry=False,
                error_message=None,
            )

        # Validation failed - determine if we should retry
        new_retry_count = retry_count + 1
        error_message = (
            f"Your response could not be parsed as valid JSON "
            f"matching the '{self.name()}' schema. "
            f"Please provide a corrected response that matches "
            f"the expected format."
        )

        if new_retry_count <= max_retries:
            return SchemaValidationResult(
                is_valid=False,
                should_retry=True,
                error_message=error_message,
            )

        return SchemaValidationResult(
            is_valid=False,
            should_retry=False,
            error_message=None,
        )


def _is_subclass_of_base_model_or_dict(t: Any) -> bool:
    if not isinstance(t, type):
        return False

    # If it's a generic alias, 'origin' will be the actual type, e.g. 'list'
    origin = get_origin(t)

    allowed_types = (BaseModel, dict)
    # If it's a generic alias e.g. list[str], then we should check the origin type i.e. list
    return issubclass(origin or t, allowed_types)


def _type_to_str(t: type[Any]) -> str:
    origin = get_origin(t)
    args = get_args(t)

    if origin is None:
        # It's a simple type like `str`, `int`, etc.
        return t.__name__

    if args:
        args_str = ", ".join(_type_to_str(arg) for arg in args)
        return f"{origin.__name__}_{args_str}"

    return str(t)


def _check_keys_match(
    obj: dict[str, Any],
    schema_properties: dict[str, Any],
    required_props: list[str] | None = None,
) -> bool:
    """Check if an object has all required properties from the schema.

    Args:
        obj: The object to check
        schema_properties: The properties from the schema
        required_props: List of required property names

    Returns:
        True if the object has all required properties
    """
    obj_keys = set(obj.keys())

    # If required properties are specified, check that all of them are present
    if required_props:
        if not all(key in obj_keys for key in required_props):
            return False

    # Object should have at least some of the schema properties
    schema_keys = set(schema_properties.keys())
    return bool(schema_keys.intersection(obj_keys))


def _find_matching_object(
    data: object,
    schema_properties: dict[str, Any],
    required_props: list[str] | None = None,
) -> dict[str, Any] | None:
    """Recursively search for an object that matches the schema properties.

    Args:
        data: The data to search
        schema_properties: The properties from the schema
        required_props: List of required property names

    Returns:
        The first matching object or None if no match is found
    """
    if isinstance(data, dict):
        current_obj: JsonObject = cast(JsonObject, data)
        # Check if current object matches
        if _check_keys_match(current_obj, schema_properties, required_props):
            return current_obj

        # Search in each value
        for value in current_obj.values():
            result = _find_matching_object(value, schema_properties, required_props)
            if result:
                return result

    elif isinstance(data, list):
        # Search in each item
        for item in cast(list[object], data):
            result = _find_matching_object(item, schema_properties, required_props)
            if result:
                return result

    return None
