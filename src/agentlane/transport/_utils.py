"""Shared internal utility helpers for the transport package."""

from ._types import ContentType, SchemaId


def coerce_schema_id(schema_id: SchemaId | str) -> SchemaId:
    """Normalize schema id input into `SchemaId`.

    Args:
        schema_id: Schema id string or typed wrapper.

    Returns:
        SchemaId: Normalized schema id value.
    """
    if isinstance(schema_id, SchemaId):
        return schema_id
    return SchemaId(schema_id)


def coerce_content_type(content_type: ContentType | str) -> ContentType:
    """Normalize content type input into `ContentType`.

    Args:
        content_type: Content type string or typed wrapper.

    Returns:
        ContentType: Normalized content type value.
    """
    if isinstance(content_type, ContentType):
        return content_type
    return ContentType(content_type)
