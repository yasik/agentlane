"""Shared internal utility helpers for the transport package."""

from ._types import ContentType, SchemaId


def coerce_schema_id(schema_id: SchemaId | str) -> SchemaId:
    """Normalize schema id input into `SchemaId`."""
    if isinstance(schema_id, SchemaId):
        return schema_id
    return SchemaId(schema_id)


def coerce_content_type(content_type: ContentType | str) -> ContentType:
    """Normalize content type input into `ContentType`."""
    if isinstance(content_type, ContentType):
        return content_type
    return ContentType(content_type)
