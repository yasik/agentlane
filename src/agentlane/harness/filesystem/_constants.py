"""Validation messages for virtual filesystem paths."""

INVALID_PATH_SEPARATORS = "path must use POSIX separators and contain no null bytes"
"""Used in errors when a virtual path contains a backslash or null byte."""

HOME_PREFIX_PATH_ERROR = "path must not contain home-directory prefixes"
"""Used in errors when a virtual path requests home-directory expansion."""
