"""Storage interfaces used by native file tools and skill loading."""

from ._local import LocalFileSystem
from ._mounted import MountedFileSystem, MountedReader
from ._paths import normalize_relative_path, normalize_virtual_path
from ._types import (
    BinaryReader,
    DirectoryEntry,
    DirectoryLister,
    FileInfo,
    FilePathResolver,
    FileReader,
    FileStat,
    FileWriter,
    ReadableFileSystem,
    SkillReader,
    WritableFileSystem,
)

__all__ = [
    "BinaryReader",
    "DirectoryEntry",
    "DirectoryLister",
    "FileInfo",
    "FilePathResolver",
    "FileReader",
    "FileStat",
    "ReadableFileSystem",
    "WritableFileSystem",
    "FileWriter",
    "LocalFileSystem",
    "MountedReader",
    "MountedFileSystem",
    "SkillReader",
    "normalize_relative_path",
    "normalize_virtual_path",
]
