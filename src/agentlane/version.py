"""Package version primitive resolved from installed distribution metadata."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("agentlane")
except importlib.metadata.PackageNotFoundError:
    # Fallback for source execution paths where package metadata is unavailable.
    __version__ = "0.0.0"
