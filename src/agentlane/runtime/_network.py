"""Private runtime networking helpers."""

_DEFAULT_HOST = "127.0.0.1"
"""Default local host used when a bind address omits the host part."""


def resolve_bound_address(address: str, port: int) -> str:
    """Return a concrete host:port string after an OS-assigned bind.

    Args:
        address: Requested bind address, which may use port `0`.
        port: Actual bound port returned by the server.

    Returns:
        str: Concrete address using the requested host or the local default host.
    """
    host, _, _ = address.rpartition(":")
    if not host:
        host = _DEFAULT_HOST
    return f"{host}:{port}"
