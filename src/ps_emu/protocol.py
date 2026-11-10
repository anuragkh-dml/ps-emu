"""Independent NumPy MessagePack implementation of the OpenPI wire format."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


class ProtocolError(ValueError):
    """A frame is invalid or cannot be represented by the OpenPI protocol."""


def _encode(value: Any) -> dict[bytes, Any]:
    if not isinstance(value, (np.ndarray, np.generic)):
        raise TypeError(f"Unsupported MessagePack value: {type(value).__name__}")
    if value.dtype.kind in "VOc":
        raise TypeError(f"Unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}


def _decode(value: dict[Any, Any]) -> Any:
    try:
        if value.get(b"__ndarray__") is True:
            dtype, shape, raw = np.dtype(value[b"dtype"]), tuple(value[b"shape"]), value[b"data"]
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if expected != len(raw) or dtype.kind in "VOc":
                raise ProtocolError("NumPy array payload length or dtype is invalid")
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        if value.get(b"__npgeneric__") is True:
            dtype = np.dtype(value[b"dtype"])
            if dtype.kind in "VOc":
                raise ProtocolError("NumPy scalar dtype is invalid")
            return dtype.type(value[b"data"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("Malformed NumPy MessagePack marker") from exc
    return value


def pack(value: Any) -> bytes:
    try:
        return msgpack.packb(value, default=_encode, use_bin_type=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Message cannot be encoded as OpenPI MessagePack") from exc


def unpack(frame: bytes, *, max_bytes: int = 64 * 1024 * 1024) -> Any:
    if not isinstance(frame, bytes) or len(frame) > max_bytes:
        raise ProtocolError("Frame is not binary or exceeds configured limit")
    try:
        return msgpack.unpackb(frame, raw=False, object_hook=_decode)
    except (ValueError, TypeError, KeyError, msgpack.ExtraData, msgpack.FormatError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("Invalid OpenPI MessagePack frame") from exc
