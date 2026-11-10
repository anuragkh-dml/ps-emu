from __future__ import annotations

import msgpack
import numpy as np
import pytest

from ps_emu.protocol import ProtocolError, pack, unpack


def test_numpy_wire_roundtrip_matches_openpi_marker_contract() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    frame = pack({"x": array, "scalar": np.int32(7), "bytes": b"ok"})
    decoded = unpack(frame)
    assert np.array_equal(decoded["x"], array)
    assert decoded["x"].dtype == array.dtype
    assert decoded["scalar"] == np.int32(7)
    assert decoded["bytes"] == b"ok"
    raw = msgpack.unpackb(frame, raw=False)
    assert raw["x"][b"__ndarray__"] is True
    assert raw["x"][b"dtype"] == "<f4"


def test_codec_rejects_malformed_array_and_oversized_frame() -> None:
    malformed = msgpack.packb(
        {"a": {b"__ndarray__": True, b"data": b"x", b"dtype": "<f4", b"shape": [2]}}
    )
    with pytest.raises(ProtocolError):
        unpack(malformed)
    with pytest.raises(ProtocolError):
        unpack(b"1234", max_bytes=3)
