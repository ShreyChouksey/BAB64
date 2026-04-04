"""
BAB64 C Acceleration Layer
===========================

Patches ImageHash._compress and hash_image with a C implementation
via ctypes. Falls back to pure Python if the shared library is missing.

Usage:
    import bab64_fast  # auto-patches on import
    # or:
    bab64_fast.patch()  # explicit
"""

import ctypes
import os
import sys
import numpy as np
from ctypes import c_uint32, c_int32, c_uint8, c_int, POINTER

# Find the shared library next to this file
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = None
_LIB_NAMES = ["bab64_fast.dylib", "bab64_fast.so"]

for name in _LIB_NAMES:
    path = os.path.join(_dir, name)
    if os.path.exists(path):
        try:
            _lib = ctypes.CDLL(path)
            break
        except OSError:
            pass

if _lib is not None:
    # Set up bab64_compress signature
    _lib.bab64_compress.restype = None
    _lib.bab64_compress.argtypes = [
        POINTER(c_uint32),   # state_in
        POINTER(c_uint8),    # message_block
        POINTER(c_uint32),   # round_constants
        POINTER(c_int32),    # rotations
        POINTER(c_uint8),    # sbox
        c_int,               # num_rounds
        c_int,               # num_rc
        c_int,               # num_rot
        POINTER(c_uint32),   # state_out
    ]

    # Set up bab64_hash_blocks signature
    _lib.bab64_hash_blocks.restype = None
    _lib.bab64_hash_blocks.argtypes = [
        POINTER(c_uint32),   # initial_state
        POINTER(c_uint8),    # image_bytes
        c_int,               # image_len
        c_int,               # block_size
        POINTER(c_uint32),   # round_constants
        POINTER(c_int32),    # rotations
        POINTER(c_uint8),    # sbox
        c_int,               # num_rounds
        c_int,               # num_rc
        c_int,               # num_rot
        POINTER(c_uint32),   # state_out
    ]


def _fast_compress(self, state, message_block, round_constants,
                   rotations, sbox, round_offset=0):
    """C-accelerated drop-in replacement for ImageHash._compress."""
    # Prepare state as contiguous uint32 array
    state_in = np.ascontiguousarray(state, dtype=np.uint32)
    state_out = np.zeros(8, dtype=np.uint32)

    # Pad message block to 32 bytes
    block_bytes = message_block
    if len(block_bytes) < 32:
        block_bytes = block_bytes + b'\x00' * (32 - len(block_bytes))
    block_arr = np.frombuffer(block_bytes[:32], dtype=np.uint8).copy()

    rc = np.ascontiguousarray(round_constants, dtype=np.uint32)
    rot = np.ascontiguousarray(rotations, dtype=np.int32)
    sb = np.ascontiguousarray(sbox, dtype=np.uint8)

    _lib.bab64_compress(
        state_in.ctypes.data_as(POINTER(c_uint32)),
        block_arr.ctypes.data_as(POINTER(c_uint8)),
        rc.ctypes.data_as(POINTER(c_uint32)),
        rot.ctypes.data_as(POINTER(c_int32)),
        sb.ctypes.data_as(POINTER(c_uint8)),
        c_int(self.config.num_rounds),
        c_int(len(rc)),
        c_int(len(rot)),
        state_out.ctypes.data_as(POINTER(c_uint32)),
    )
    return state_out


def _fast_hash_image(self, image):
    """
    C-accelerated hash_image: derives parameters in Python,
    then runs the full Merkle-Damgard chain in C (one FFI call).
    """
    # Step 1: Derive parameters (Python — not the bottleneck)
    round_constants = self._derive_round_constants(image)
    rotations = self._derive_rotations(image)
    sbox = self._derive_sbox(image)
    state = self._derive_initial_state(image)

    # Step 2: Prepare arrays
    img_bytes = np.ascontiguousarray(image, dtype=np.uint8)
    rc = np.ascontiguousarray(round_constants, dtype=np.uint32)
    rot = np.ascontiguousarray(rotations, dtype=np.int32)
    sb = np.ascontiguousarray(sbox, dtype=np.uint8)
    st = np.ascontiguousarray(state, dtype=np.uint32)
    state_out = np.zeros(8, dtype=np.uint32)

    # Step 3: Run full chain in C
    _lib.bab64_hash_blocks(
        st.ctypes.data_as(POINTER(c_uint32)),
        img_bytes.ctypes.data_as(POINTER(c_uint8)),
        c_int(len(image)),
        c_int(self.config.block_size),
        rc.ctypes.data_as(POINTER(c_uint32)),
        rot.ctypes.data_as(POINTER(c_int32)),
        sb.ctypes.data_as(POINTER(c_uint8)),
        c_int(self.config.num_rounds),
        c_int(len(rc)),
        c_int(len(rot)),
        state_out.ctypes.data_as(POINTER(c_uint32)),
    )

    # Step 4: Serialize to bytes (big-endian)
    result = b''
    for word in state_out:
        result += int(word).to_bytes(4, 'big')
    return result


_patched = False


def patch():
    """Monkey-patch ImageHash with C-accelerated methods."""
    global _patched
    if _patched:
        return True
    if _lib is None:
        return False

    from bab64_engine import ImageHash
    ImageHash._compress = _fast_compress
    ImageHash.hash_image = _fast_hash_image
    _patched = True
    return True


def is_available():
    """Check if C acceleration is available."""
    return _lib is not None


# Auto-patch on import
if _lib is not None:
    patch()
