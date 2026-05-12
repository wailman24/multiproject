"""
Part 4 — Entropy coding
Lossless compression via pickle serialisation + zlib, written to a .bin file.

File layout:
  [8 bytes] magic  MP4SIM\x00\x01
  [4 bytes] big-endian uint32  — compressed payload length
  [N bytes] zlib-compressed pickle of {'header': dict, 'frames': list}
"""

import pickle
import zlib
import struct

_MAGIC = b'MP4SIM\x00\x01'


def encode_to_bin(header: dict, frames: list, out_path: str) -> int:
    """
    Serialise header + frame list with pickle, compress with zlib level 9,
    and write to out_path.  Returns the total number of bytes written.
    """
    payload = zlib.compress(
        pickle.dumps({'header': header, 'frames': frames}, protocol=4),
        level=9,
    )
    with open(out_path, 'wb') as f:
        f.write(_MAGIC)
        f.write(struct.pack('>I', len(payload)))
        f.write(payload)
    return len(_MAGIC) + 4 + len(payload)


def decode_from_bin(in_path: str):
    """
    Read a .bin file produced by encode_to_bin.
    Returns (header: dict, frames: list).
    """
    with open(in_path, 'rb') as f:
        magic = f.read(len(_MAGIC))
        if magic != _MAGIC:
            raise ValueError(f'Bad magic bytes: {magic!r}  (expected {_MAGIC!r})')
        plen    = struct.unpack('>I', f.read(4))[0]
        payload = f.read(plen)

    data = pickle.loads(zlib.decompress(payload))
    return data['header'], data['frames']
