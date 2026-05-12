"""
Part 2 — Intra-frame coding (I-frames)
Block DCT + quantisation on 8×8 tiles, following the JPEG standard tables.
"""

import numpy as np
from scipy.fft import dctn, idctn


# Standard JPEG luminance quantisation table
_Q_LUMA = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)

# Standard JPEG chrominance quantisation table
_Q_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def get_quant_table(channel: str, quant_factor: float) -> np.ndarray:
    """Return the quantisation table scaled by quant_factor, clamped to [1, 255]."""
    base = _Q_LUMA if channel == 'luma' else _Q_CHROMA
    return np.clip(base * quant_factor, 1.0, 255.0)


def dct2(block: np.ndarray) -> np.ndarray:
    """2-D Type-II DCT with orthonormal normalisation."""
    return dctn(block, norm='ortho')


def idct2(block: np.ndarray) -> np.ndarray:
    """2-D inverse DCT (Type-III) with orthonormal normalisation."""
    return idctn(block, norm='ortho')


# ── internal helpers ──────────────────────────────────────────────────────────

def _pad(channel: np.ndarray, block: int = 8):
    """Edge-pad channel so both dims are multiples of block."""
    h, w = channel.shape
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    padded = np.pad(channel, ((0, ph), (0, pw)), mode='edge')
    return padded, (h, w)


def _encode_channel(channel: np.ndarray, Q: np.ndarray):
    """
    Tile the channel into 8×8 blocks, level-shift by 128, apply DCT,
    quantise, and return int16 coefficient array + original shape.
    """
    padded, orig_shape = _pad(channel)
    h, w = padded.shape
    bh, bw = h // 8, w // 8

    coeffs = np.zeros((bh, bw, 8, 8), dtype=np.int16)
    for i in range(bh):
        for j in range(bw):
            block = padded[i*8:(i+1)*8, j*8:(j+1)*8] - 128.0
            coeffs[i, j] = np.round(dct2(block) / Q).astype(np.int16)

    return coeffs, orig_shape


def _decode_channel(coeffs: np.ndarray, orig_shape: tuple,
                    Q: np.ndarray) -> np.ndarray:
    """
    Dequantise int16 coefficients, apply IDCT, undo level shift,
    and crop back to orig_shape.
    """
    bh, bw = coeffs.shape[:2]
    out = np.zeros((bh * 8, bw * 8), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            dct_block = coeffs[i, j].astype(np.float32) * Q
            out[i*8:(i+1)*8, j*8:(j+1)*8] = idct2(dct_block) + 128.0

    oh, ow = orig_shape
    return np.clip(out[:oh, :ow], 0.0, 255.0)


# ── public API ────────────────────────────────────────────────────────────────

def encode_iframe(Y: np.ndarray, Cb_sub: np.ndarray, Cr_sub: np.ndarray,
                  quant_factor: float = 1.0) -> dict:
    """Encode a single I-frame. Y is full-res; Cb_sub/Cr_sub are 4:2:0 subsampled."""
    Qy = get_quant_table('luma',   quant_factor)
    Qc = get_quant_table('chroma', quant_factor)
    return {
        'type': 'I',
        'Y':    _encode_channel(Y,      Qy),
        'Cb':   _encode_channel(Cb_sub, Qc),
        'Cr':   _encode_channel(Cr_sub, Qc),
    }


def decode_iframe(frame: dict, quant_factor: float = 1.0):
    """Decode an I-frame. Returns (Y_full, Cb_sub, Cr_sub) as float32."""
    Qy = get_quant_table('luma',   quant_factor)
    Qc = get_quant_table('chroma', quant_factor)
    Y      = _decode_channel(*frame['Y'],  Qy)
    Cb_sub = _decode_channel(*frame['Cb'], Qc)
    Cr_sub = _decode_channel(*frame['Cr'], Qc)
    return Y, Cb_sub, Cr_sub
