"""
Part 3 — Inter-frame coding (P-frames)
Full-search block matching on 16×16 macroblocks; DCT + quantisation on
the motion-compensated residual.  Cb/Cr are coded intra (like I-frames).
"""

import numpy as np
from intra_coding import _encode_channel, _decode_channel, get_quant_table, dct2, idct2

MB = 16  # macroblock size in pixels


# ── helpers ───────────────────────────────────────────────────────────────────

def _pad_mb(arr: np.ndarray) -> np.ndarray:
    """Edge-pad to the next multiple of MB."""
    h, w = arr.shape
    ph = (MB - h % MB) % MB
    pw = (MB - w % MB) % MB
    return np.pad(arr, ((0, ph), (0, pw)), mode='edge')


def _motion_estimate(cur_blk: np.ndarray, ref_pad: np.ndarray,
                     r0: int, c0: int, S: int) -> tuple:
    """
    Full-search block matching within a ±S pixel window.
    Returns the (dy, dx) motion vector that minimises SAD.
    """
    h, w   = ref_pad.shape
    best   = (0, 0)
    best_s = float('inf')
    cur_f  = cur_blk.astype(np.float32)

    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            rr, cc = r0 + dy, c0 + dx
            # skip candidates that fall outside the padded reference
            if rr < 0 or cc < 0 or rr + MB > h or cc + MB > w:
                continue
            sad = np.sum(np.abs(cur_f - ref_pad[rr:rr+MB, cc:cc+MB].astype(np.float32)))
            if sad < best_s:
                best_s = sad
                best   = (dy, dx)

    return best


def decode_residual(res_coeffs: np.ndarray, Qy: np.ndarray) -> np.ndarray:
    """Reconstruct residual image from quantised DCT coefficients."""
    rbh, rbw = res_coeffs.shape[:2]
    resY = np.zeros((rbh * 8, rbw * 8), dtype=np.float32)
    for i in range(rbh):
        for j in range(rbw):
            resY[i*8:(i+1)*8, j*8:(j+1)*8] = idct2(res_coeffs[i, j].astype(np.float32) * Qy)
    return resY


# ── public API ────────────────────────────────────────────────────────────────

def encode_pframe(Y, Cb_sub, Cr_sub, ref_Y, ref_Cb_sub, ref_Cr_sub,
                  quant_factor: float = 1.0, search_window: int = 8) -> dict:
    """
    Encode a P-frame.
    • Y channel:   16×16 block matching + DCT/quantise residual.
    • Cb/Cr:       coded intra (same as I-frame).
    Returns a dict with motion vectors and quantised residual coefficients.
    """
    Qy = get_quant_table('luma',   quant_factor)
    Qc = get_quant_table('chroma', quant_factor)

    orig_h, orig_w = Y.shape
    Yp  = _pad_mb(Y)
    rYp = _pad_mb(ref_Y)
    mh  = Yp.shape[0] // MB
    mw  = Yp.shape[1] // MB

    mvs  = np.zeros((mh, mw, 2), dtype=np.int16)
    resY = np.zeros_like(Yp, dtype=np.float32)

    for i in range(mh):
        for j in range(mw):
            r0, c0 = i * MB, j * MB
            cur_blk = Yp[r0:r0+MB, c0:c0+MB]
            dy, dx  = _motion_estimate(cur_blk, rYp, r0, c0, search_window)
            mvs[i, j] = (dy, dx)

            pred = rYp[r0+dy:r0+dy+MB, c0+dx:c0+dx+MB]
            resY[r0:r0+MB, c0:c0+MB] = cur_blk.astype(np.float32) - pred.astype(np.float32)

    # DCT-quantise residual in 8×8 sub-blocks (residuals are ~0-centred; no level shift)
    rh, rw     = resY.shape
    rbh, rbw   = rh // 8, rw // 8
    res_coeffs = np.zeros((rbh, rbw, 8, 8), dtype=np.int16)

    for i in range(rbh):
        for j in range(rbw):
            blk = resY[i*8:(i+1)*8, j*8:(j+1)*8]
            res_coeffs[i, j] = np.round(dct2(blk) / Qy).astype(np.int16)

    return {
        'type':       'P',
        'orig_shape': (orig_h, orig_w),
        'mvs':        mvs,
        'res_coeffs': res_coeffs,
        'Cb':         _encode_channel(Cb_sub, Qc),
        'Cr':         _encode_channel(Cr_sub, Qc),
    }


def decode_pframe(frame: dict, ref_Y, ref_Cb_sub, ref_Cr_sub,
                  quant_factor: float = 1.0):
    """
    Decode a P-frame.
    Reconstructs Y by adding the decoded residual to the motion-compensated
    prediction from ref_Y.  Returns (Y, Cb_sub, Cr_sub) as float32.
    """
    Qy = get_quant_table('luma',   quant_factor)
    Qc = get_quant_table('chroma', quant_factor)

    orig_h, orig_w = frame['orig_shape']
    mvs            = frame['mvs']
    mh, mw         = mvs.shape[:2]

    rYp  = _pad_mb(ref_Y)
    resY = decode_residual(frame['res_coeffs'], Qy)

    Yp = np.zeros_like(rYp, dtype=np.float32)
    for i in range(mh):
        for j in range(mw):
            r0, c0 = i * MB, j * MB
            dy, dx = int(mvs[i, j, 0]), int(mvs[i, j, 1])
            pred   = rYp[r0+dy:r0+dy+MB, c0+dx:c0+dx+MB].astype(np.float32)
            Yp[r0:r0+MB, c0:c0+MB] = pred + resY[r0:r0+MB, c0:c0+MB]

    Y      = np.clip(Yp[:orig_h, :orig_w], 0.0, 255.0)
    Cb_sub = _decode_channel(*frame['Cb'], Qc)
    Cr_sub = _decode_channel(*frame['Cr'], Qc)
    return Y, Cb_sub, Cr_sub
