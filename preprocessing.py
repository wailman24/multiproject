"""
Part 1 — Pre-processing
BGR ↔ YCbCr colour-space conversion and 4:2:0 chroma sub/upsampling.
"""

import numpy as np
import cv2


def bgr_to_ycbcr(frame_bgr: np.ndarray):
    """Convert a BGR uint8 frame to float32 YCbCr (Y/Cb/Cr all in [0, 255])."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    Y  =  0.299    * R + 0.587    * G + 0.114    * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5      * B + 128.0
    Cr =  0.5      * R - 0.418688 * G - 0.081312  * B + 128.0

    return Y, Cb, Cr


def chroma_subsample_420(Cb: np.ndarray, Cr: np.ndarray):
    """4:2:0 subsampling — keep every other sample in both axes."""
    return Cb[::2, ::2].copy(), Cr[::2, ::2].copy()


def chroma_upsample_420(Cb_sub: np.ndarray, Cr_sub: np.ndarray,
                         target_h: int, target_w: int):
    """Bilinear upsampling back to full luma resolution."""
    Cb_up = cv2.resize(Cb_sub, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_sub, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return Cb_up, Cr_up


def ycbcr_to_bgr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """Convert full-resolution float32 YCbCr back to BGR uint8."""
    R = Y + 1.402    * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772    * (Cb - 128.0)

    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
