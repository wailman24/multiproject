"""
Main Encoder — orchestrates all five pipeline stages for the input frame sequence.
"""

import cv2
import numpy as np
from pathlib import Path

from preprocessing  import bgr_to_ycbcr, chroma_subsample_420
from intra_coding   import encode_iframe, decode_iframe
from inter_coding   import encode_pframe, decode_pframe
from entropy_coding import encode_to_bin


class Encoder:
    """
    Encodes a folder of sequential images into a compressed .bin file.

    Parameters
    ----------
    frames_dir    : directory containing .png / .jpg input frames (sorted)
    output_file   : path for the .bin output
    gop_size      : every gop_size-th frame is an I-frame; the rest are P-frames
    quant_factor  : multiplier applied to the JPEG quantisation tables (>1 = more lossy)
    search_window : ±S pixel search radius for block matching
    """

    def __init__(self, frames_dir: str, output_file: str,
                 gop_size: int = 10, quant_factor: float = 1.0,
                 search_window: int = 8):
        self.frames_dir    = Path(frames_dir)
        self.output_file   = str(output_file)
        self.gop_size      = gop_size
        self.quant_factor  = quant_factor
        self.search_window = search_window

        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        self.frame_files = sorted(
            p for ext in exts for p in self.frames_dir.glob(ext)
        )
        if not self.frame_files:
            raise FileNotFoundError(f'No image frames found in: {frames_dir}')

        # Populated after encode()
        self.encoded_frames: list = []
        self.frame_types:    list = []
        self.compressed_size: int = 0
        self.header:         dict = {}
        self.vis_data:       dict = {}  # intermediate data for visualisation

    # ── main pipeline ─────────────────────────────────────────────────────────

    def encode(self) -> dict:
        """Run the full encoding pipeline and write the .bin file."""
        n = len(self.frame_files)
        print(f'Encoding {n} frames  '
              f'[GOP={self.gop_size}  QF={self.quant_factor}  S=±{self.search_window}]')

        ref_Y = ref_Cb = ref_Cr = None
        probe = cv2.imread(str(self.frame_files[0]))
        h0, w0 = probe.shape[:2]

        for idx, path in enumerate(self.frame_files):
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise IOError(f'Cannot read frame: {path}')

            # ── Stage 1: pre-processing ──────────────────────────────────────
            Y, Cb, Cr      = bgr_to_ycbcr(bgr)
            Cb_sub, Cr_sub = chroma_subsample_420(Cb, Cr)

            # save first-frame data for later visualisation
            if idx == 0:
                self.vis_data.update(
                    first_bgr    = bgr.copy(),
                    first_Y      = Y.copy(),
                    first_Cb     = Cb.copy(),
                    first_Cr     = Cr.copy(),
                    first_Cb_sub = Cb_sub.copy(),
                    first_Cr_sub = Cr_sub.copy(),
                )

            is_iframe = (idx % self.gop_size == 0)

            if is_iframe:
                # ── Stage 2: intra coding ────────────────────────────────────
                fd = encode_iframe(Y, Cb_sub, Cr_sub, self.quant_factor)
                self.frame_types.append('I')
                rec_Y, rec_Cb, rec_Cr = decode_iframe(fd, self.quant_factor)
            else:
                # ── Stage 3: inter coding ────────────────────────────────────
                fd = encode_pframe(Y, Cb_sub, Cr_sub,
                                   ref_Y, ref_Cb, ref_Cr,
                                   self.quant_factor, self.search_window)
                self.frame_types.append('P')
                rec_Y, rec_Cb, rec_Cr = decode_pframe(fd, ref_Y, ref_Cb, ref_Cr,
                                                       self.quant_factor)

                # save first P-frame for visualisation
                if 'first_pframe_fd' not in self.vis_data:
                    self.vis_data['first_pframe_fd']    = fd
                    self.vis_data['first_pframe_ref_Y'] = ref_Y.copy()
                    self.vis_data['first_pframe_rec_Y'] = rec_Y.copy()

            ref_Y, ref_Cb, ref_Cr = rec_Y, rec_Cb, rec_Cr
            self.encoded_frames.append(fd)
            print(f'  [{idx:4d}/{n-1}]  {fd["type"]}-frame')

        original_size = h0 * w0 * 3 * n   # uncompressed RGB byte count

        self.header = {
            'num_frames':    n,
            'width':         w0,
            'height':        h0,
            'gop_size':      self.gop_size,
            'quant_factor':  self.quant_factor,
            'search_window': self.search_window,
            'frame_types':   self.frame_types,
            'original_size': original_size,
        }

        # ── Stage 4: entropy coding ──────────────────────────────────────────
        self.compressed_size = encode_to_bin(
            self.header, self.encoded_frames, self.output_file
        )

        ratio = original_size / self.compressed_size
        print(f'\nEncoding complete.')
        print(f'  Original   : {original_size / 1024:.1f} KB')
        print(f'  Compressed : {self.compressed_size / 1024:.1f} KB')
        print(f'  Ratio      : {ratio:.2f}×')
        print(f'  I-frames   : {self.frame_types.count("I")}')
        print(f'  P-frames   : {self.frame_types.count("P")}')
        return self.header
