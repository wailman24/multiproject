"""
Main Decoder — reconstructs frames from a .bin file produced by the Encoder.
"""

import cv2
from pathlib import Path

from preprocessing  import ycbcr_to_bgr, chroma_upsample_420
from intra_coding   import decode_iframe
from inter_coding   import decode_pframe
from entropy_coding import decode_from_bin


class Decoder:
    """
    Decodes a .bin file back into individual PNG frames.

    Parameters
    ----------
    input_file : path to the .bin file
    output_dir : directory where decoded frames will be written
    """

    def __init__(self, input_file: str, output_dir: str):
        self.input_file = str(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def decode(self) -> dict:
        """Decompress the bitstream and write reconstructed frames to output_dir."""
        # ── Stage 4 (inverse): entropy decoding ─────────────────────────────
        header, frames = decode_from_bin(self.input_file)

        h   = header['height']
        w   = header['width']
        qf  = header['quant_factor']
        n   = header['num_frames']

        print(f'Decoding {n} frames  {w}×{h}  QF={qf}')

        ref_Y = ref_Cb = ref_Cr = None

        for idx, fd in enumerate(frames):
            # ── Stage 2/3 (inverse): intra / inter decoding ──────────────────
            if fd['type'] == 'I':
                Y, Cb_sub, Cr_sub = decode_iframe(fd, qf)
            else:
                Y, Cb_sub, Cr_sub = decode_pframe(fd, ref_Y, ref_Cb, ref_Cr, qf)

            ref_Y, ref_Cb, ref_Cr = Y, Cb_sub, Cr_sub

            # ── Stage 1 (inverse): upsample chroma + colour conversion ────────
            Cb_up, Cr_up = chroma_upsample_420(Cb_sub, Cr_sub, h, w)
            bgr = ycbcr_to_bgr(Y, Cb_up, Cr_up)

            out_path = self.output_dir / f'frame_{idx:04d}.png'
            cv2.imwrite(str(out_path), bgr)
            print(f'  [{idx:4d}/{n-1}]  {fd["type"]}-frame  →  {out_path.name}')

        print(f'\nDecoding complete.  Frames saved to: {self.output_dir}')
        return header
