#!/usr/bin/env python3
"""
Utility — generate synthetic test frames so you can run the pipeline
without a real video.

Produces a moving coloured circle on a gradient background.

Usage:
  python generate_test_frames.py [--output frames] [--n 30] [--w 320] [--h 240]
"""

import argparse
import os
import cv2
import numpy as np


def generate_frames(output_dir: str, n_frames: int = 30,
                    width: int = 320, height: int = 240) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Colour gradient background
        for y in range(height):
            frame[y, :, 0] = int(y * 200 / height)          # B channel
            frame[y, :, 2] = int((height - y) * 180 / height)  # R channel

        # Slow-panning overlay
        for x in range(width):
            frame[:, x, 1] = int(x * 100 / width + i * 3) % 256  # G channel

        # Moving circle
        cx = int(width  * 0.1 + width  * 0.8 * (i / max(n_frames - 1, 1)))
        cy = int(height * 0.5 + height * 0.3 * np.sin(2 * np.pi * i / n_frames))
        cv2.circle(frame, (cx, cy), max(width // 10, 10), (0, 255, 100), -1)

        # Static rectangle in corner
        cv2.rectangle(frame, (8, 8), (width // 5, height // 5), (200, 80, 40), -1)

        out = os.path.join(output_dir, f'frame_{i:04d}.png')
        cv2.imwrite(out, frame)

    print(f'Generated {n_frames} frames ({width}×{height}) in "{output_dir}/"')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate synthetic test frames')
    p.add_argument('--output', default='frames',  help='Output directory (default: frames)')
    p.add_argument('--n',      type=int, default=30,  help='Number of frames (default: 30)')
    p.add_argument('--w',      type=int, default=320, help='Frame width  (default: 320)')
    p.add_argument('--h',      type=int, default=240, help='Frame height (default: 240)')
    args = p.parse_args()
    generate_frames(args.output, args.n, args.w, args.h)
