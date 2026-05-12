#!/usr/bin/env python3
"""
Simplified MPEG-4 Video Encoder/Decoder — CLI entry point.

Sub-commands
------------
  encode      Encode a folder of frames into a .bin file.
  decode      Decode a .bin file back into PNG frames.
  visualise   Encode + produce the full pipeline visualisation figure.
  analyse     Run QF-sweep and GOP-sweep experiments and save plots.

Examples
--------
  python main.py encode     --frames frames/ --output video.bin --gop 10 --qf 1.0 --sw 8
  python main.py decode     --input  video.bin --output decoded/
  python main.py visualise  --frames frames/ --output video.bin --vis pipeline.png
  python main.py analyse    --frames frames/ --output _tmp.bin
"""

import argparse
import sys

from encoder    import Encoder
from decoder    import Decoder
from evaluation import visualise_pipeline, print_metrics, plot_qf_vs_ratio, plot_gop_vs_ratio


# ── sub-command handlers ──────────────────────────────────────────────────────

def cmd_encode(args) -> None:
    enc = Encoder(
        frames_dir    = args.frames,
        output_file   = args.output,
        gop_size      = args.gop,
        quant_factor  = args.qf,
        search_window = args.sw,
    )
    enc.encode()
    print_metrics(enc)
    if args.vis:
        visualise_pipeline(enc, output_path=args.vis)


def cmd_decode(args) -> None:
    dec = Decoder(args.input, args.output)
    dec.decode()


def cmd_visualise(args) -> None:
    enc = Encoder(
        frames_dir    = args.frames,
        output_file   = args.output,
        gop_size      = args.gop,
        quant_factor  = args.qf,
        search_window = args.sw,
    )
    enc.encode()
    print_metrics(enc)
    visualise_pipeline(enc, output_path=args.vis)


def cmd_analyse(args) -> None:
    print('=== QF sweep ===')
    plot_qf_vs_ratio(
        frames_dir  = args.frames,
        output_bin  = args.output,
        gop         = args.gop,
        output_path = 'qf_vs_ratio.png',
    )
    print('\n=== GOP sweep ===')
    plot_gop_vs_ratio(
        frames_dir  = args.frames,
        output_bin  = args.output,
        qf          = args.qf,
        output_path = 'gop_vs_ratio.png',
    )


# ── argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Simplified MPEG-4 Video Encoder / Decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # shared encoder arguments
    def add_enc_args(p):
        p.add_argument('--frames',  required=True, help='Input frames directory')
        p.add_argument('--output',  required=True, help='Output .bin file path')
        p.add_argument('--gop',  type=int,   default=10,  metavar='G',
                       help='GOP size — every G-th frame is an I-frame (default 10)')
        p.add_argument('--qf',   type=float, default=1.0, metavar='QF',
                       help='Quantisation factor; >1 = more lossy (default 1.0)')
        p.add_argument('--sw',   type=int,   default=8,   metavar='S',
                       help='Motion-search window ±S pixels (default 8)')

    # encode
    pe = sub.add_parser('encode', help='Encode frames → .bin')
    add_enc_args(pe)
    pe.add_argument('--vis', default=None, metavar='PATH',
                    help='Also save pipeline visualisation to this file')

    # decode
    pd = sub.add_parser('decode', help='Decode .bin → frames')
    pd.add_argument('--input',  required=True, help='Input .bin file')
    pd.add_argument('--output', required=True, help='Output directory for decoded frames')

    # visualise
    pv = sub.add_parser('visualise', help='Encode + save pipeline figure')
    add_enc_args(pv)
    pv.add_argument('--vis', default='pipeline_visualization.png',
                    help='Output path for the visualisation PNG (default: pipeline_visualization.png)')

    # analyse
    pa = sub.add_parser('analyse', help='Run QF and GOP sweep experiments')
    add_enc_args(pa)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        'encode':    cmd_encode,
        'decode':    cmd_decode,
        'visualise': cmd_visualise,
        'analyse':   cmd_analyse,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()
