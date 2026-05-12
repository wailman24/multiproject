"""
Part 5 — Evaluation & Visualisation
Quality metrics and a single matplotlib figure covering all pipeline stages.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # work without a display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

from intra_coding  import dct2, idct2, get_quant_table
from inter_coding  import decode_residual


# ── Quality metrics ──────────────────────────────────────────────────────────

def psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two uint8/float images."""
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    return 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / compressed_bytes


def print_metrics(encoder) -> None:
    """Print a summary table of compression statistics."""
    h     = encoder.header
    ratio = compression_ratio(h['original_size'], encoder.compressed_size)
    print('\n=== Quality / Compression Metrics ===')
    print(f'  Compression ratio : {ratio:.3f}×')
    print(f'  Original size     : {h["original_size"] / 1024:.1f} KB')
    print(f'  Compressed size   : {encoder.compressed_size / 1024:.1f} KB')
    print(f'  I-frames          : {encoder.frame_types.count("I")}')
    print(f'  P-frames          : {encoder.frame_types.count("P")}')


# ── Pipeline visualisation ────────────────────────────────────────────────────

def visualise_pipeline(encoder, output_path: str = 'pipeline_visualization.png') -> None:
    """
    Produce a single figure with five rows, one per pipeline stage:
      Row 0 — Original frame sequence
      Row 1 — YCbCr colour-space decomposition
      Row 2 — DCT & quantisation on one 8×8 block
      Row 3 — Motion vectors overlaid on a P-frame
      Row 4 — Residuals, reconstructed frame, compression summary
    """
    vis   = encoder.vis_data
    files = encoder.frame_files
    qf    = encoder.quant_factor

    fig = plt.figure(figsize=(24, 22))
    fig.suptitle('Simplified MPEG-4 Encoder — Pipeline Visualisation',
                 fontsize=17, fontweight='bold', y=0.99)
    gs = GridSpec(5, 5, figure=fig, hspace=0.50, wspace=0.35)

    # ── Row 0: Original frame sequence ───────────────────────────────────────
    n_show = min(5, len(files))
    for k in range(n_show):
        ax  = fig.add_subplot(gs[0, k])
        bgr = cv2.imread(str(files[k]))
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ft  = encoder.frame_types[k] if k < len(encoder.frame_types) else '?'
        ax.set_title(f'Frame {k}  ({ft})', fontsize=9)
        ax.axis('off')

    # ── Row 1: Colour-space decomposition of frame 0 ─────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(cv2.cvtColor(vis['first_bgr'], cv2.COLOR_BGR2RGB))
    ax.set_title('Original (BGR)', fontsize=9)
    ax.axis('off')

    chan_info = [
        (vis['first_Y'],       'Y — Luma',          'gray'),
        (vis['first_Cb'],      'Cb — Chroma-Blue',   'Blues'),
        (vis['first_Cr'],      'Cr — Chroma-Red',    'Reds'),
        (vis['first_Cb_sub'],  f'Cb 4:2:0 ↓  {vis["first_Cb_sub"].shape}', 'Blues'),
    ]
    for k, (ch, title, cmap) in enumerate(chan_info):
        ax = fig.add_subplot(gs[1, k + 1])
        im = ax.imshow(ch, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 2: DCT & Quantisation on one 8×8 block ───────────────────────────
    Qy     = get_quant_table('luma', qf)
    raw8   = vis['first_Y'][0:8, 0:8].astype(np.float32)
    dct_b  = dct2(raw8 - 128.0)
    qnt_b  = np.round(dct_b / Qy).astype(np.float32)
    reco_b = idct2(qnt_b * Qy) + 128.0

    steps = [
        (raw8,   'Raw 8×8 pixels',      'gray',   0.0,  255.0),
        (dct_b,  'DCT coefficients',    'RdBu_r', None, None),
        (qnt_b,  'Quantised coefs',     'RdBu_r', None, None),
        (reco_b, 'Reconstructed block', 'gray',   0.0,  255.0),
        (Qy,     f'Quant table ×{qf}',  'hot',    None, None),
    ]
    for k, (data, title, cmap, vmin, vmax) in enumerate(steps):
        ax = fig.add_subplot(gs[2, k])
        if vmin is None:
            mx = np.max(np.abs(data)); vmin, vmax = -mx, mx
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.tick_params(labelsize=5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 3: Motion vectors ─────────────────────────────────────────────────
    if 'first_pframe_fd' in vis:
        fd   = vis['first_pframe_fd']
        refY = vis['first_pframe_ref_Y']
        mvs  = fd['mvs']             # (mh, mw, 2)
        mh_n, mw_n = mvs.shape[:2]

        ax = fig.add_subplot(gs[3, :3])
        ax.imshow(refY, cmap='gray', interpolation='bilinear')
        ax.set_title('Motion vectors overlaid on reference frame', fontsize=9)
        for i in range(mh_n):
            for j in range(mw_n):
                dy, dx = int(mvs[i, j, 0]), int(mvs[i, j, 1])
                y0, x0 = i * 16 + 8, j * 16 + 8
                if dy != 0 or dx != 0:
                    ax.annotate('',
                                xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax.set_xlim(0, refY.shape[1]); ax.set_ylim(refY.shape[0], 0)
        ax.axis('off')

        # MV magnitude histogram
        ax   = fig.add_subplot(gs[3, 3])
        mags = np.sqrt(mvs[..., 0].astype(float)**2 +
                       mvs[..., 1].astype(float)**2).ravel()
        ax.hist(mags, bins=20, color='coral', edgecolor='black')
        ax.set_title('MV magnitude distribution', fontsize=8)
        ax.set_xlabel('Magnitude (pixels)'); ax.set_ylabel('Count')

        # Zero-MV fraction
        ax   = fig.add_subplot(gs[3, 4])
        pct0 = np.mean(mags == 0) * 100
        ax.bar(['Zero MV', 'Non-zero MV'], [pct0, 100 - pct0],
               color=['steelblue', 'salmon'])
        ax.set_ylim(0, 115); ax.set_title('MV type (%)', fontsize=8)
        for xi, v in enumerate([pct0, 100 - pct0]):
            ax.text(xi, v + 2, f'{v:.1f}%', ha='center', fontsize=8)
    else:
        ax = fig.add_subplot(gs[3, :])
        ax.text(0.5, 0.5, 'No P-frames in this sequence',
                ha='center', va='center', transform=ax.transAxes, fontsize=13)
        ax.axis('off')

    # ── Row 4: Residuals + Reconstruction + Compression summary ──────────────
    if 'first_pframe_fd' in vis:
        fd   = vis['first_pframe_fd']
        recY = vis['first_pframe_rec_Y']
        refY = vis['first_pframe_ref_Y']

        # Decode the quantised residual for exact visualisation
        Qy   = get_quant_table('luma', qf)
        resY = decode_residual(fd['res_coeffs'], Qy)
        oh, ow = fd['orig_shape']
        resY_crop = resY[:oh, :ow]

        ax = fig.add_subplot(gs[4, 0])
        ax.imshow(refY, cmap='gray'); ax.set_title('Reference Y', fontsize=8)
        ax.axis('off')

        ax = fig.add_subplot(gs[4, 1])
        vmax = max(1.0, float(np.max(np.abs(resY_crop))))
        im   = ax.imshow(resY_crop, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title('Decoded residual (Y)', fontsize=8); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[4, 2])
        ax.imshow(recY, cmap='gray'); ax.set_title('Reconstructed P-frame (Y)', fontsize=8)
        ax.axis('off')
    else:
        for k in range(3):
            ax = fig.add_subplot(gs[4, k]); ax.axis('off')

    # Frame-type pie
    ax  = fig.add_subplot(gs[4, 3])
    ic  = encoder.frame_types.count('I')
    pc  = encoder.frame_types.count('P')
    ax.pie([ic, pc],
           labels=[f'I-frames ({ic})', f'P-frames ({pc})'],
           colors=['steelblue', 'coral'], autopct='%1.0f%%', startangle=90)
    ax.set_title('Frame-type breakdown', fontsize=8)

    # Compression bar chart
    ax      = fig.add_subplot(gs[4, 4])
    orig_kb = encoder.header['original_size'] / 1024
    comp_kb = encoder.compressed_size / 1024
    ratio   = orig_kb / comp_kb if comp_kb else 0
    bars    = ax.bar(['Original', 'Compressed'], [orig_kb, comp_kb],
                     color=['steelblue', 'coral'])
    ax.set_title(f'Compression ratio: {ratio:.2f}×', fontsize=8)
    ax.set_ylabel('Size (KB)')
    for bar, v in zip(bars, [orig_kb, comp_kb]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + orig_kb * 0.01,
                f'{v:.0f}', ha='center', va='bottom', fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Visualisation saved → {output_path}')


# ── Experimental analysis plots ───────────────────────────────────────────────

def plot_qf_vs_ratio(frames_dir: str, output_bin: str = '_tmp.bin',
                     qf_values=None, gop: int = 10,
                     output_path: str = 'qf_vs_ratio.png') -> None:
    """
    Encode the same sequence at several quantisation factors and plot
    compression ratio vs QF.
    """
    from encoder import Encoder

    if qf_values is None:
        qf_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    ratios = []
    for qf in qf_values:
        enc = Encoder(frames_dir, output_bin, gop_size=gop, quant_factor=qf)
        enc.encode()
        ratios.append(enc.header['original_size'] / enc.compressed_size)
        print(f'  QF={qf:5.1f}  ratio={ratios[-1]:.2f}×')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(qf_values, ratios, 'o-', color='steelblue', linewidth=2, markersize=7)
    ax.set_xlabel('Quantisation Factor (QF)', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression Ratio vs Quantisation Factor', fontsize=13)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'QF analysis plot saved → {output_path}')


def plot_gop_vs_ratio(frames_dir: str, output_bin: str = '_tmp.bin',
                      gop_values=None, qf: float = 1.0,
                      output_path: str = 'gop_vs_ratio.png') -> None:
    """
    Encode at several GOP sizes and plot compression ratio vs GOP size.
    """
    from encoder import Encoder

    if gop_values is None:
        gop_values = [1, 2, 5, 10, 20, 50]

    ratios = []
    for gop in gop_values:
        enc = Encoder(frames_dir, output_bin, gop_size=gop, quant_factor=qf)
        enc.encode()
        ratios.append(enc.header['original_size'] / enc.compressed_size)
        print(f'  GOP={gop:3d}  ratio={ratios[-1]:.2f}×')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gop_values, ratios, 's-', color='coral', linewidth=2, markersize=7)
    ax.set_xlabel('GOP Size (G)', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('Compression Ratio vs GOP Size', fontsize=13)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'GOP analysis plot saved → {output_path}')
