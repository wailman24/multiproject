# Simplified MPEG-4 Video Encoder Pipeline

A complete Python implementation of a simplified MPEG-4-like video codec, built for the Multimedia Systems module.  
The pipeline takes a folder of image frames, compresses them into a single `.bin` file, and can decode them back into images.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [How the Pipeline Works](#2-how-the-pipeline-works)
   - [Stage 1 — Pre-processing](#stage-1--pre-processing)
   - [Stage 2 — Intra-frame Coding (I-frames)](#stage-2--intra-frame-coding-i-frames)
   - [Stage 3 — Inter-frame Coding (P-frames)](#stage-3--inter-frame-coding-p-frames)
   - [Stage 4 — Entropy Coding](#stage-4--entropy-coding)
   - [Stage 5 — Evaluation & Visualisation](#stage-5--evaluation--visualisation)
3. [Installation](#3-installation)
4. [Running the Project](#4-running-the-project)
5. [All Generated Output Files](#5-all-generated-output-files)
6. [Command Reference](#6-command-reference)
7. [Viewing the Results](#7-viewing-the-results)
8. [Understanding the Parameters](#8-understanding-the-parameters)
9. [File-by-File Code Explanation](#9-file-by-file-code-explanation)

---

## 1. Project Structure

```
multiproj/
│
├── preprocessing.py        # Stage 1 — colour space conversion + chroma subsampling
├── intra_coding.py         # Stage 2 — I-frame DCT + quantisation
├── inter_coding.py         # Stage 3 — P-frame motion estimation + residual coding
├── entropy_coding.py       # Stage 4 — lossless compression into .bin file
├── encoder.py              # Main Encoder class — ties all stages together
├── decoder.py              # Main Decoder class — reverses all stages
├── evaluation.py           # Stage 5 — metrics and matplotlib visualisation
├── main.py                 # CLI entry point (the file you run)
├── generate_test_frames.py # Helper — generates synthetic test frames
└── requirements.txt        # Python dependencies
```

---

## 2. How the Pipeline Works

### The Big Picture

```
Input Frames (PNG/JPG)
        │
        ▼
┌───────────────────┐
│  Stage 1          │  BGR → YCbCr colour space
│  Pre-processing   │  Shrink Cb and Cr channels by 2× (4:2:0 subsampling)
└────────┬──────────┘
         │
         ├──── Every G-th frame ────▶ ┌───────────────────┐
         │                            │  Stage 2           │  Split into 8×8 blocks
         │                            │  I-frame coding    │  DCT + Quantise
         │                            └────────┬──────────┘
         │                                     │
         └──── All other frames ────▶ ┌────────┴──────────┐
                                      │  Stage 3           │  Find motion vectors
                                      │  P-frame coding    │  DCT + Quantise residual
                                      └────────┬──────────┘
                                               │
                                      ┌────────▼──────────┐
                                      │  Stage 4           │  pickle + zlib compress
                                      │  Entropy coding    │  Write to video.bin
                                      └────────┬──────────┘
                                               │
                                      ┌────────▼──────────┐
                                      │  Stage 5           │  Compression ratio
                                      │  Evaluation        │  Pipeline visualisation
                                      └───────────────────┘
```

---

### Stage 1 — Pre-processing

**File:** `preprocessing.py`

**What it does:**

Every input frame is a BGR image (Blue, Green, Red — OpenCV's default). We convert it to **YCbCr**, a colour space designed for compression:

| Channel | Meaning | Why it matters |
|---------|---------|----------------|
| **Y**  | Brightness (luma) | Human eyes are very sensitive to brightness detail |
| **Cb** | Blue-difference chroma | Human eyes are less sensitive to colour detail |
| **Cr** | Red-difference chroma | Human eyes are less sensitive to colour detail |

**Conversion formula (BT.601):**
```
Y  =  0.299·R  + 0.587·G  + 0.114·B
Cb = −0.169·R  − 0.331·G  + 0.500·B  + 128
Cr =  0.500·R  − 0.419·G  − 0.081·B  + 128
```

**4:2:0 Chroma Subsampling:**

After conversion, we keep Y at full resolution but downsample Cb and Cr to half size in both dimensions. For a 320×240 frame:

```
Y  channel →  320×240   (full resolution)
Cb channel →  160×120   (half resolution — every other pixel)
Cr channel →  160×120   (half resolution — every other pixel)
```

This alone reduces the data by about 50% with almost no visible quality loss, because the human eye cannot perceive fine colour detail.

**On decoding:** Cb and Cr are upsampled back to full resolution using bilinear interpolation, then converted back to BGR.

---

### Stage 2 — Intra-frame Coding (I-frames)

**File:** `intra_coding.py`

**What it does:**

Every G-th frame (frame 0, G, 2G, …) is an **I-frame** — compressed entirely on its own, like a JPEG. No reference to any other frame.

**Step-by-step for each channel (Y, Cb, Cr):**

**1. Pad the channel** so its dimensions are multiples of 8.

**2. Split into 8×8 blocks** and level-shift by subtracting 128 (centres values around 0).

**3. Apply 2D DCT** (Discrete Cosine Transform) to each block:
```
Raw pixels (spatial domain)  →  DCT  →  Frequency coefficients
```
The DCT reorganises the block so that:
- Top-left coefficient = average brightness (DC component)
- Other coefficients = increasing levels of fine detail (AC components)
- Most energy concentrates in the top-left corner
- Bottom-right coefficients are usually near zero

**4. Quantise** — divide each coefficient by the quantisation table and round:
```
quantised[i,j] = round( DCT[i,j] / Q[i,j] )
```
The quantisation table has small values top-left (preserve low frequencies) and large values bottom-right (aggressively discard high frequencies). After rounding, most high-frequency coefficients become 0.

**The JPEG luminance quantisation table (default QF=1.0):**
```
16  11  10  16  24  40  51  61
12  12  14  19  26  58  60  55
14  13  16  24  40  57  69  56
14  17  22  29  51  87  80  62
18  22  37  56  68 109 103  77
24  35  55  64  81 104 113  92
49  64  78  87 103 121 120 101
72  92  95  98 112 100 103  99
```

**5. Store as int16** — the quantised coefficients (mostly zeros) are stored compactly.

**Decoding reverses the process:**
```
int16 coefficients  →  multiply by Q  →  IDCT  →  +128  →  clip [0,255]
```

---

### Stage 3 — Inter-frame Coding (P-frames)

**File:** `inter_coding.py`

**What it does:**

All frames between I-frames are **P-frames**. Instead of encoding the full frame, we only encode what *changed* since the last frame. This is the biggest source of compression.

**Key insight:** Consecutive video frames are almost identical. A background stays the same; objects just move slightly. We exploit this with **motion estimation**.

**Step-by-step:**

**1. Divide Y into 16×16 macroblocks:**
```
For a 320×240 frame:  20 columns × 15 rows = 300 macroblocks
```

**2. Block matching (motion estimation) for each macroblock:**

For each 16×16 block in the current frame, search the previous reconstructed frame within a ±S pixel window to find the best matching block. "Best" = lowest SAD (Sum of Absolute Differences).

```
Search all positions (dy, dx) where -S ≤ dy ≤ S and -S ≤ dx ≤ S
For each: SAD = sum of |current_pixel - reference_pixel|
Keep the (dy, dx) with the smallest SAD  →  this is the motion vector
```

**3. Compute the residual:**
```
residual = current_block − reference_block_at(motion_vector)
```
If the prediction was perfect, the residual is all zeros. In practice it's small.

**4. DCT + quantise the residual** (same as Stage 2, but no level-shift since residuals are already ~0-centred).

**5. Store:** motion vectors (int16) + quantised residual coefficients (int16).

**Cb and Cr** in P-frames are coded intra (same as I-frames) since chroma subsampling already handles most temporal redundancy there.

**Decoding:**
```
IDCT(residual coefficients)  →  residual
reference_block + residual   →  reconstructed block
```

---

### Stage 4 — Entropy Coding

**File:** `entropy_coding.py`

**What it does:**

After Stages 1–3, we have a Python list of encoded frame dictionaries containing arrays of integers (lots of zeros, small numbers). We compress this losslessly:

1. **Serialise** with `pickle` (converts Python objects to bytes)
2. **Compress** with `zlib` level 9 (same algorithm as `.zip` / `.gz`)

**Binary file format (video.bin):**
```
Bytes 0–7   : Magic header  "MP4SIM\x00\x01"  (identifies the file type)
Bytes 8–11  : uint32 big-endian  (length of compressed payload in bytes)
Bytes 12+   : zlib-compressed pickle data
```

**Decoding:** read magic → read length → read payload → zlib decompress → unpickle.

---

### Stage 5 — Evaluation & Visualisation

**File:** `evaluation.py`

**Quality metrics computed:**
- **Compression ratio** = original size (bytes) ÷ compressed size (bytes)
- **PSNR** (Peak Signal-to-Noise Ratio) = how close decoded frames are to originals
- **Frame-type breakdown** = count of I-frames vs P-frames

**Pipeline visualisation figure (5 rows):**

| Row | What is shown |
|-----|---------------|
| Row 1 | First 5 input frames, labelled I or P |
| Row 2 | Y, Cb, Cr channels of frame 0, plus 4:2:0 subsampled Cb |
| Row 3 | One 8×8 block: raw pixels → DCT → quantised → reconstructed → quantisation table |
| Row 4 | Motion vectors drawn as red arrows on the reference frame + MV histogram |
| Row 5 | Reference Y / decoded residual / reconstructed P-frame + pie chart + compression bar |

**Analysis plots:**
- `qf_vs_ratio.png` — compression ratio vs quantisation factor
- `gop_vs_ratio.png` — compression ratio vs GOP size

---

## 3. Installation

> You are on Arch Linux. Arch manages Python packages system-wide, so you **must** use a virtual environment.

```bash
# Navigate to the project folder
cd /home/wailarch/Projects/multiproj

# Create a virtual environment called "venv" inside the project
python -m venv venv

# Activate it (you must do this every time you open a new terminal)
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt
```

**To confirm everything installed correctly:**
```bash
python -c "import numpy, cv2, scipy, matplotlib; print('All good!')"
```

**Every time you open a new terminal**, activate the environment first:
```bash
cd /home/wailarch/Projects/multiproj
source venv/bin/activate
```

---

## 4. Running the Project

### Step 1 — Generate test frames

If you do not have your own video frames, generate synthetic ones:

```bash
python generate_test_frames.py --n 30 --w 320 --h 240
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--n` | Number of frames | 30 |
| `--w` | Frame width in pixels | 320 |
| `--h` | Frame height in pixels | 240 |

**Output:** `frames/frame_0000.png` … `frames/frame_0029.png`

> To use your own video: extract frames with `ffmpeg -i myvideo.mp4 frames/frame_%04d.png` and skip this step.

---

### Step 2 — Encode

```bash
python main.py encode --frames frames/ --output video.bin
```

**With custom parameters:**
```bash
python main.py encode --frames frames/ --output video.bin --gop 10 --qf 2.0 --sw 8
```

**Output:** `video.bin`

You will see:
```
Encoding 30 frames  [GOP=10  QF=2.0  S=±8]
  [   0/29]  I-frame
  [   1/29]  P-frame
  ...
Encoding complete.
  Original   : 6750.0 KB
  Compressed : 95.2 KB
  Ratio      : 70.91×
  I-frames   : 3
  P-frames   : 27
```

---

### Step 3 — Decode

```bash
python main.py decode --input video.bin --output decoded/
```

**Output:** `decoded/frame_0000.png` … `decoded/frame_0029.png`

---

### Step 4 — Pipeline Visualisation

```bash
python main.py visualise --frames frames/ --output video.bin --vis pipeline_visualization.png
```

**Output:** `pipeline_visualization.png` — the full 5-row figure.

---

### Step 5 — Experimental Analysis (for the report)

```bash
python main.py analyse --frames frames/ --output _tmp.bin
```

**Output:**
- `qf_vs_ratio.png` — compression ratio vs QF plot
- `gop_vs_ratio.png` — compression ratio vs GOP size plot

> This re-encodes the video multiple times so it takes a few minutes.

---

## 5. All Generated Output Files

| File / Folder | Generated by | What it is |
|---------------|-------------|------------|
| `frames/` | `generate_test_frames.py` | Input video frames (PNG) |
| `video.bin` | `encode` command | Compressed binary video file |
| `decoded/` | `decode` command | Reconstructed frames after decoding |
| `pipeline_visualization.png` | `visualise` command | Full pipeline figure (5 rows) |
| `qf_vs_ratio.png` | `analyse` command | Report graph — QF experiment |
| `gop_vs_ratio.png` | `analyse` command | Report graph — GOP experiment |

---

## 6. Command Reference

```bash
# Encode frames into a .bin file
python main.py encode \
  --frames  <input_dir>   \   # folder containing PNG/JPG frames
  --output  <file.bin>    \   # output compressed file
  --gop     <int>         \   # GOP size (default: 10)
  --qf      <float>       \   # quantisation factor (default: 1.0)
  --sw      <int>         \   # search window ±S pixels (default: 8)
  --vis     <file.png>        # optional: also save visualisation

# Decode a .bin file back to frames
python main.py decode \
  --input   <file.bin>    \   # compressed input file
  --output  <output_dir>      # folder to write decoded frames into

# Encode + produce full pipeline visualisation
python main.py visualise \
  --frames  <input_dir>   \
  --output  <file.bin>    \
  --gop     <int>         \
  --qf      <float>       \
  --sw      <int>         \
  --vis     <file.png>        # output visualisation path (default: pipeline_visualization.png)

# Run QF sweep + GOP sweep and save plots
python main.py analyse \
  --frames  <input_dir>   \
  --output  <tmp.bin>     \
  --gop     <int>         \
  --qf      <float>
```

---

## 7. Viewing the Results

### View the visualisation PNG
```bash
feh pipeline_visualization.png
# or
eog pipeline_visualization.png
```

### View decoded frames as a slideshow
```bash
feh decoded/
# use arrow keys to browse frames
```

### Convert decoded frames to a playable video
```bash
# install ffmpeg if needed
sudo pacman -S ffmpeg mpv

# convert frames to MP4
ffmpeg -framerate 25 -i decoded/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4

# play it
mpv output.mp4
```

---

## 8. Understanding the Parameters

### `--qf` (Quantisation Factor)

Controls the trade-off between **file size** and **image quality**.

| QF value | Effect |
|----------|--------|
| `0.5` | High quality, larger file |
| `1.0` | Balanced (default, standard JPEG tables) |
| `2.0` | More compression, visible blurring |
| `8.0` | Very high compression, blocky artefacts |

> Higher QF = bigger numbers divide the DCT coefficients = more are rounded to zero = smaller file but lower quality.

### `--gop` (Group of Pictures size)

Controls how often a full I-frame is inserted.

| GOP value | Effect |
|-----------|--------|
| `1` | Every frame is an I-frame — largest file, no temporal compression |
| `5` | I-frame every 5 frames — moderate compression |
| `10` | I-frame every 10 frames — default, good compression |
| `30` | I-frame every 30 frames — maximum temporal compression |

> Larger GOP = more P-frames = better compression, but errors accumulate over more frames.

### `--sw` (Search Window)

Controls how far the motion estimator searches for matching blocks.

| SW value | Effect |
|----------|--------|
| `4` | Fast encoding, only catches small motion |
| `8` | Default — good balance of speed vs quality |
| `16` | Slow encoding, catches large motion (fast camera pans) |

> Larger search window = better motion prediction = smaller residuals = better compression, but encoding is much slower (O(S²) per macroblock).

---

## 9. File-by-File Code Explanation

### `preprocessing.py`
- `bgr_to_ycbcr(frame)` — converts one BGR frame to float32 Y, Cb, Cr arrays
- `chroma_subsample_420(Cb, Cr)` — halves Cb and Cr by taking every other sample
- `chroma_upsample_420(Cb, Cr, h, w)` — bilinear resize back to full resolution
- `ycbcr_to_bgr(Y, Cb, Cr)` — converts float32 YCbCr back to uint8 BGR

### `intra_coding.py`
- `get_quant_table(channel, qf)` — returns the scaled JPEG quantisation table
- `dct2(block)` / `idct2(block)` — 2D DCT and inverse DCT via `scipy.fft`
- `_encode_channel(channel, Q)` — pads, tiles into 8×8, DCT, quantise → int16 array
- `_decode_channel(coeffs, shape, Q)` — dequantise, IDCT, unpad → float32 array
- `encode_iframe(Y, Cb, Cr, qf)` — encodes all 3 channels, returns a dict
- `decode_iframe(frame, qf)` — decodes the dict back to Y, Cb, Cr arrays

### `inter_coding.py`
- `_motion_estimate(cur, ref, r0, c0, S)` — full search block matching, returns (dy, dx)
- `encode_pframe(Y, Cb, Cr, ref_Y, ref_Cb, ref_Cr, qf, sw)` — motion estimation + residual DCT
- `decode_pframe(frame, ref_Y, ref_Cb, ref_Cr, qf)` — IDCT residual + motion compensation
- `decode_residual(res_coeffs, Q)` — helper used by both decoder and visualisation

### `entropy_coding.py`
- `encode_to_bin(header, frames, path)` — pickle + zlib → write .bin file, returns bytes written
- `decode_from_bin(path)` — read .bin → zlib decompress → unpickle → return (header, frames)

### `encoder.py`
- `Encoder(frames_dir, output, gop, qf, sw)` — constructor, loads frame file list
- `Encoder.encode()` — runs the full pipeline, populates `self.vis_data` for visualisation, returns header dict

### `decoder.py`
- `Decoder(input_file, output_dir)` — constructor
- `Decoder.decode()` — reads .bin, decodes all frames, writes PNGs, returns header

### `evaluation.py`
- `psnr(orig, recon)` — computes PSNR in dB
- `compression_ratio(orig_bytes, comp_bytes)` — simple division
- `print_metrics(encoder)` — prints a stats table
- `visualise_pipeline(encoder, path)` — draws the 5-row matplotlib figure
- `plot_qf_vs_ratio(frames_dir, ...)` — sweeps QF values, saves plot
- `plot_gop_vs_ratio(frames_dir, ...)` — sweeps GOP values, saves plot

### `main.py`
- Parses command-line arguments with `argparse`
- Sub-commands: `encode`, `decode`, `visualise`, `analyse`
- Each sub-command calls the appropriate class/function from the modules above

### `generate_test_frames.py`
- Creates synthetic frames with a moving circle and colour gradient
- Used when you do not have a real video to test with

---

## Quick Start (all steps in one block)

```bash
cd /home/wailarch/Projects/multiproj

# 1. Setup (one time only)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate test frames
python generate_test_frames.py --n 30 --w 320 --h 240

# 3. Encode
python main.py encode --frames frames/ --output video.bin --gop 10 --qf 1.0

# 4. Decode
python main.py decode --input video.bin --output decoded/

# 5. Visualise
python main.py visualise --frames frames/ --output video.bin --vis pipeline_visualization.png

# 6. Analysis plots for the report
python main.py analyse --frames frames/ --output _tmp.bin

# 7. Convert decoded frames to MP4 and watch
ffmpeg -framerate 25 -i decoded/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
mpv output.mp4
```
