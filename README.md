# BLUNT

**Basic Lifting and UNprojection Tool**

Turn any photo into a 3D Gaussian Splat in seconds. Supports single images, multi-image reconstruction, and 360 panoramas.

Three depth engines:
- **DA3** (default for single + multi) -- Depth Anything 3. Best quality, multi-image support. Apache 2.0 (base/small).
- **DA360** (default for 360 when checkpoint found) -- Panoramic-native depth with circular padding. No seam artifacts for 360. Auto-downloads from HuggingFace when using `--engine da360`. MIT.
- **DA2** -- Depth Anything V2 Small. Lightweight fallback, runs on CPU with no extra dependencies. Apache 2.0.

### Default Engine Selection

| Mode | Default Engine | Condition |
|------|---------------|-----------|
| `single` | DA3 | Always |
| `multi` | DA3 | Always (DA3 required) |
| `360` | DA360 | If a local checkpoint is found in `da360/DA360_large.pth` |
| `360` | DA2 | Fallback when no local checkpoint is found |

To force DA360 with auto-download (no local checkpoint needed):

```bash
python blunt.py panorama.jpg --mode 360 --engine da360 --device cuda
```

## Installation

Requires **Python 3.9+** and **pip**.

### Step 1: Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Depth Anything 3 (REQUIRED -- default engine)

DA3 is the default depth engine for single and multi-image modes. You must install it before running BLUNT unless you plan to always use `--engine da2`.

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3 && pip install -e .
```

### Step 3 (optional): DA360 for 360 panoramas

The DA360 checkpoint (~1.4GB) is auto-downloaded from HuggingFace when you use `--engine da360`. No manual setup is needed. To use a local checkpoint instead:

```bash
python blunt.py panorama.jpg --mode 360 --da360-checkpoint /path/to/DA360_large.pth --device cuda
```

If you place the checkpoint at `da360/DA360_large.pth` (relative to `blunt.py`), it will be auto-detected and DA360 becomes the default engine for 360 mode.

## Quick Start

```bash
# Single image (default: DA3 engine)
python blunt.py photo.jpg --device cuda

# Single image with DA2 (lightweight fallback, no DA3 dependency needed)
python blunt.py photo.jpg --engine da2 --device cuda

# Multi-image reconstruction (DA3: consistent depth + auto camera poses)
python blunt.py img1.jpg img2.jpg img3.jpg --mode multi --device cuda

# 360 panorama (force DA360 with auto-download from HuggingFace)
python blunt.py panorama.jpg --mode 360 --engine da360 --device cuda

# 360 panorama with local DA360 checkpoint
python blunt.py panorama.jpg --mode 360 --da360-checkpoint DA360_large.pth --device cuda

# 360 panorama with DA2 fallback (no checkpoint needed)
python blunt.py panorama.jpg --mode 360 --device cuda

# Apple Silicon GPU
python blunt.py photo.jpg --device mps

# Metric depth (DA2 only, real-world scale)
python blunt.py photo.jpg --engine da2 --depth-mode metric-outdoor --device cuda

# Fast mode (3-5x fewer splats)
python blunt.py photo.jpg --fast --device cuda

# Skip sky Gaussians
python blunt.py outdoor.jpg --no-sky --device cuda

# Raw depth values (skip normalization)
python blunt.py photo.jpg --no-normalize --device cuda

# Manual FOV override
python blunt.py photo.jpg --fov 90 --device cuda
```

On first run, models are downloaded automatically from Hugging Face (~100MB for DA2, ~500MB for DA3-BASE, ~1.4GB for DA360).

## Output

Standard 3DGS binary PLY files. Load them in:
- [SuperSplat](https://playcanvas.com/supersplat/editor)
- [StorySplat](https://storysplat.com)
- Any 3D Gaussian Splatting viewer

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | auto | `single`, `360`, or `multi`. Auto-detected from input count (multiple files = `multi`, single file = `single`). |
| `--engine` | auto | `da2`, `da3`, or `da360`. Auto-selected based on mode (see [Default Engine Selection](#default-engine-selection)). |
| `--resolution` | `2048` | Processing resolution in pixels (higher = more splats). |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon). |
| `--depth-mode` | `relative` | `relative`, `metric-indoor`, or `metric-outdoor`. DA2 engine only. |
| `--fov` | auto | Manual FOV override in degrees. Otherwise extracted from EXIF or falls back to `max(w,h) * 0.7` heuristic. |
| `--no-sky` | off | Remove sky Gaussians entirely (detected via depth + brightness + saturation + connected components). |
| `--fast` | off | Adaptive stride -- keeps detail in high-frequency areas, skips flat regions. 3-5x fewer splats. |
| `--no-normalize` | off | Skip position/scale normalization. Outputs raw depth values instead of fitting the scene to a standard bounding volume. |
| `--no-inpaint` | off | Disable occlusion inpainting (shadow filling behind depth edges). |
| `--overlap` | `1.3` | Gaussian overlap factor (prevents gaps between splats). |
| `--disc-threshold` | `0.1` | Depth discontinuity threshold for flying pixel removal. |
| `--da3-model` | `DA3-BASE` | DA3 model variant: `DA3-SMALL`, `DA3-BASE`, `DA3-LARGE`. Note: `DA3-LARGE` and `DA3-GIANT` are CC BY-NC 4.0 (non-commercial). |
| `--da360-checkpoint` | -- | Path to DA360 checkpoint file. If omitted, BLUNT looks for `da360/DA360_large.pth` relative to the script. |
| `--segment` | off | Run SAM2 segmentation (generates `.segments.bin` + `.segments.json` alongside the PLY). |
| `-o` | `{input}_splat.ply` | Output file path. |

## How It Works

### Single Image Pipeline

1. **Resize** -- Image resized to processing resolution for depth estimation. Original image (up to 2048px) kept for high-res color sampling.
2. **Depth estimation** -- DA3 (default) or DA2 produces a disparity/depth map. DA2 optionally uses metric depth models for real-world scale.
3. **EXIF focal length** -- Extracts camera focal length from EXIF data (35mm equivalent or raw + crop factor). Falls back to `max(w,h) * 0.7` heuristic.
4. **Depth processing** -- Median filter for noise reduction, then disparity-to-depth conversion via 1/d mapping (or direct meters for DA2 metric mode).
5. **Occlusion inpainting** -- Detects depth discontinuities and fills the shadow regions behind foreground edges by inpainting both depth and color. Prevents black gaps in the final splat. Disable with `--no-inpaint`.
6. **Unprojection** -- Projects each pixel into 3D using a pinhole camera model (DA3 uses predicted intrinsics).
7. **Filtering** -- Removes bad splats:
   - **Depth discontinuity filter** -- Removes stretched "flying pixels" at object boundaries
   - **Near-camera cull** -- Removes closest 2% by z-depth to eliminate curved shell artifacts (DA2 relative mode only)
   - **Sky masking** (with `--no-sky`) -- Detects and removes sky pixels using depth + brightness + saturation + connected component analysis
   - **Floater pruning** -- Reduces opacity of depth outliers using local median/std comparison
   - **Adaptive stride** (with `--fast`) -- Keeps dense sampling in high-detail areas, skips flat regions
8. **Gaussian generation** -- Creates a 3D Gaussian per surviving pixel with position, SH color, edge-aware scale, camera-facing rotation, and edge-aware opacity.
9. **Normalization** -- Centers the scene on its median position and scales it to fit a standard bounding volume (98th percentile extent mapped to +/-50 units). Gaussian sizes are scaled accordingly. Skip with `--no-normalize`.
10. **PLY output** -- Standard 3DGS binary PLY file.

### 360 Panorama Pipeline (DA360)

Single-pass panoramic depth estimation -- no cube faces, no seams:

1. **DA360 depth** -- Runs depth estimation directly on the equirectangular image using circular padding to handle 360 wrap-around and a learned shift parameter for scale-invariant output
2. **Spherical unprojection** -- Projects each pixel into 3D using spherical coordinates (longitude/latitude to Cartesian), with latitude-aware Gaussian scaling
3. **Filtering + normalization + output** -- Same filtering pipeline, normalization, and output as standard PLY

### 360 Panorama Pipeline (DA2 fallback)

For equirectangular panoramas (2:1 aspect ratio):

1. **Cube face extraction** -- 6 perspective faces at 95 FOV
2. **Per-face processing** -- Depth estimation, unprojection, and filtering per face
3. **World-space transform** -- Rotates each face's splats into shared coordinates
4. **Merge + normalization + output** -- Concatenates all faces, normalizes, and writes a single PLY

### Multi-Image Pipeline (DA3)

Reconstructs 3D from multiple perspective images:

1. **DA3 inference** -- Runs Depth Anything 3 on all images jointly, producing consistent depth maps and camera poses
2. **Per-image unprojection** -- Projects each image's depth into camera-local 3D using predicted intrinsics
3. **World-space transform** -- Transforms each image's splats into shared world coordinates using predicted extrinsics
4. **Merge + normalization + output** -- Concatenates all images' splats, normalizes, and writes a single PLY

## Programmatic Usage

```python
from blunt import load_da3_model, generate_single_da3, generate_multi, write_ply, normalize_gaussians

# DA3: Single image (default, best quality)
da3 = load_da3_model("DA3-BASE", "cuda")
gaussians = generate_single_da3("photo.jpg", da3)
gaussians = normalize_gaussians(gaussians)
write_ply(gaussians, "output.ply")

# DA3: Multi-image reconstruction
gaussians = generate_multi(["img1.jpg", "img2.jpg", "img3.jpg"], da3)
gaussians = normalize_gaussians(gaussians)
write_ply(gaussians, "output_multi.ply")

# DA360: 360 panorama (auto-downloads checkpoint)
from blunt import generate_360_da360
from da360 import load_da360_model
da360_model, h, w = load_da360_model(device="cuda")  # auto-downloads from HF
from PIL import Image
pano = Image.open("panorama.jpg")
gaussians = generate_360_da360(pano, da360_model, h, w, "cuda")
gaussians = normalize_gaussians(gaussians)
write_ply(gaussians, "output_360.ply")

# DA2: Lightweight fallback (no DA3 dependency needed)
from blunt import load_depth_model, generate_single
processor, model = load_depth_model("cuda")
image = Image.open("photo.jpg")
gaussians = generate_single(image, processor, model, device="cuda")
gaussians = normalize_gaussians(gaussians)
write_ply(gaussians, "output_da2.ply")
```

## Performance

| Setup | Single Image (2048px) | 360 (2048px) |
|-------|----------------------|---------------|
| NVIDIA T4 GPU | ~5 seconds | ~30 seconds |
| Apple M-series (MPS) | ~10 seconds | ~60 seconds |
| CPU | ~30 seconds | ~3 minutes |

Use `--fast` for 3-5x speedup with minimal quality loss.

## License

MIT -- use it however you want, commercially or otherwise.

### Dependency Licenses

| Component | License |
|-----------|---------|
| BLUNT (this project) | MIT |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Apache 2.0 |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (DA3-BASE/SMALL) | Apache 2.0 |
| [DA360](https://github.com/Insta360-Research-Team/DA360) | MIT |
| [PyTorch](https://github.com/pytorch/pytorch) / torchvision | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [Pillow](https://github.com/python-pillow/Pillow) | HPND (permissive) |
| [transformers](https://github.com/huggingface/transformers) / huggingface_hub | Apache 2.0 |
| [SciPy](https://github.com/scipy/scipy) | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv) (headless) | BSD-3-Clause |

All core dependencies are permissively licensed. Note: DA3-LARGE and DA3-GIANT models are CC BY-NC 4.0 (non-commercial). Use DA3-BASE or DA3-SMALL for commercial projects.

## Credits

- Depth estimation: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Apache 2.0)
- Multi-image depth: [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- Panoramic depth: [DA360](https://github.com/Insta360-Research-Team/DA360) by Insta360 Research Team
- Built by [StorySplat](https://storysplat.com)
