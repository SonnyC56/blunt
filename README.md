# BLUNT

**Basic Lifting and UNprojection Tool**

Turn any photo into a 3D Gaussian Splat in seconds. Supports single images, multi-image reconstruction, and 360 panoramas.

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

The DA360 checkpoint (~1.4GB) is auto-downloaded from HuggingFace when you use `--engine da360`. No manual setup is needed. To use a local checkpoint instead, place it at `da360/DA360_large.pth` (relative to `blunt.py`) and it will be auto-detected as the default for `--mode 360`.

## Three Modes

| Mode | Trigger | Default Engine |
|------|---------|----------------|
| **Single image** | One file (default) | DA3 (best quality) |
| **360 panorama** | `--mode 360` | DA360 (auto-downloads checkpoint) |
| **Multi-image** | Multiple files | DA3 (joint depth + poses) |

```bash
# Single image -- just works
python blunt.py photo.jpg --device cuda

# 360 panorama -- just works (auto-downloads DA360 from HuggingFace)
python blunt.py panorama.jpg --mode 360 --device cuda

# Multi-image -- auto-detected from multiple inputs
python blunt.py img1.jpg img2.jpg img3.jpg --device cuda
```

## Depth Engines

| Engine | Flag | Use Case |
|--------|------|----------|
| **DA3** (default) | `--engine da3` | Best quality, metric depth, predicted intrinsics |
| **DA360** (default for 360) | `--engine da360` | Panoramic-native, no seams, circular padding |
| **DA2** (fallback) | `--engine da2` | Lightweight, no DA3 dependency, runs on CPU |

```bash
# Force DA2 (no DA3 install needed)
python blunt.py photo.jpg --engine da2 --device cuda

# Force DA360 with auto-download (even without local checkpoint)
python blunt.py panorama.jpg --mode 360 --engine da360 --device cuda

# Use a local DA360 checkpoint
python blunt.py panorama.jpg --mode 360 --da360-checkpoint /path/to/DA360_large.pth --device cuda
```

## Quality Controls

| Flag | Default | What it does |
|------|---------|-------------|
| `--resolution N` | 2048 (single), 6144 (DA360) | Controls splat density. Higher = more splats, sharper output |
| `--overlap F` | 1.3 | Gaussian overlap factor. Higher = fewer gaps between splats |
| `--disc-threshold F` | 0.1 | Depth discontinuity filter sensitivity. Lower = more aggressive edge removal |
| `--fov N` | auto (EXIF) | Manual FOV override in degrees |

```bash
# Higher resolution (more splats)
python blunt.py photo.jpg --resolution 3072 --device cuda

# Sharper 360 (even more splats than default 6144)
python blunt.py panorama.jpg --mode 360 --resolution 8192 --device cuda

# Tighter overlap (denser coverage)
python blunt.py photo.jpg --overlap 1.5 --device cuda
```

## Feature Toggles

| Flag | Default | What it does |
|------|---------|-------------|
| `--no-inpaint` | inpaint ON | Disables shadow filling behind depth edges. On by default to fill black gaps |
| `--no-normalize` | normalize ON | Disables position/scale normalization. On by default for consistent viewer scale |
| `--no-sky` | sky ON | Removes sky gaussians entirely (detected via depth + brightness + saturation) |
| `--fast` | OFF | Adaptive stride -- keeps detail in complex areas, skips flat regions. 3-5x fewer splats |
| `--segment` | OFF | Runs SAM2 segmentation, outputs `.segments.bin` + `.segments.json` alongside PLY |

```bash
# Remove sky
python blunt.py outdoor.jpg --no-sky --device cuda

# Fast mode (fewer splats, faster)
python blunt.py photo.jpg --fast --device cuda

# Raw depth values (skip normalization)
python blunt.py photo.jpg --no-normalize --device cuda

# With segmentation
python blunt.py photo.jpg --segment --device cuda
```

## DA2-Only Options

| Flag | Default | What it does |
|------|---------|-------------|
| `--depth-mode` | `relative` | `relative`, `metric-indoor`, or `metric-outdoor`. Real-world scale depth |

```bash
python blunt.py photo.jpg --engine da2 --depth-mode metric-outdoor --device cuda
```

## DA3 Options

| Flag | Default | What it does |
|------|---------|-------------|
| `--da3-model` | `DA3-BASE` | Model variant: `DA3-SMALL`, `DA3-BASE`, `DA3-LARGE`. Large/Giant are CC BY-NC 4.0 |

```bash
python blunt.py photo.jpg --da3-model DA3-LARGE --device cuda
```

## Output

| Flag | Default | What it does |
|------|---------|-------------|
| `-o PATH` | `{input}_splat.ply` | Output file path |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon) |

Standard 3DGS binary PLY files. Load them in:
- [SuperSplat](https://playcanvas.com/supersplat/editor)
- [StorySplat](https://storysplat.com)
- Any 3D Gaussian Splatting viewer

On first run, models are downloaded automatically from Hugging Face (~100MB for DA2, ~500MB for DA3-BASE, ~1.4GB for DA360).

## What Each Feature Does

- **Occlusion inpainting**: Before unprojecting pixels to 3D, detects depth edges and fills the "shadow" behind foreground objects with plausible depth+color. Without this, you get black gaps where the camera can't see behind objects.

- **Normalization**: Centers the scene on its median and scales everything so the 98th percentile fits within +/-50 units. This means DA2, DA3, and DA360 outputs all load at the same scale in any viewer.

- **Near-camera cull**: DA2's relative depth produces a curved shell artifact at the near plane. We cull the closest 2% of splats to fix this. DA3/DA360 use metric depth so this isn't needed.

- **Floater pruning**: Compares each pixel's depth against its local neighborhood. Outliers (likely depth estimation errors) get their opacity reduced to 10%.

- **Edge-aware scale/opacity**: Splats near depth edges are made smaller and slightly more transparent to reduce stretching artifacts at object boundaries.

## How It Works

### Single Image Pipeline

1. **Resize** -- Image resized to processing resolution for depth estimation. Original image (up to 2048px) kept for high-res color sampling.
2. **Depth estimation** -- DA3 (default) or DA2 produces a disparity/depth map. DA2 optionally uses metric depth models for real-world scale.
3. **EXIF focal length** -- Extracts camera focal length from EXIF data (35mm equivalent or raw + crop factor). Falls back to `max(w,h) * 0.7` heuristic.
4. **Depth processing** -- Median filter for noise reduction, then disparity-to-depth conversion via 1/d mapping (or direct meters for DA3 metric mode).
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
2. **Resample** -- Image and depth resampled to unprojection grid resolution (default 6144x3072) for controlled splat density
3. **Spherical unprojection** -- Projects each pixel into 3D using spherical coordinates (longitude/latitude to Cartesian), with latitude-aware Gaussian scaling
4. **Filtering + normalization + output** -- Same filtering pipeline, normalization, and output as standard PLY

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

| Setup | Single Image (2048px) | 360 (6144px) |
|-------|----------------------|---------------|
| NVIDIA RTX 4080 | ~5 seconds | ~16 seconds |
| NVIDIA T4 GPU | ~10 seconds | ~40 seconds |
| Apple M-series (MPS) | ~15 seconds | ~90 seconds |
| CPU | ~30 seconds | ~5 minutes |

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
