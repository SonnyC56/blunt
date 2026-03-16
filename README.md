# BLUNT

**Basic Lifting and UNprojection Tool**

Turn any photo into a 3D Gaussian Splat in seconds. Supports single images, multi-image reconstruction, and 360° panoramas.

Three depth engines:
- **DA2** (default) — Depth Anything V2 Small. Lightweight, runs on CPU. Apache 2.0.
- **DA3** — Depth Anything 3. Better quality + multi-image support. Apache 2.0 (base/small).
- **DA360** — Panoramic-native depth with circular padding. No seam artifacts for 360°. MIT.

## Quick Start

Requires **Python 3.9+** and **pip**.

```bash
pip install -r requirements.txt

# Single image (default: DA2 engine, 2048px resolution)
python blunt.py photo.jpg

# Single image with DA3 (better depth quality)
python blunt.py photo.jpg --engine da3

# Multi-image reconstruction (DA3: consistent depth + auto camera poses)
python blunt.py img1.jpg img2.jpg img3.jpg --mode multi

# 360° panorama (cube face method)
python blunt.py panorama.jpg --mode 360

# 360° panorama with DA360 (single-pass, no seams)
python blunt.py panorama.jpg --mode 360 --da360-checkpoint DA360_large.pth

# Use GPU (CUDA or Apple Silicon MPS)
python blunt.py photo.jpg --device cuda
python blunt.py photo.jpg --device mps

# Metric depth (real-world scale)
python blunt.py photo.jpg --depth-mode metric-outdoor

# Fast mode (3-5x fewer splats)
python blunt.py photo.jpg --fast

# Skip sky Gaussians
python blunt.py outdoor.jpg --no-sky

# Manual FOV override
python blunt.py photo.jpg --fov 90
```

On first run, the Depth Anything V2 Small model (~100MB) will be downloaded automatically from Hugging Face.

### Optional: DA3 Setup

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3 && pip install -e .
```

### Optional: DA360 Setup

Download the checkpoint from [Google Drive](https://drive.google.com/drive/folders/1FMLWZfJ_IPKOa_cEbVqrq8_BRkl3oB_2) and pass it with `--da360-checkpoint`.

## Output

Standard 3DGS binary PLY files. Load them in:
- [SuperSplat](https://playcanvas.com/supersplat/editor)
- [StorySplat](https://storysplat.com)
- Any 3D Gaussian Splatting viewer

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | auto | `single`, `360`, or `multi` (auto-detected from input count) |
| `--engine` | auto | `da2`, `da3`, or `da360` (auto-selected from mode) |
| `--resolution` | `2048` | Processing resolution in pixels (higher = more splats) |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon) |
| `--depth-mode` | `relative` | `relative`, `metric-indoor`, or `metric-outdoor` (DA2 only) |
| `--fov` | auto | Manual FOV override in degrees (otherwise uses EXIF or default) |
| `--no-sky` | off | Remove sky Gaussians entirely |
| `--fast` | off | Adaptive stride mode — keeps detail where it matters, skips flat areas |
| `--overlap` | `1.3` | Gaussian overlap factor (prevents gaps between splats) |
| `--disc-threshold` | `0.1` | Depth discontinuity threshold for flying pixel removal |
| `--da3-model` | `DA3-BASE` | DA3 model variant (DA3-SMALL, DA3-BASE, DA3-LARGE) |
| `--da360-checkpoint` | — | Path to DA360 checkpoint file |
| `-o` | `{input}_splat.ply` | Output file path |

## How It Works

### Single Image Pipeline

1. **Resize** — Image resized to processing resolution for depth estimation. Original image (up to 2048px) kept for high-res color sampling.
2. **Depth estimation** — [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) (Apache 2.0) produces a disparity map. Optionally uses metric depth models for real-world scale.
3. **EXIF focal length** — Extracts camera focal length from EXIF data (35mm equivalent or raw + crop factor). Falls back to `max(w,h) * 0.7` heuristic.
4. **Depth processing** — Median filter for noise reduction, then disparity-to-depth conversion via 1/d mapping (or direct meters for metric mode).
5. **Unprojection** — Projects each pixel into 3D using a pinhole camera model.
6. **Filtering** — Removes bad splats:
   - **Depth discontinuity filter** — Removes stretched "flying pixels" at object boundaries
   - **Near-camera cull** — Removes closest 8% by z-depth to eliminate curved shell artifacts
   - **Sky masking** (with `--no-sky`) — Detects and removes sky pixels using depth + brightness + saturation + connected component analysis
   - **Floater pruning** — Reduces opacity of depth outliers using local median/std comparison
   - **Adaptive stride** (with `--fast`) — Keeps dense sampling in high-detail areas, skips flat regions
7. **Gaussian generation** — Creates a 3D Gaussian per surviving pixel with position, SH color, edge-aware scale, camera-facing rotation, and edge-aware opacity.
8. **PLY output** — Standard 3DGS binary PLY file.

### 360° Panorama Pipeline (DA2 fallback)

For equirectangular panoramas (2:1 aspect ratio):

1. **Cube face extraction** — 6 perspective faces at 95° FOV
2. **Per-face processing** — Depth estimation, unprojection, and filtering per face
3. **World-space transform** — Rotates each face's splats into shared coordinates
4. **Merge** — Concatenates all faces into a single PLY

### 360° Panorama Pipeline (DA360)

Single-pass panoramic depth estimation — no cube faces, no seams:

1. **DA360 depth** — Runs depth estimation directly on the equirectangular image using circular padding to handle 360° wrap-around and a learned shift parameter for scale-invariant output
2. **Spherical unprojection** — Projects each pixel into 3D using spherical coordinates (longitude/latitude → Cartesian), with latitude-aware Gaussian scaling
3. **Filtering + output** — Same filtering pipeline, output as standard PLY

### Multi-Image Pipeline (DA3)

Reconstructs 3D from multiple perspective images:

1. **DA3 inference** — Runs Depth Anything 3 on all images jointly, producing consistent depth maps and camera poses
2. **Per-image unprojection** — Projects each image's depth into camera-local 3D using predicted intrinsics
3. **World-space transform** — Transforms each image's splats into shared world coordinates using predicted extrinsics
4. **Merge** — Concatenates all images' splats into a single PLY

## Programmatic Usage

```python
from blunt import load_depth_model, generate_single, generate_360, write_ply
from PIL import Image

# DA2: Single image
processor, model = load_depth_model("cuda")
image = Image.open("photo.jpg")
gaussians = generate_single(image, processor, model, device="cuda")
write_ply(gaussians, "output.ply")

# DA2: 360° panorama
pano = Image.open("panorama.jpg")
gaussians = generate_360(pano, processor, model, device="cuda")
write_ply(gaussians, "output_360.ply")

# DA3: Single image (better quality)
from blunt import load_da3_model, generate_single_da3
da3 = load_da3_model("DA3-BASE", "cuda")
gaussians = generate_single_da3("photo.jpg", da3)
write_ply(gaussians, "output_da3.ply")

# DA3: Multi-image reconstruction
from blunt import generate_multi
gaussians = generate_multi(["img1.jpg", "img2.jpg", "img3.jpg"], da3)
write_ply(gaussians, "output_multi.ply")

# DA360: 360° panorama (single-pass, no seams)
from blunt import generate_360_da360
from da360 import load_da360_model
da360_model, h, w = load_da360_model("DA360_large.pth", "cuda")
pano = Image.open("panorama.jpg")
gaussians = generate_360_da360(pano, da360_model, h, w, "cuda")
write_ply(gaussians, "output_da360.ply")
```

## Performance

| Setup | Single Image (2048px) | 360° (2048px) |
|-------|----------------------|---------------|
| NVIDIA T4 GPU | ~5 seconds | ~30 seconds |
| Apple M-series (MPS) | ~10 seconds | ~60 seconds |
| CPU | ~30 seconds | ~3 minutes |

Use `--fast` for 3-5x speedup with minimal quality loss.

## License

MIT — use it however you want, commercially or otherwise.

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
