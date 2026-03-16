# BLUNT

**Basic Lifting and UNprojection Tool**

Turn any photo into a 3D Gaussian Splat in seconds. Supports single images and 360° panoramas.

Every component is permissively licensed (MIT / Apache 2.0 / BSD) — fully safe for commercial use.

## Quick Start

Requires **Python 3.9+** and **pip**.

```bash
pip install -r requirements.txt

# Single image (default 2048px resolution)
python blunt.py photo.jpg

# 360° panorama
python blunt.py panorama.jpg --mode 360

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

## Output

Standard 3DGS binary PLY files. Load them in:
- [SuperSplat](https://playcanvas.com/supersplat/editor)
- [StorySplat](https://storysplat.com)
- Any 3D Gaussian Splatting viewer

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `single` | `single` for photos, `360` for equirectangular panoramas |
| `--resolution` | `2048` | Processing resolution in pixels (higher = more splats) |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (Apple Silicon) |
| `--depth-mode` | `relative` | `relative`, `metric-indoor`, or `metric-outdoor` |
| `--fov` | auto | Manual FOV override in degrees (otherwise uses EXIF or default) |
| `--no-sky` | off | Remove sky Gaussians entirely |
| `--fast` | off | Adaptive stride mode — keeps detail where it matters, skips flat areas |
| `--overlap` | `1.3` | Gaussian overlap factor (prevents gaps between splats) |
| `--disc-threshold` | `0.1` | Depth discontinuity threshold for flying pixel removal |
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

### 360° Panorama Pipeline

For equirectangular panoramas (2:1 aspect ratio):

1. **Cube face extraction** — 6 perspective faces at 95° FOV
2. **Per-face processing** — Depth estimation, unprojection, and filtering per face
3. **World-space transform** — Rotates each face's splats into shared coordinates
4. **Merge** — Concatenates all faces into a single PLY

## Programmatic Usage

```python
from blunt import load_depth_model, generate_single, generate_360, write_ply
from PIL import Image

# Load the depth model once
processor, model = load_depth_model("cuda")  # or "cpu", "mps"

# Single image
image = Image.open("photo.jpg")
gaussians = generate_single(image, processor, model, device="cuda")
write_ply(gaussians, "output.ply")

# 360° panorama
pano = Image.open("panorama.jpg")
gaussians = generate_360(pano, processor, model, device="cuda")
write_ply(gaussians, "output_360.ply")

# Metric depth for real-world scale
processor, model = load_depth_model("cuda", depth_mode="metric-outdoor")
gaussians = generate_single(image, processor, model, device="cuda", depth_mode="metric-outdoor")
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
| [PyTorch](https://github.com/pytorch/pytorch) / torchvision | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [Pillow](https://github.com/python-pillow/Pillow) | HPND (permissive) |
| [transformers](https://github.com/huggingface/transformers) / huggingface_hub | Apache 2.0 |
| [SciPy](https://github.com/scipy/scipy) | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv) (headless) | BSD-3-Clause |

All dependencies are permissively licensed. No GPL, AGPL, or restrictively-licensed components.

## Credits

- Depth estimation: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Apache 2.0)
- Built by [StorySplat](https://storysplat.com)
