"""
BLUNT -- Basic Lifting and UNprojection Tool
Converts images to 3D Gaussian Splatting PLY files.

Depth engines:
  - DA2: Depth Anything V2 Small (default, lightweight, Apache 2.0)
  - DA3: Depth Anything 3 (better quality, multi-image support, requires depth-anything-3)
  - DA360: Panoramic-native depth with circular padding (auto-downloads from HuggingFace)

Usage:
    python blunt.py input.jpg                                      # Single image (DA3)
    python blunt.py input.jpg --engine da2                         # Single image (DA2 fallback)
    python blunt.py img1.jpg img2.jpg img3.jpg --mode multi        # Multi-image (DA3)
    python blunt.py panorama.jpg --mode 360                        # 360 panorama (DA360, auto-downloads)
    python blunt.py panorama.jpg --mode 360 --da360-checkpoint X   # 360 panorama (local checkpoint)
    python blunt.py input.jpg --resolution 1536                    # Higher res (more splats)
    python blunt.py input.jpg --depth-mode metric-outdoor          # Metric depth
    python blunt.py input.jpg --no-sky                             # Skip sky Gaussians
    python blunt.py input.jpg --fast                               # Adaptive stride (fewer splats)
    python blunt.py input.jpg --fov 60                             # Manual FOV override
    python blunt.py input.jpg --segment                            # SAM2 segmentation
"""

import argparse
import json
import time
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch


# --- Depth Estimation --------------------------------------------------------

DEPTH_MODELS = {
    "relative": "depth-anything/Depth-Anything-V2-Small-hf",
    "metric-indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}


def load_depth_model(device: str = "cpu", depth_mode: str = "relative"):
    """Load Depth Anything V2 Small (Apache 2.0 licensed)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_name = DEPTH_MODELS[depth_mode]
    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("Depth model loaded.")
    return processor, model


def estimate_depth(image: Image.Image, processor, model, device: str = "cpu") -> np.ndarray:
    """Run depth estimation, return depth map as float32 array matching image size."""
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth.astype(np.float32)


# --- DA3: Depth Anything 3 (optional) ----------------------------------------

def load_da3_model(model_name: str = "da3-base", device: str = "cpu"):
    """Load Depth Anything 3 model. Requires: pip install depth-anything-3"""
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        raise ImportError(
            "DA3 engine requires depth-anything-3.\n"
            "Install: git clone https://github.com/ByteDance-Seed/Depth-Anything-3 && cd Depth-Anything-3 && pip install -e ."
        )

    print(f"Loading DA3 model: {model_name}...")
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name.upper()}")
    model = model.to(device=torch.device(device))
    print("DA3 model loaded.")
    return model


def da3_single_depth(image_path: str, da3_model, process_res: int = 504):
    """Run DA3 on a single image, return (depth, image_np, intrinsics)."""
    prediction = da3_model.inference([image_path], process_res=process_res)
    return (
        prediction.depth[0],            # (H, W) float32
        prediction.processed_images[0], # (H, W, 3) uint8
        prediction.intrinsics[0],       # (3, 3) float32
    )


# --- DA360: Panoramic Depth (optional) ---------------------------------------

def load_da360_model_wrapper(checkpoint_path: str, device: str = "cpu"):
    """Load DA360 model from vendored code. Requires checkpoint file."""
    from da360 import load_da360_model
    return load_da360_model(checkpoint_path, device)


# --- EXIF Focal Length --------------------------------------------------------

def extract_focal_from_exif(image: Image.Image, w: int, h: int) -> float | None:
    """Extract focal length from EXIF data, return as pixels or None."""
    try:
        exif = image.getexif()
        if not exif:
            return None

        # Tag 41989: FocalLengthIn35mmFilm (preferred -- already normalized)
        focal_35mm = exif.get(41989)
        if focal_35mm and focal_35mm > 0:
            f_px = focal_35mm / 36.0 * max(w, h)
            print(f"  EXIF focal (35mm equiv): {focal_35mm}mm -> {f_px:.0f}px")
            return f_px

        # Tag 37386: FocalLength (needs sensor size, less reliable)
        focal_mm = exif.get(37386)
        if focal_mm and focal_mm > 0:
            focal_35mm_est = float(focal_mm) * 1.5
            f_px = focal_35mm_est / 36.0 * max(w, h)
            print(f"  EXIF focal: {focal_mm}mm (est 35mm: {focal_35mm_est:.0f}mm) -> {f_px:.0f}px")
            return f_px
    except Exception:
        pass
    return None


# --- Sky Masking --------------------------------------------------------------

def detect_sky_mask(image_np: np.ndarray, metric_depth: np.ndarray, far_plane: float) -> np.ndarray:
    """
    Detect sky pixels conservatively.
    Returns boolean mask where True = sky pixel.
    """
    h, w = metric_depth.shape

    # Far-depth pixels (>85th percentile — catches more sky)
    depth_threshold = np.percentile(metric_depth, 85)
    far_mask = metric_depth > depth_threshold

    # Brightness check: use relative threshold (above image median)
    # This handles overcast/hazy skies that aren't absolutely "bright"
    brightness = np.mean(image_np.astype(np.float32), axis=-1)
    brightness_threshold = max(np.percentile(brightness, 60), 100)
    bright_mask = brightness > brightness_threshold

    # Low saturation check: sky tends to be desaturated
    max_ch = np.max(image_np.astype(np.float32), axis=-1)
    min_ch = np.min(image_np.astype(np.float32), axis=-1)
    saturation = (max_ch - min_ch) / np.maximum(max_ch, 1e-6)
    low_sat_mask = saturation < 0.4

    # Candidate: far AND (bright OR low-saturation)
    candidate_mask = far_mask & (bright_mask | low_sat_mask)

    from scipy.ndimage import label
    labeled, n_labels = label(candidate_mask)
    # Must be connected to the top edge
    top_labels = set(labeled[0, :]) - {0}

    sky_mask = np.zeros((h, w), dtype=bool)
    for lbl in top_labels:
        sky_mask |= (labeled == lbl)

    # Accept if it covers a meaningful portion of the top quarter
    top_coverage = np.mean(sky_mask[:h // 4, :])
    if top_coverage < 0.2:
        return np.zeros((h, w), dtype=bool)

    n_sky = int(np.sum(sky_mask))
    print(f"  Sky mask: {n_sky:,} pixels ({n_sky / (h * w) * 100:.1f}%)")
    return sky_mask


# --- Floater Pruning ---------------------------------------------------------

def prune_floaters(metric_depth: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """
    Reduce opacity of isolated depth outliers (floaters).
    Returns an opacity scale factor (0.0-1.0) per pixel, same shape as metric_depth.
    """
    from scipy.ndimage import median_filter, uniform_filter

    local_median = median_filter(metric_depth, size=5).astype(np.float32)
    # Fast local std via variance formula: std = sqrt(mean(x^2) - mean(x)^2)
    local_mean = uniform_filter(metric_depth.astype(np.float64), size=5)
    local_sq_mean = uniform_filter((metric_depth.astype(np.float64)) ** 2, size=5)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0)).astype(np.float32)
    local_std = np.maximum(local_std, 1e-6)

    deviation = np.abs(metric_depth - local_median)
    floater_mask = (deviation > (2.0 * local_std)) & keep_mask

    opacity_scale = np.ones_like(metric_depth)
    opacity_scale[floater_mask] = 0.1

    n_floaters = int(np.sum(floater_mask))
    if n_floaters > 0:
        print(f"  Floater pruning: {n_floaters:,} splats suppressed")

    return opacity_scale


# --- Adaptive Stride (--fast) ------------------------------------------------

def compute_importance_mask(metric_depth: np.ndarray, image_np: np.ndarray) -> np.ndarray:
    """
    Compute importance per pixel. Returns boolean mask (True = keep).
    High-detail areas keep stride 1, flat areas use stride 2-4.
    """
    h, w = metric_depth.shape

    depth_grad_x = np.abs(np.diff(metric_depth, axis=1, prepend=metric_depth[:, :1]))
    depth_grad_y = np.abs(np.diff(metric_depth, axis=0, prepend=metric_depth[:1, :]))
    depth_grad = depth_grad_x + depth_grad_y

    gray = np.mean(image_np.astype(np.float32), axis=-1)
    img_grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    img_grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    img_grad = img_grad_x + img_grad_y

    dg_max = np.percentile(depth_grad, 98)
    ig_max = np.percentile(img_grad, 98)
    if dg_max > 0:
        depth_grad = np.clip(depth_grad / dg_max, 0, 1)
    if ig_max > 0:
        img_grad = np.clip(img_grad / ig_max, 0, 1)

    importance = np.maximum(depth_grad, img_grad)

    mask = np.zeros((h, w), dtype=bool)

    high = importance > 0.3
    mask |= high

    medium = (importance > 0.1) & ~high
    stride2 = np.zeros((h, w), dtype=bool)
    stride2[::2, ::2] = True
    mask |= (medium & stride2)

    low = ~high & ~medium
    stride4 = np.zeros((h, w), dtype=bool)
    stride4[::4, ::4] = True
    mask |= (low & stride4)

    kept = int(np.sum(mask))
    total = h * w
    print(f"  Adaptive stride: keeping {kept:,}/{total:,} pixels ({kept / total * 100:.1f}%)")
    return mask



# --- Occlusion Inpainting ----------------------------------------------------

def inpaint_occlusions(
    image_np: np.ndarray,
    depth: np.ndarray,
    disc_threshold: float = 0.1,
    dilate_px: int = 5,
    inpaint_radius: int = 7,
) -> tuple:
    """
    Fill shadow regions behind depth discontinuities by inpainting both
    the depth map and color image. Returns (inpainted_image, inpainted_depth).

    image_np and depth must be the same spatial resolution.
    """
    import cv2

    h, w = depth.shape

    # Find depth discontinuities (same gradient logic as the filter)
    grad_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    grad_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
    max_grad = np.maximum(grad_x, grad_y)
    disc_mask = max_grad >= (disc_threshold * np.maximum(depth, 1e-8))

    # Dilate to cover full shadow region behind edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    inpaint_mask = cv2.dilate(disc_mask.astype(np.uint8), kernel, iterations=1)

    n_pixels = int(inpaint_mask.sum())
    if n_pixels == 0:
        return image_np, depth

    pct = n_pixels / (h * w) * 100
    print(f"  Inpainting {n_pixels:,} occlusion pixels ({pct:.1f}%)")

    # Inpaint depth: scale to uint8 range for cv2.inpaint, then scale back
    d_min, d_max = depth.min(), depth.max()
    d_range = d_max - d_min if d_max - d_min > 1e-8 else 1.0
    depth_u8 = ((depth - d_min) / d_range * 255).astype(np.uint8)
    depth_inpainted_u8 = cv2.inpaint(depth_u8, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    depth_inpainted = depth_inpainted_u8.astype(np.float32) / 255.0 * d_range + d_min

    # Only replace masked pixels (keep original data elsewhere)
    mask_bool = inpaint_mask.astype(bool)
    depth_out = depth.copy()
    depth_out[mask_bool] = depth_inpainted[mask_bool]

    # Inpaint color
    image_inpainted = cv2.inpaint(image_np, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    image_out = image_np.copy()
    image_out[mask_bool] = image_inpainted[mask_bool]

    return image_out, depth_out


def inpaint_color_from_mask(image_np: np.ndarray, inpaint_mask: np.ndarray,
                            inpaint_radius: int = 7) -> np.ndarray:
    """Inpaint a color image using a pre-computed binary mask (uint8, 0/1)."""
    import cv2
    mask_bool = inpaint_mask.astype(bool)
    if not np.any(mask_bool):
        return image_np
    image_inpainted = cv2.inpaint(image_np, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    image_out = image_np.copy()
    image_out[mask_bool] = image_inpainted[mask_bool]
    return image_out


# --- Depth to Gaussian Conversion --------------------------------------------

SH_C0 = 0.28209479177387814


def depth_to_gaussians(
    image_np: np.ndarray,
    depth: np.ndarray,
    focal_length: float,
    overlap_factor: float = 1.3,
    depth_disc_threshold: float = 0.1,
    flat_ratio: float = 0.1,
    color_image_np: np.ndarray = None,
    is_metric: bool = False,
    skip_sky: bool = False,
    fast_mode: bool = False,
    return_keep_mask: bool = False,
    inpaint: bool = True,
) -> dict | tuple:
    """
    Convert an RGB image + depth map into 3D Gaussian Splatting parameters.
    """
    from scipy.ndimage import median_filter

    h, w = depth.shape
    cx, cy = w / 2.0, h / 2.0

    # Median filter for noise reduction
    depth = median_filter(depth, size=3).astype(np.float32)

    # Inpaint occlusion shadows before unprojection
    if inpaint:
        import cv2
        image_np, depth = inpaint_occlusions(
            image_np, depth, disc_threshold=depth_disc_threshold,
        )
        if color_image_np is not None and color_image_np.shape[:2] != image_np.shape[:2]:
            # High-res color at different resolution: compute mask from depth,
            # scale mask up to color resolution, then inpaint color separately
            grad_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
            grad_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
            max_grad = np.maximum(grad_x, grad_y)
            disc_mask = max_grad >= (depth_disc_threshold * np.maximum(depth, 1e-8))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask_lo = cv2.dilate(disc_mask.astype(np.uint8), kernel, iterations=1)
            h_hi, w_hi = color_image_np.shape[:2]
            mask_hi = cv2.resize(mask_lo, (w_hi, h_hi), interpolation=cv2.INTER_NEAREST)
            color_image_np = inpaint_color_from_mask(color_image_np, mask_hi)
        elif color_image_np is not None:
            # Same resolution: inpaint directly
            color_image_np, _ = inpaint_occlusions(
                color_image_np, depth, disc_threshold=depth_disc_threshold,
            )

    if is_metric:
        metric_depth = np.clip(depth, 0.01, 1000.0).astype(np.float32)
        far_plane = float(np.percentile(metric_depth, 98))
    else:
        d_min, d_max = np.percentile(depth, [2, 98])
        if d_max - d_min < 1e-6:
            d_max = d_min + 1.0
        depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)

        near_plane = 1.0
        far_plane = 100.0
        disparity = np.maximum(depth_norm, 0.01)
        inv_range = 1.0 / 0.01 - 1.0
        metric_depth = near_plane + (1.0 / disparity - 1.0) / inv_range * (far_plane - near_plane)
        metric_depth = np.clip(metric_depth, near_plane, far_plane).astype(np.float32)

    # Sky detection (only run when skip_sky is requested)
    sky_mask = detect_sky_mask(image_np, metric_depth, far_plane) if skip_sky else np.zeros((h, w), dtype=bool)

    # Build pixel coordinate grids
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Unproject to 3D
    x = (uu - cx) * metric_depth / focal_length
    y = (vv - cy) * metric_depth / focal_length
    z = metric_depth

    # Depth discontinuity filter
    grad_x = np.abs(np.diff(metric_depth, axis=1, prepend=metric_depth[:, :1]))
    grad_y = np.abs(np.diff(metric_depth, axis=0, prepend=metric_depth[:1, :]))
    max_grad = np.maximum(grad_x, grad_y)
    keep_mask = max_grad < (depth_disc_threshold * metric_depth)

    # Near-camera cull (remove closest 2% to avoid curved-shell artifacts, DA2 only)
    if not is_metric:
        z_cull = np.percentile(metric_depth[keep_mask], 2)
        keep_mask &= metric_depth > z_cull

    # Sky handling
    if np.any(sky_mask):
        if skip_sky:
            n_sky = int(np.sum(sky_mask & keep_mask))
            keep_mask &= ~sky_mask
            print(f"  Sky removal: {n_sky:,} splats removed")

    # Floater pruning
    floater_opacity_scale = prune_floaters(metric_depth, keep_mask)

    # Adaptive stride
    if fast_mode:
        importance_mask = compute_importance_mask(metric_depth, image_np)
        keep_mask &= importance_mask

    n_total = h * w
    n_killed = n_total - int(np.sum(keep_mask))
    print(f"  Depth filter: {n_killed:,} splats removed ({n_killed / n_total * 100:.1f}%)")

    # Edge-aware scale factor
    edge_proximity = max_grad[keep_mask] / np.maximum(metric_depth[keep_mask], 1e-6)
    edge_scale_factor = np.clip(1.0 - edge_proximity * 5.0, 0.3, 1.0).astype(np.float32)

    # Apply mask
    x_masked = x[keep_mask]
    y_masked = y[keep_mask]
    z_masked = z[keep_mask]
    d = metric_depth[keep_mask]
    floater_scale = floater_opacity_scale[keep_mask]

    # Color sampling
    if color_image_np is not None and color_image_np.shape[:2] != image_np.shape[:2]:
        h_hi, w_hi = color_image_np.shape[:2]
        scale_x = w_hi / w
        scale_y = h_hi / h
        kept_v, kept_u = np.where(keep_mask)
        hi_u = np.clip((kept_u * scale_x).astype(np.int32), 0, w_hi - 1)
        hi_v = np.clip((kept_v * scale_y).astype(np.int32), 0, h_hi - 1)
        colors = color_image_np[hi_v, hi_u]
    else:
        colors = image_np[keep_mask]

    # Gaussian scale
    pixel_scale = d / focal_length * overlap_factor * edge_scale_factor
    scale_tangent = pixel_scale
    scale_normal = pixel_scale * flat_ratio

    log_scale_0 = np.log(np.maximum(scale_tangent, 1e-7))
    log_scale_1 = np.log(np.maximum(scale_tangent, 1e-7))
    log_scale_2 = np.log(np.maximum(scale_normal, 1e-7))

    # Color: RGB to SH DC
    rgb = colors.astype(np.float32) / 255.0

    # Opacity: edge-aware + floater-aware
    opacity = (4.6 * edge_scale_factor + 2.2 * (1.0 - edge_scale_factor)).astype(np.float32)
    opacity *= floater_scale

    # Camera-facing rotation (better for single-image novel views)
    view_dir = np.stack([x_masked, y_masked, z_masked], axis=-1)
    view_norm = np.sqrt(np.sum(view_dir**2, axis=-1, keepdims=True))
    view_dir = view_dir / np.maximum(view_norm, 1e-8)

    dot = view_dir[:, 2]
    cross_x = -view_dir[:, 1]
    cross_y = view_dir[:, 0]
    qw = 1.0 + dot
    anti = qw < 1e-6
    qw[anti] = 0.0
    cross_x[anti] = 1.0
    cross_y[anti] = 0.0
    q_norm = np.maximum(np.sqrt(qw**2 + cross_x**2 + cross_y**2), 1e-8)

    n = len(x_masked)
    result = {
        "x": x_masked.astype(np.float32), "y": y_masked.astype(np.float32), "z": z_masked.astype(np.float32),
        "nx": np.zeros(n, dtype=np.float32),
        "ny": np.zeros(n, dtype=np.float32),
        "nz": np.zeros(n, dtype=np.float32),
        "f_dc_0": ((rgb[:, 0] - 0.5) / SH_C0).astype(np.float32),
        "f_dc_1": ((rgb[:, 1] - 0.5) / SH_C0).astype(np.float32),
        "f_dc_2": ((rgb[:, 2] - 0.5) / SH_C0).astype(np.float32),
        "opacity": opacity,
        "scale_0": log_scale_0.astype(np.float32),
        "scale_1": log_scale_1.astype(np.float32),
        "scale_2": log_scale_2.astype(np.float32),
        "rot_0": (qw / q_norm).astype(np.float32),
        "rot_1": (cross_x / q_norm).astype(np.float32),
        "rot_2": (cross_y / q_norm).astype(np.float32),
        "rot_3": np.zeros(n, dtype=np.float32),
    }

    if return_keep_mask:
        return result, keep_mask
    return result


# --- SAM2 Segmentation (Phase 2) --------------------------------------------

def run_segmentation(image: Image.Image) -> np.ndarray:
    """
    Run SAM2 automatic mask generator on the input image.
    Returns (H, W) uint16 array where each pixel = segment_id (0 = unsegmented).
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    # Determine device from image tensor or default to cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2.1 Tiny model on {device}...")
    sam2 = build_sam2_hf("facebook/sam2.1-hiera-tiny", device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    image_rgb = image.convert("RGB")
    orig_w, orig_h = image_rgb.size

    # Cap SAM2 input to 512px max side for speed (rescale masks back after)
    sam2_max = 512
    if max(orig_w, orig_h) > sam2_max:
        ratio = sam2_max / max(orig_w, orig_h)
        sam2_w, sam2_h = int(orig_w * ratio), int(orig_h * ratio)
        sam2_image = image_rgb.resize((sam2_w, sam2_h), Image.LANCZOS)
        print(f"  SAM2 input resized: {orig_w}x{orig_h} -> {sam2_w}x{sam2_h}")
    else:
        sam2_image = image_rgb
        sam2_w, sam2_h = orig_w, orig_h

    image_np = np.array(sam2_image)
    print("Running SAM2 segmentation...")
    masks = mask_generator.generate(image_np)

    segment_map_small = np.zeros((sam2_h, sam2_w), dtype=np.uint16)

    masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for i, mask_data in enumerate(masks):
        segment_id = i + 1
        segment_map_small[mask_data["segmentation"]] = segment_id

    # Upscale segment map back to original resolution if we downscaled
    if sam2_w != orig_w or sam2_h != orig_h:
        seg_pil = Image.fromarray(segment_map_small)
        seg_pil = seg_pil.resize((orig_w, orig_h), Image.NEAREST)
        segment_map = np.array(seg_pil, dtype=np.uint16)
    else:
        segment_map = segment_map_small

    print(f"  SAM2: {len(masks)} segments detected")
    return segment_map


def write_segments(segment_ids: np.ndarray, gaussians: dict, output_path: str):
    """
    Write segment data alongside the PLY file.
    - .segments.bin: Uint16Array, one per Gaussian
    - .segments.json: metadata about each segment
    """
    base = Path(output_path).with_suffix("")

    bin_path = str(base) + ".segments.bin"
    segment_ids.astype(np.uint16).tofile(bin_path)
    print(f"  Wrote segments binary: {len(segment_ids):,} entries -> {bin_path}")

    unique_ids = np.unique(segment_ids)
    unique_ids = unique_ids[unique_ids > 0]

    segments_meta = []
    for seg_id in unique_ids:
        mask = segment_ids == seg_id
        count = int(np.sum(mask))

        cx = float(np.mean(gaussians["x"][mask]))
        cy = float(np.mean(gaussians["y"][mask]))
        cz = float(np.mean(gaussians["z"][mask]))

        r = float(np.mean(gaussians["f_dc_0"][mask]) * SH_C0 + 0.5) * 255
        g = float(np.mean(gaussians["f_dc_1"][mask]) * SH_C0 + 0.5) * 255
        b = float(np.mean(gaussians["f_dc_2"][mask]) * SH_C0 + 0.5) * 255
        avg_color = [int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))]

        segments_meta.append({
            "id": int(seg_id),
            "name": f"Segment {seg_id}",
            "gaussian_count": count,
            "centroid_3d": [cx, cy, cz],
            "avg_color": avg_color,
        })

    meta = {
        "version": 1,
        "total_gaussians": len(segment_ids),
        "segments": segments_meta,
    }

    json_path = str(base) + ".segments.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote segments metadata: {len(segments_meta)} segments -> {json_path}")


# --- 360 Equirectangular Pipeline --------------------------------------------

def equirect_to_cube_faces(image_np, face_size=768, fov_deg=95.0):
    """Extract 6 perspective cube faces from an equirectangular panorama."""
    h_eq, w_eq = image_np.shape[:2]
    half_fov = math.radians(fov_deg / 2.0)
    f_px = face_size / (2.0 * math.tan(half_fov))

    face_defs = [
        ("front",  np.array([0, 0, 1.]),  np.array([0, -1, 0.])),
        ("right",  np.array([1, 0, 0.]),  np.array([0, -1, 0.])),
        ("back",   np.array([0, 0, -1.]), np.array([0, -1, 0.])),
        ("left",   np.array([-1, 0, 0.]), np.array([0, -1, 0.])),
        ("up",     np.array([0, 1, 0.]),  np.array([0, 0, 1.])),
        ("down",   np.array([0, -1, 0.]), np.array([0, 0, -1.])),
    ]

    faces = []
    for name, fwd, up in face_defs:
        fwd = fwd / np.linalg.norm(fwd)
        up = up / np.linalg.norm(up)
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.column_stack([right, up, fwd])

        c = face_size / 2.0
        uc = np.arange(face_size, dtype=np.float64) - c
        vc = np.arange(face_size, dtype=np.float64) - c
        uu, vv = np.meshgrid(uc, vc)

        rays = np.stack([uu / f_px, -vv / f_px, np.ones_like(uu)], axis=-1) @ R.T
        xw, yw, zw = rays[..., 0], rays[..., 1], rays[..., 2]
        norm = np.sqrt(xw**2 + yw**2 + zw**2)
        lon = np.arctan2(xw, zw)
        lat = np.arcsin(np.clip(yw / norm, -1, 1))

        px = (lon / math.pi + 1.0) * 0.5 * (w_eq - 1)
        py = (0.5 - lat / math.pi) * (h_eq - 1)
        px = np.clip(px, 0, w_eq - 1)
        py = np.clip(py, 0, h_eq - 1)

        x0 = np.floor(px).astype(np.int32)
        y0 = np.floor(py).astype(np.int32)
        x1 = np.minimum(x0 + 1, w_eq - 1)
        y1 = np.minimum(y0 + 1, h_eq - 1)
        wx = (px - x0)[..., np.newaxis]
        wy = (py - y0)[..., np.newaxis]

        face_img = (
            image_np[y0, x0] * (1 - wx) * (1 - wy) +
            image_np[y0, x1] * wx * (1 - wy) +
            image_np[y1, x0] * (1 - wx) * wy +
            image_np[y1, x1] * wx * wy
        ).astype(np.uint8)

        faces.append((name, face_img, R, f_px))
        print(f"  Extracted {name} face: {face_img.shape}")

    return faces


def transform_gaussians_to_world(gaussians, R_face):
    """Transform Gaussian positions from face-local space to world space."""
    pos = np.stack([gaussians["x"], gaussians["y"], gaussians["z"]], axis=-1)
    pos_world = pos @ R_face.T
    gaussians["x"] = pos_world[:, 0].astype(np.float32)
    gaussians["y"] = pos_world[:, 1].astype(np.float32)
    gaussians["z"] = pos_world[:, 2].astype(np.float32)
    return gaussians


def merge_gaussian_dicts(all_gaussians):
    """Concatenate multiple Gaussian dicts into one."""
    merged = {}
    for key in all_gaussians[0].keys():
        merged[key] = np.concatenate([g[key] for g in all_gaussians])
    print(f"  Merged: {len(merged['x']):,} total splats from {len(all_gaussians)} faces")
    return merged


# --- Gaussian Normalization ---------------------------------------------------

def normalize_gaussians(gaussians, target_extent=50.0):
    """
    Normalize gaussian positions so the scene fits within [-target_extent, target_extent].
    Scales gaussian sizes (log-space) accordingly. This ensures consistent scale
    across all depth engines (DA2, DA3, DA360) for viewer compatibility.
    """
    positions = np.stack([gaussians["x"], gaussians["y"], gaussians["z"]], axis=-1)

    # Center on median (robust to outliers)
    center = np.median(positions, axis=0)
    positions -= center

    # Find 98th percentile extent (ignore outliers)
    max_extent = np.percentile(np.abs(positions), 98)
    if max_extent < 1e-6:
        max_extent = 1.0

    scale_factor = target_extent / max_extent
    positions *= scale_factor

    gaussians["x"] = positions[:, 0].astype(np.float32)
    gaussians["y"] = positions[:, 1].astype(np.float32)
    gaussians["z"] = positions[:, 2].astype(np.float32)

    # Scale gaussian sizes (stored in log-space)
    log_sf = np.float32(np.log(scale_factor))
    gaussians["scale_0"] = gaussians["scale_0"] + log_sf
    gaussians["scale_1"] = gaussians["scale_1"] + log_sf
    gaussians["scale_2"] = gaussians["scale_2"] + log_sf

    print(f"  Normalized: scale_factor={scale_factor:.2f}x, center=[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    return gaussians


# --- PLY Writer ---------------------------------------------------------------

def write_ply(gaussians, output_path):
    """Write Gaussian parameters to a standard 3DGS binary PLY file."""
    props = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    props = [p for p in props if p in gaussians]
    n = len(gaussians["x"])

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    for p in props:
        header += f"property float {p}\n"
    header += "end_header\n"

    data = np.empty((n, len(props)), dtype=np.float32)
    for i, p in enumerate(props):
        data[:, i] = gaussians[p]

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Wrote PLY: {n:,} splats, {size_mb:.1f}MB -> {output_path}")


# --- High-Level Pipeline Functions -------------------------------------------
# Used by both the CLI and the Modal service.

def generate_single(image, processor, model, device="cpu",
                    resolution=2048, overlap=1.3, disc_threshold=0.1,
                    depth_mode="relative", fov_override=None,
                    skip_sky=False, fast_mode=False, segment=False,
                    inpaint=True):
    """Single image pipeline: returns Gaussian dict (and optionally segment_ids)."""
    w_orig, h_orig = image.size
    image_rgb = image.convert("RGB")

    # Resize for depth estimation
    aspect = w_orig / h_orig
    if aspect >= 1:
        new_w, new_h = resolution, int(resolution / aspect)
    else:
        new_w, new_h = int(resolution * aspect), resolution
    image_resized = image_rgb.resize((new_w, new_h), Image.LANCZOS)
    print(f"  Depth at: {new_w}x{new_h}")

    # High-res color image: use original or cap at 2048px
    color_cap = 2048
    if max(w_orig, h_orig) > color_cap:
        if aspect >= 1:
            cw, ch = color_cap, int(color_cap / aspect)
        else:
            cw, ch = int(color_cap * aspect), color_cap
        color_image = np.array(image_rgb.resize((cw, ch), Image.LANCZOS))
    else:
        color_image = np.array(image_rgb)
    print(f"  Color at: {color_image.shape[1]}x{color_image.shape[0]}")

    # Focal length: CLI override > EXIF > default heuristic
    if fov_override is not None:
        focal_length = max(new_w, new_h) / (2.0 * math.tan(math.radians(fov_override / 2.0)))
        print(f"  Focal length (from --fov {fov_override}deg): {focal_length:.0f}px")
    else:
        exif_focal = extract_focal_from_exif(image, new_w, new_h)
        if exif_focal is not None:
            focal_length = exif_focal
        else:
            focal_length = max(new_w, new_h) * 0.7
            print(f"  Focal length (default): {focal_length:.0f}px")

    is_metric = depth_mode != "relative"
    depth = estimate_depth(image_resized, processor, model, device)

    # Segmentation (run on image before depth conversion)
    segment_map = None
    if segment:
        segment_map = run_segmentation(image_resized)

    if segment and segment_map is not None:
        # Get gaussians AND the exact keep_mask used internally
        gaussians, keep_mask = depth_to_gaussians(
            np.array(image_resized), depth, focal_length,
            overlap_factor=overlap,
            depth_disc_threshold=disc_threshold,
            color_image_np=color_image,
            is_metric=is_metric,
            skip_sky=skip_sky,
            fast_mode=fast_mode,
            return_keep_mask=True,
            inpaint=inpaint,
        )
        segment_ids = segment_map[keep_mask]
        return gaussians, segment_ids

    gaussians = depth_to_gaussians(
        np.array(image_resized), depth, focal_length,
        overlap_factor=overlap,
        depth_disc_threshold=disc_threshold,
        color_image_np=color_image,
        is_metric=is_metric,
        skip_sky=skip_sky,
        fast_mode=fast_mode,
        inpaint=inpaint,
    )

    return gaussians


def generate_360(image, processor, model, device="cpu",
                 face_size=1024, overlap=1.3, disc_threshold=0.1,
                 depth_mode="relative", skip_sky=False, fast_mode=False,
                 inpaint=True):
    """360 equirectangular pipeline: returns Gaussian dict."""
    image_rgb = image.convert("RGB")
    pano_np = np.array(image_rgb)
    w, h = image_rgb.size
    print(f"  360 input: {w}x{h}, aspect={w / h:.2f}")

    is_metric = depth_mode != "relative"
    faces = equirect_to_cube_faces(pano_np, face_size=face_size, fov_deg=95.0)

    all_gaussians = []
    for name, face_img, R_face, f_px in faces:
        print(f"  Processing {name}...")
        face_pil = Image.fromarray(face_img)
        depth = estimate_depth(face_pil, processor, model, device)

        gaussians = depth_to_gaussians(
            face_img, depth, f_px,
            overlap_factor=overlap,
            depth_disc_threshold=disc_threshold,
            is_metric=is_metric,
            skip_sky=skip_sky,
            fast_mode=fast_mode,
            inpaint=inpaint,
        )
        gaussians = transform_gaussians_to_world(gaussians, R_face)

        if name in ("up", "down"):
            gaussians["y"] = -gaussians["y"]

        all_gaussians.append(gaussians)
        print(f"    {len(gaussians['x']):,} splats")

    return merge_gaussian_dicts(all_gaussians)


# --- DA360 Equirectangular Pipeline ------------------------------------------

def equirect_depth_to_gaussians(
    image_np: np.ndarray,
    depth: np.ndarray,
    overlap_factor: float = 1.3,
    depth_disc_threshold: float = 0.1,
    flat_ratio: float = 0.1,
    skip_sky: bool = False,
    fast_mode: bool = False,
    input_is_depth: bool = False,
    inpaint: bool = True,
) -> dict:
    """Convert equirectangular RGB + depth into 3D Gaussians via spherical unprojection."""
    from scipy.ndimage import median_filter

    h, w = depth.shape
    depth = median_filter(depth, size=3).astype(np.float32)

    # Inpaint occlusion shadows before unprojection
    if inpaint:
        image_np, depth = inpaint_occlusions(
            image_np, depth, disc_threshold=depth_disc_threshold,
        )

    near_plane, far_plane = 1.0, 100.0

    if input_is_depth:
        # Input is already proper depth (e.g. from DA360) — just rescale to [near, far]
        d_min, d_max = np.percentile(depth, [2, 98])
        if d_max - d_min < 1e-6:
            d_max = d_min + 1.0
        depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
        metric_depth = near_plane + depth_norm * (far_plane - near_plane)
    else:
        # Input is disparity-like (DA2) — invert to get depth
        d_min, d_max = np.percentile(depth, [2, 98])
        if d_max - d_min < 1e-6:
            d_max = d_min + 1.0
        depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
        disparity = np.maximum(depth_norm, 0.01)
        inv_range = 1.0 / 0.01 - 1.0
        metric_depth = near_plane + (1.0 / disparity - 1.0) / inv_range * (far_plane - near_plane)

    metric_depth = np.clip(metric_depth, near_plane, far_plane).astype(np.float32)

    sky_mask = detect_sky_mask(image_np, metric_depth, far_plane) if skip_sky else np.zeros((h, w), dtype=bool)

    # Spherical coordinate grids
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    lon = (uu / w - 0.5) * 2.0 * np.pi   # [-π, π]
    lat = (0.5 - vv / h) * np.pi          # [π/2, -π/2]

    # Spherical to Cartesian
    x = metric_depth * np.cos(lat) * np.sin(lon)
    y = -metric_depth * np.sin(lat)
    z = metric_depth * np.cos(lat) * np.cos(lon)

    # Depth discontinuity filter
    grad_x = np.abs(np.diff(metric_depth, axis=1, prepend=metric_depth[:, :1]))
    grad_y = np.abs(np.diff(metric_depth, axis=0, prepend=metric_depth[:1, :]))
    max_grad = np.maximum(grad_x, grad_y)
    keep_mask = max_grad < (depth_disc_threshold * metric_depth)

    if not input_is_depth:
        z_cull = np.percentile(metric_depth[keep_mask], 2)
        keep_mask &= metric_depth > z_cull

    if np.any(sky_mask) and skip_sky:
        keep_mask &= ~sky_mask

    floater_opacity_scale = prune_floaters(metric_depth, keep_mask)

    if fast_mode:
        keep_mask &= compute_importance_mask(metric_depth, image_np)

    n_total = h * w
    n_killed = n_total - int(np.sum(keep_mask))
    print(f"  Depth filter: {n_killed:,} splats removed ({n_killed / n_total * 100:.1f}%)")

    edge_proximity = max_grad[keep_mask] / np.maximum(metric_depth[keep_mask], 1e-6)
    edge_scale_factor = np.clip(1.0 - edge_proximity * 5.0, 0.3, 1.0).astype(np.float32)

    x_m, y_m, z_m = x[keep_mask], y[keep_mask], z[keep_mask]
    d = metric_depth[keep_mask]
    floater_scale = floater_opacity_scale[keep_mask]
    colors = image_np[keep_mask]

    # Scale based on solid angle per pixel at this latitude
    lat_m = lat[keep_mask]
    dtheta = 2.0 * np.pi / w
    dphi = np.pi / h
    scale_h = d * np.abs(np.cos(lat_m)) * dtheta * overlap_factor * edge_scale_factor
    scale_v = d * dphi * overlap_factor * edge_scale_factor
    scale_tangent = np.maximum(scale_h, scale_v)
    scale_normal = scale_tangent * flat_ratio

    rgb = colors.astype(np.float32) / 255.0
    opacity = (4.6 * edge_scale_factor + 2.2 * (1.0 - edge_scale_factor)).astype(np.float32)
    opacity *= floater_scale

    # Camera-facing rotation (face toward origin)
    view_dir = np.stack([x_m, y_m, z_m], axis=-1)
    view_norm = np.sqrt(np.sum(view_dir**2, axis=-1, keepdims=True))
    view_dir = view_dir / np.maximum(view_norm, 1e-8)
    dot = view_dir[:, 2]
    cross_x = -view_dir[:, 1]
    cross_y = view_dir[:, 0]
    qw = 1.0 + dot
    anti = qw < 1e-6
    qw[anti] = 0.0
    cross_x[anti] = 1.0
    cross_y[anti] = 0.0
    q_norm = np.maximum(np.sqrt(qw**2 + cross_x**2 + cross_y**2), 1e-8)

    n = len(x_m)
    return {
        "x": x_m.astype(np.float32), "y": y_m.astype(np.float32), "z": z_m.astype(np.float32),
        "nx": np.zeros(n, np.float32), "ny": np.zeros(n, np.float32), "nz": np.zeros(n, np.float32),
        "f_dc_0": ((rgb[:, 0] - 0.5) / SH_C0).astype(np.float32),
        "f_dc_1": ((rgb[:, 1] - 0.5) / SH_C0).astype(np.float32),
        "f_dc_2": ((rgb[:, 2] - 0.5) / SH_C0).astype(np.float32),
        "opacity": opacity,
        "scale_0": np.log(np.maximum(scale_tangent, 1e-7)).astype(np.float32),
        "scale_1": np.log(np.maximum(scale_tangent, 1e-7)).astype(np.float32),
        "scale_2": np.log(np.maximum(scale_normal, 1e-7)).astype(np.float32),
        "rot_0": (qw / q_norm).astype(np.float32),
        "rot_1": (cross_x / q_norm).astype(np.float32),
        "rot_2": (cross_y / q_norm).astype(np.float32),
        "rot_3": np.zeros(n, np.float32),
    }


def generate_360_da360(image, da360_model, da360_h, da360_w, device="cpu",
                       overlap=1.3, disc_threshold=0.1, skip_sky=False, fast_mode=False,
                       inpaint=True, resolution=6144):
    """360 pipeline using DA360: single-pass panoramic depth, no cube faces.

    resolution controls the unprojection grid width (height = width/2).
    The depth model runs at its native resolution regardless; this controls
    how densely splats are sampled from the depth map.
    """
    import cv2
    from da360 import estimate_depth_da360

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    w_orig, h_orig = image_rgb.size
    print(f"  DA360 input: {w_orig}x{h_orig}")

    depth = estimate_depth_da360(image, da360_model, da360_h, da360_w, device)
    print(f"  DA360 depth: range=[{depth.min():.2f}, {depth.max():.2f}]")

    # Resample image and depth to unprojection resolution
    # This controls splat density without affecting depth quality
    unproj_w = min(resolution, w_orig)
    unproj_h = unproj_w // 2
    if unproj_w < w_orig:
        print(f"  Unprojection grid: {unproj_w}x{unproj_h} ({unproj_w * unproj_h:,} pixels)")
        image_np = cv2.resize(image_np, (unproj_w, unproj_h), interpolation=cv2.INTER_LANCZOS4)
        depth = cv2.resize(depth, (unproj_w, unproj_h), interpolation=cv2.INTER_LINEAR)
    else:
        print(f"  Unprojection grid: {w_orig}x{h_orig} (full resolution)")

    return equirect_depth_to_gaussians(
        image_np, depth,
        overlap_factor=overlap,
        depth_disc_threshold=disc_threshold,
        skip_sky=skip_sky,
        fast_mode=fast_mode,
        input_is_depth=True,
        inpaint=inpaint,
    )


# --- DA3 Multi-Image Pipeline ------------------------------------------------

def generate_multi(image_paths, da3_model, overlap=1.3, disc_threshold=0.1,
                   skip_sky=False, fast_mode=False, resolution=2048,
                   inpaint=True):
    """Multi-image pipeline using DA3: consistent depth + auto poses → merged Gaussians."""
    print(f"  Running DA3 on {len(image_paths)} images...")
    prediction = da3_model.inference(image_paths, process_res=resolution)

    all_gaussians = []
    n_images = prediction.depth.shape[0]

    for i in range(n_images):
        depth_map = prediction.depth[i]
        image_np = prediction.processed_images[i]
        ext = prediction.extrinsics[i]   # (3, 4) w2c
        ixt = prediction.intrinsics[i]   # (3, 3)

        h, w = depth_map.shape
        fx, fy = ixt[0, 0], ixt[1, 1]
        focal_length = (fx + fy) / 2.0

        print(f"  Image {i+1}/{n_images}: {w}x{h}, focal={focal_length:.0f}px")

        gaussians = depth_to_gaussians(
            image_np, depth_map, focal_length,
            overlap_factor=overlap,
            depth_disc_threshold=disc_threshold,
            is_metric=True,
            skip_sky=skip_sky,
            fast_mode=fast_mode,
            inpaint=inpaint,
        )

        # Transform camera-space → world-space using predicted extrinsics
        R_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c

        pos = np.stack([gaussians["x"], gaussians["y"], gaussians["z"]], axis=-1)
        pos_world = pos @ R_c2w.T + t_c2w
        gaussians["x"] = pos_world[:, 0].astype(np.float32)
        gaussians["y"] = pos_world[:, 1].astype(np.float32)
        gaussians["z"] = pos_world[:, 2].astype(np.float32)

        all_gaussians.append(gaussians)
        print(f"    {len(gaussians['x']):,} splats")

    return merge_gaussian_dicts(all_gaussians)


def generate_single_da3(image_path, da3_model, overlap=1.3, disc_threshold=0.1,
                        skip_sky=False, fast_mode=False, resolution=2048,
                        inpaint=True):
    """Single image pipeline using DA3 instead of DA2."""
    depth, image_np, ixt = da3_single_depth(image_path, da3_model, process_res=resolution)
    h, w = depth.shape
    fx, fy = ixt[0, 0], ixt[1, 1]
    focal_length = (fx + fy) / 2.0
    print(f"  DA3 depth: {w}x{h}, focal={focal_length:.0f}px")

    return depth_to_gaussians(
        image_np, depth, focal_length,
        overlap_factor=overlap,
        depth_disc_threshold=disc_threshold,
        is_metric=True,
        skip_sky=skip_sky,
        fast_mode=fast_mode,
        inpaint=inpaint,
    )


# --- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BLUNT -- Basic Lifting and UNprojection Tool")
    parser.add_argument("input", nargs="+", help="Input image path(s). Multiple paths for --mode multi.")
    parser.add_argument("-o", "--output", help="Output PLY path (default: input_splat.ply)")
    parser.add_argument("--mode", choices=["single", "360", "multi"], default=None,
                        help="Pipeline mode (auto-detected if not set)")
    parser.add_argument("--engine", choices=["da2", "da3", "da360"], default=None,
                        help="Depth engine (auto-selected based on mode if not set)")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overlap", type=float, default=1.3)
    parser.add_argument("--disc-threshold", type=float, default=0.1)
    parser.add_argument("--depth-mode", choices=["relative", "metric-indoor", "metric-outdoor"],
                        default="relative", help="Depth model variant (DA2 only)")
    parser.add_argument("--fov", type=float, default=None,
                        help="Manual FOV override in degrees")
    parser.add_argument("--no-sky", action="store_true",
                        help="Skip sky Gaussians entirely")
    parser.add_argument("--no-inpaint", action="store_true",
                        help="Disable occlusion inpainting (shadow filling)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip position/scale normalization (raw depth values)")
    parser.add_argument("--fast", action="store_true",
                        help="Adaptive stride mode (fewer splats, faster)")
    parser.add_argument("--segment", action="store_true",
                        help="Run SAM2 segmentation (generates .segments.bin + .segments.json)")
    # DA3 options
    parser.add_argument("--da3-model", type=str, default="DA3-BASE",
                        help="DA3 model name (default: DA3-BASE, Apache 2.0)")
    # DA360 options
    parser.add_argument("--da360-checkpoint", type=str, default=None,
                        help="Path to DA360 checkpoint (e.g., DA360_large.pth)")
    args = parser.parse_args()

    # Auto-detect mode
    if args.mode is None:
        args.mode = "multi" if len(args.input) > 1 else "single"

    # Auto-find DA360 checkpoint if not specified
    if args.mode == "360" and args.da360_checkpoint is None:
        # Look for checkpoint in common locations relative to script
        script_dir = Path(__file__).parent
        for candidate in [
            script_dir / "da360" / "DA360_large.pth",
            script_dir / "DA360_large.pth",
        ]:
            if candidate.exists():
                args.da360_checkpoint = str(candidate)
                print(f"Auto-found DA360 checkpoint: {args.da360_checkpoint}")
                break

    # Auto-select engine: DA3 default for single/multi, DA360 for 360 if checkpoint available
    if args.engine is None:
        if args.mode == "multi":
            args.engine = "da3"
        elif args.mode == "360" and args.da360_checkpoint:
            args.engine = "da360"
        elif args.mode == "360":
            args.engine = "da2"
        else:
            args.engine = "da3"

    # Validate
    if args.mode == "multi" and args.engine != "da3":
        parser.error("--mode multi requires --engine da3 (or omit --engine for auto)")
    if args.engine == "da360" and not args.da360_checkpoint:
        # Will auto-download from HuggingFace
        print("No DA360 checkpoint specified — will auto-download from HuggingFace on first use.")

    if args.output is None:
        stem = Path(args.input[0]).stem
        suffix = "_multi" if args.mode == "multi" else ""
        args.output = f"{stem}{suffix}_splat.ply"

    start = time.time()

    # --- Multi-image mode (DA3) ---
    if args.mode == "multi":
        print(f"Multi-image mode: {len(args.input)} images")
        da3_model = load_da3_model(args.da3_model, args.device)
        print("Generating splat...")
        gaussians = generate_multi(
            args.input, da3_model,
            overlap=args.overlap, disc_threshold=args.disc_threshold,
            skip_sky=args.no_sky, fast_mode=args.fast,
            resolution=args.resolution, inpaint=not args.no_inpaint,
        )
        if not args.no_normalize:
            gaussians = normalize_gaussians(gaussians)
        write_ply(gaussians, args.output)
        print(f"Total: {time.time() - start:.1f}s")
        return

    # --- Single input modes ---
    image = Image.open(args.input[0])
    print(f"Input: {image.size[0]}x{image.size[1]}, engine={args.engine}")

    segment_ids = None

    if args.mode == "360" and args.engine == "da360":
        # DA360: single-pass panoramic depth
        da360_model, da360_h, da360_w = load_da360_model_wrapper(args.da360_checkpoint, args.device)
        print("Generating splat (DA360)...")
        # Use 6144 default for DA360 unless user explicitly set --resolution
        da360_res = args.resolution if args.resolution != 2048 else 6144
        gaussians = generate_360_da360(
            image, da360_model, da360_h, da360_w, args.device,
            overlap=args.overlap, disc_threshold=args.disc_threshold,
            skip_sky=args.no_sky, fast_mode=args.fast,
            inpaint=not args.no_inpaint,
            resolution=da360_res,
        )

    elif args.mode == "360":
        # DA2: cube face fallback
        print("Loading depth model...")
        processor, model = load_depth_model(args.device, args.depth_mode)
        print("Generating splat (cube faces)...")
        gaussians = generate_360(
            image, processor, model, args.device,
            face_size=args.resolution, overlap=args.overlap,
            disc_threshold=args.disc_threshold, depth_mode=args.depth_mode,
            skip_sky=args.no_sky, fast_mode=args.fast,
            inpaint=not args.no_inpaint,
        )

    elif args.engine == "da3":
        # DA3: single image
        da3_model = load_da3_model(args.da3_model, args.device)
        print("Generating splat (DA3)...")
        gaussians = generate_single_da3(
            args.input[0], da3_model,
            overlap=args.overlap, disc_threshold=args.disc_threshold,
            skip_sky=args.no_sky, fast_mode=args.fast,
            resolution=args.resolution, inpaint=not args.no_inpaint,
        )

    else:
        # DA2: single image (default)
        print("Loading depth model...")
        processor, model = load_depth_model(args.device, args.depth_mode)
        print("Generating splat...")
        result = generate_single(
            image, processor, model, args.device,
            resolution=args.resolution, overlap=args.overlap,
            disc_threshold=args.disc_threshold, depth_mode=args.depth_mode,
            fov_override=args.fov, skip_sky=args.no_sky,
            fast_mode=args.fast, segment=args.segment,
            inpaint=not args.no_inpaint,
        )
        if args.segment:
            gaussians, segment_ids = result
        else:
            gaussians = result

    if not args.no_normalize:
        gaussians = normalize_gaussians(gaussians)
    write_ply(gaussians, args.output)

    if args.segment and segment_ids is not None:
        write_segments(segment_ids, gaussians, args.output)

    print(f"Total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
