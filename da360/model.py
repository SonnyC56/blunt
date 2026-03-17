"""DA360 model wrapper for BLUNT integration."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

from .layers import MultiLayerMLP, modify_conv_layers
from .depth_anything_v2.dpt import DepthAnythingV2


class DA360(nn.Module):
    """Depth Anything in 360° -- panoramic depth with circular padding."""

    def __init__(self, equi_h=518, equi_w=1036, dinov2_encoder="vits", **kwargs):
        super().__init__()

        self.equi_h = equi_h
        self.equi_w = equi_w
        self.dinov2_encoder = dinov2_encoder

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        self.depth_anything = DepthAnythingV2(**model_configs[dinov2_encoder])
        self.depth_anything.eval()
        # Replace standard Conv2d with circular-padded ERP convolutions
        self.depth_anything.depth_head.apply(modify_conv_layers)

        vit_dim = model_configs[dinov2_encoder]["out_channels"][-1]
        self.shift_mlp = MultiLayerMLP(
            input_dim=vit_dim,
            hidden_dims=[vit_dim // 2, vit_dim // 4],
            output_activation="softplus",
        )
        self.eps = 1e-4

    def forward(self, x):
        ssidisp, cls_token = self.depth_anything(x, return_cls_token=True)
        shift = self.shift_mlp(cls_token).unsqueeze(-1).unsqueeze(-1) + self.eps
        return {"pred_disp": ssidisp + shift}


DEFAULT_DA360_HF_REPO = "storysplat/DA360-Large"
DEFAULT_DA360_HF_FILENAME = "DA360_large.pth"


def load_da360_model(checkpoint_path=None, device="cpu"):
    """Load DA360 model from a local checkpoint or auto-download from HuggingFace."""
    import os
    from pathlib import Path

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        # Auto-download from HuggingFace
        from huggingface_hub import hf_hub_download
        repo_id = checkpoint_path if checkpoint_path and "/" in checkpoint_path else DEFAULT_DA360_HF_REPO
        print(f"Downloading DA360 checkpoint from HuggingFace: {repo_id}...")
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=DEFAULT_DA360_HF_FILENAME,
        )
        print(f"Cached at: {checkpoint_path}")

    print(f"Loading DA360 checkpoint: {checkpoint_path}")
    model_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    dinov2_encoder = model_dict.get("dinov2_encoder", "vits")
    height = model_dict.get("height", 518)
    width = model_dict.get("width", 1036)

    model = DA360(equi_h=height, equi_w=width, dinov2_encoder=dinov2_encoder)
    model_state = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in model_dict.items() if k in model_state},
        strict=False,
    )
    model.to(device)
    model.eval()
    print(f"DA360 loaded: {dinov2_encoder} encoder, input {width}x{height}")
    return model, height, width


def estimate_depth_da360(image, model, height, width, device="cpu"):
    """
    Run DA360 depth estimation on an equirectangular image.
    Returns depth map as float32 array at original resolution.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image_np = np.array(image)
    else:
        image_np = image

    orig_h, orig_w = image_np.shape[:2]

    # Resize to model input size
    resized = cv2.resize(image_np, (width, height), interpolation=cv2.INTER_CUBIC)

    # Normalize and run inference
    tensor = transforms.ToTensor()(resized)
    tensor = normalize(tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        pred_disp = outputs["pred_disp"].squeeze().cpu().numpy()

    # Convert disparity to depth (scale-invariant)
    depth = 1.0 / np.maximum(pred_disp, 1e-8)
    depth = depth / depth.min()

    # Resize back to original resolution
    if depth.shape != (orig_h, orig_w):
        depth = cv2.resize(depth.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return depth.astype(np.float32)
