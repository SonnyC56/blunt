"""
DA360 custom layers: circular padding for ERP images and shift MLP.
Vendored from https://github.com/Insta360-Research-Team/DA360 (MIT License)
"""

from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class ERPCircularConv2d(nn.Module):
    """
    Conv2d with ERP-aware circular padding.
    Horizontal: circular (360° continuity).
    Vertical: pole-aware padding (roll + flip).
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, **kwargs):
        super().__init__()
        self._original_padding = _pair(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = self.conv.stride
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups
        self.padding_mode = self.conv.padding_mode

    def forward(self, x):
        if not any(p != 0 for p in self._original_padding):
            return self.conv(x)

        pad_h, pad_w = self._original_padding
        b, c, h, w = x.shape

        y = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)

        if pad_h > 0:
            top_fill = torch.flip(torch.roll(x[:, :, :pad_h, :], w // 2, -1), dims=[-2])
            y[:, :, :pad_h, pad_w : pad_w + w] = top_fill
            bottom_fill = torch.flip(torch.roll(x[:, :, -pad_h:, :], w // 2, -1), dims=[-2])
            y[:, :, -pad_h:, pad_w : pad_w + w] = bottom_fill

        if pad_w > 0:
            y[:, :, :, :pad_w] = y[:, :, :, -2 * pad_w : -pad_w]
            y[:, :, :, -pad_w:] = y[:, :, :, pad_w : 2 * pad_w]

        return self.conv(y)

    @property
    def padding(self):
        if self._original_padding[0] == self._original_padding[1]:
            return self._original_padding[0]
        return self._original_padding


class MultiLayerMLP(nn.Module):
    """Configurable MLP with optional normalization and output activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        init_method: str = "kaiming",
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.depth = len(hidden_dims)

        activation_dict = {
            "relu": nn.ReLU(inplace=True),
            "elu": nn.ELU(alpha=1.0, inplace=True),
            "softplus": nn.Softplus(),
        }
        self.activation_fn = activation_dict.get(activation.lower(), nn.ReLU(inplace=True))
        self.output_activation_fn = (
            activation_dict.get(output_activation.lower(), nn.Identity())
            if output_activation
            else None
        )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if (use_batch_norm or use_layer_norm) else None

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_method == "kaiming":
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                elif init_method == "xavier":
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.depth):
            x = self.layers[i](x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        if self.output_activation_fn:
            x = self.output_activation_fn(x)
        return x


def modify_conv_layers(module):
    """Recursively replace Conv2d layers with ERPCircularConv2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            custom_conv = ERPCircularConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            custom_conv.weight.data = child.weight.data.clone()
            if child.bias is not None:
                custom_conv.bias.data = child.bias.data.clone()
            setattr(module, name, custom_conv)
