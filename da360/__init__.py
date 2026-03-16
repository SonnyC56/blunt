"""
DA360 -- Depth Anything in 360°
Vendored from https://github.com/Insta360-Research-Team/DA360 (MIT License)

Provides panoramic-native depth estimation with circular padding
and scale-invariant shift prediction.
"""

from .model import DA360, load_da360_model, estimate_depth_da360
