"""
LatentSync face detection, affine alignment, and compositing utilities.

Provides InsightFace-based face detection, Procrustes affine alignment to 512x512,
and inverse affine compositing with soft blending for video-to-video lip sync.

Adapted from LatentSync.
"""

from .face_detector import FaceDetector
from .affine_transform import AlignRestore
from .image_processor import ImageProcessor, load_fixed_mask

__all__ = [
    "FaceDetector",
    "AlignRestore",
    "ImageProcessor",
    "load_fixed_mask",
]
