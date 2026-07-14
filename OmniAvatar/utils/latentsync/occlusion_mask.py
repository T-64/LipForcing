"""Occlusion-aware paste-back mask via BiSeNet face parsing.

Runs BiSeNet on the *aligned* (512x512, tight-cropped) face — the distribution
it was trained on — to identify which pixels genuinely belong to the face
(skin/nose/lips/brows/eyes). Pixels labeled as background, cloth, hair, hat, or
glasses (i.e. objects occluding the face like a billboard or a hand) are
excluded from the paste-back mask, so the generated lip-sync face does not
smear over occluding objects.

The caller is responsible for inverse-warping the aligned-space mask into the
original frame with the same affine used for the generated face.

BiSeNet class ids (CelebAMask-HQ):
  0 background, 1 skin, 2/3 brows, 4/5 eyes, 6 eye_glass,
  7/8 ears, 9 ear_ring, 10 nose, 11 mouth, 12/13 lips,
  14 neck, 15 necklace, 16 cloth, 17 hair, 18 hat
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from ..face_parsing.model import BiSeNet


# Classes that are genuinely "face surface we want to overwrite". Anything else
# is treated as occlusion and preserved from the original frame.
# Include neck (14) because the aligned face crop can include a bit of neck at
# the bottom and we do not want the composite to leave a hard neck boundary.
FACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13, 14)


def _default_weight_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "face_parsing", "79999_iter.pth"))


class OcclusionMasker:
    """Runs BiSeNet on aligned face crops and returns per-frame skin masks.

    The returned mask is 1.0 where the pixel is face-surface (safe to overwrite
    with generated content) and 0.0 where it is occluding content (hand,
    billboard, hair, glasses, cloth, background -- keep the original).
    """

    def __init__(
        self,
        weight_path: Optional[str] = None,
        device: str = "cuda",
        input_size: int = 512,
        batch_size: int = 8,
        ema_alpha: float = 0.7,
        feather_ksize: int = 15,
    ):
        self.device = device
        self.input_size = input_size
        self.batch_size = batch_size
        self.ema_alpha = ema_alpha
        self.feather_ksize = feather_ksize

        wp = weight_path or _default_weight_path()
        if not os.path.isfile(wp):
            raise FileNotFoundError(
                f"BiSeNet weights not found: {wp}. Copy 79999_iter.pth into "
                f"OmniAvatar/utils/face_parsing/."
            )

        net = BiSeNet(n_classes=19)
        state = torch.load(wp, map_location="cpu")
        net.load_state_dict(state)
        net.to(device).eval()
        self.net = net

        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

        face_lut = torch.zeros(19, dtype=torch.float32)
        for c in FACE_CLASSES:
            face_lut[c] = 1.0
        self.face_lut = face_lut.to(device)

    @torch.no_grad()
    def compute_aligned(self, aligned_faces_rgb: np.ndarray) -> np.ndarray:
        """Compute skin masks in aligned (crop) coordinates.

        Args:
            aligned_faces_rgb: uint8 array [N, H, W, 3] (RGB), typically 512x512
                tight face crops.

        Returns:
            float32 array [N, H, W] with values in [0, 1] in the same aligned
            coordinate frame.
        """
        assert aligned_faces_rgb.dtype == np.uint8 and aligned_faces_rgb.ndim == 4, (
            f"expected [N,H,W,3] uint8, got {aligned_faces_rgb.shape}/{aligned_faces_rgb.dtype}"
        )
        n, h, w, _ = aligned_faces_rgb.shape
        masks: List[np.ndarray] = []

        prev_mask: Optional[np.ndarray] = None
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = aligned_faces_rgb[start:end]  # [b, H, W, 3]

            t = torch.from_numpy(batch).to(self.device).float() / 255.0
            t = t.permute(0, 3, 1, 2)  # NCHW
            if h != self.input_size or w != self.input_size:
                t = F.interpolate(
                    t, size=(self.input_size, self.input_size),
                    mode="bilinear", align_corners=False,
                )
            t = self.normalize(t)

            logits = self.net(t)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            parsing = logits.argmax(dim=1)  # [b, S, S]

            # class-id -> face/not-face
            face_bin = self.face_lut[parsing]  # [b, S, S] float

            if face_bin.shape[-1] != w or face_bin.shape[-2] != h:
                face_bin = F.interpolate(
                    face_bin.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            face_bin_np = face_bin.detach().cpu().numpy()

            for i in range(face_bin_np.shape[0]):
                m = face_bin_np[i]

                if self.feather_ksize and self.feather_ksize > 1:
                    k = int(self.feather_ksize) | 1  # odd
                    m = cv2.GaussianBlur(m, (k, k), 0)

                if prev_mask is not None and 0.0 < self.ema_alpha < 1.0:
                    m = self.ema_alpha * m + (1.0 - self.ema_alpha) * prev_mask

                m = np.clip(m, 0.0, 1.0).astype(np.float32)
                masks.append(m)
                prev_mask = m

        return np.stack(masks, axis=0)

    @staticmethod
    def warp_to_original(
        aligned_mask: np.ndarray,
        affine_matrix,
        original_hw: tuple,
    ) -> np.ndarray:
        """Inverse-warp an aligned-space mask into original-frame coordinates.

        Args:
            aligned_mask: [Ha, Wa] float32 in [0, 1].
            affine_matrix: 2x3 forward affine (original -> aligned) as np.ndarray
                or torch.Tensor [1, 2, 3].
            original_hw: (H, W) of the target original frame.

        Returns:
            [H, W] float32 mask in original-frame coordinates.
        """
        if hasattr(affine_matrix, "detach"):
            M = affine_matrix.detach().cpu().numpy()
            if M.ndim == 3:
                M = M[0]
        else:
            M = np.asarray(affine_matrix)
            if M.ndim == 3:
                M = M[0]
        M = M.astype(np.float32)
        # Invert 2x3 affine.
        A = np.eye(3, dtype=np.float32)
        A[:2, :] = M
        A_inv = np.linalg.inv(A)[:2, :]

        H, W = original_hw
        warped = cv2.warpAffine(
            aligned_mask.astype(np.float32), A_inv, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        return np.clip(warped, 0.0, 1.0)
