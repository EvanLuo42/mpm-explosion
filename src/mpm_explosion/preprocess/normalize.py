from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import trimesh


@dataclass
class NormalizeResult:
    mesh: trimesh.Trimesh
    center: np.ndarray          # (3,)
    scale: float                # scalar
    transform: np.ndarray       # (4,4) applied to original -> normalized


def normalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    mode: Literal["unit_box", "target_size"] = "unit_box",
    target_size: float = 1.0,
    center: bool = True,
) -> NormalizeResult:
    """
    Normalize mesh to help simulation stability.

    - unit_box: scale so AABB max extent becomes 1
    - target_size: scale so AABB max extent becomes target_size

    Returns a new mesh with transform applied.
    """
    m = mesh.copy()
    bounds = m.bounds
    bmin, bmax = bounds
    c = (bmin + bmax) * 0.5
    extent = (bmax - bmin)
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        raise ValueError("Mesh has degenerate bounds")

    if mode == "unit_box":
        s = 1.0 / max_extent
    elif mode == "target_size":
        s = float(target_size) / max_extent
    else:
        raise ValueError(f"Unknown mode: {mode}")

    T = np.eye(4, dtype=np.float32)

    if center:
        # translate to origin
        T_translate = np.eye(4, dtype=np.float32)
        T_translate[:3, 3] = -c.astype(np.float32)
        T = T_translate @ T

    # scale
    T_scale = np.eye(4, dtype=np.float32)
    T_scale[0, 0] = s
    T_scale[1, 1] = s
    T_scale[2, 2] = s
    T = T_scale @ T

    m.apply_transform(T)

    return NormalizeResult(mesh=m, center=c.astype(np.float32), scale=s, transform=T)