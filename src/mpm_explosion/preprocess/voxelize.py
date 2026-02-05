from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import trimesh


@dataclass
class VoxelGrid:
    origin: np.ndarray     # (3,) world-space origin of voxel grid
    voxel_size: float
    dims: Tuple[int, int, int]  # (nx, ny, nz)
    solid: np.ndarray      # (nx, ny, nz) bool


def voxelize_solid_aabb(
    mesh: trimesh.Trimesh,
    *,
    voxel_size: float,
    pad: float = 0.0,
) -> VoxelGrid:
    """
    Create a simple voxel solid that fills the mesh's AABB (plus optional padding).

    Useful as a baseline for collisions / bounds / debug.
    """
    bmin, bmax = mesh.bounds
    bmin = bmin - pad
    bmax = bmax + pad
    dims_f = (bmax - bmin) / float(voxel_size)
    nx, ny, nz = np.ceil(dims_f).astype(int)
    solid = np.ones((nx, ny, nz), dtype=bool)
    return VoxelGrid(
        origin=bmin.astype(np.float32),
        voxel_size=float(voxel_size),
        dims=(int(nx), int(ny), int(nz)),
        solid=solid,
    )