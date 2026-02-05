from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh


@dataclass
class SampleResult:
    positions: np.ndarray   # (P,3)
    volume: np.ndarray      # (P,) per-particle volume estimate
    mass: Optional[np.ndarray] = None  # (P,) optional, if density provided


def sample_mesh_volume_grid(
    mesh: trimesh.Trimesh,
    *,
    spacing: float,
    jitter: float = 0.0,
    density: Optional[float] = None,
    require_watertight: bool = True,
) -> SampleResult:
    """
    Sample interior points of a closed mesh on a regular grid.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    spacing : float
        target particle spacing
    jitter : float
        random jitter scale in [0, 1], applied as +/- spacing*jitter*0.5
    density : float | None
        if given, compute mass = density * volume
    require_watertight : bool
        if true, raise if mesh is not watertight

    Returns
    -------
    SampleResult with positions, volume, and optional mass.
    """
    m = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)

    if require_watertight and (not m.is_watertight):
        raise ValueError("Mesh must be watertight for volume sampling (or set require_watertight=False).")

    bmin, bmax = m.bounds
    dims = bmax - bmin
    if np.any(dims <= 0):
        raise ValueError("Degenerate mesh bounds")

    nx, ny, nz = np.ceil(dims / spacing).astype(int)
    grid = np.mgrid[0:nx, 0:ny, 0:nz].reshape(3, -1).T.astype(np.float32)
    pts = bmin.astype(np.float32) + (grid + 0.5) * float(spacing)

    inside = m.contains(pts)
    pts = pts[inside]

    if pts.shape[0] == 0:
        raise ValueError("No points sampled inside mesh. Check spacing or mesh scale.")

    if jitter > 0.0:
        r = (np.random.rand(*pts.shape).astype(np.float32) - 0.5)
        pts = pts + r * (float(spacing) * float(jitter))

    vol = np.full((pts.shape[0],), float(spacing) ** 3, dtype=np.float32)

    mass = None
    if density is not None:
        mass = vol * float(density)

    return SampleResult(positions=pts.astype(np.float32), volume=vol, mass=mass)