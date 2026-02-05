from __future__ import annotations
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import trimesh


_RESERVED = {"compress"}

@dataclass
class LoadedMesh:
    """Unified result for either a single mesh or a scene with multiple geometries."""
    mesh: Optional[trimesh.Trimesh] = None
    scene: Optional[trimesh.Scene] = None
    path: Optional[str] = None

    def is_scene(self) -> bool:
        return self.scene is not None

    def as_single_mesh(self) -> trimesh.Trimesh:
        """
        Convert scene → concatenated mesh, or return mesh if already.
        Note: geometry transforms are applied.
        """
        if self.mesh is not None:
            return self.mesh
        assert self.scene is not None
        
        meshes = self.scene.dump(concatenate=False)
        if len(meshes) == 0:
            raise ValueError("Scene contains no meshes")
        return trimesh.util.concatenate(meshes)


def load_mesh(
    path: Union[str, Path],
    *,
    force: str = "mesh",
    process: bool = True,
) -> LoadedMesh:
    """
    Load mesh/scene using trimesh.

    Parameters
    ----------
    path : str | Path
    force : "mesh" | "scene"
        - "mesh": try to load as a single mesh (may still return scene depending on format)
        - "scene": load as scene to preserve hierarchy
    process : bool
        trimesh processing (merging vertices, fixing normals, etc.)

    Returns
    -------
    LoadedMesh
    """
    path = str(Path(path))
    obj = trimesh.load(path, force=force, process=process)

    if isinstance(obj, trimesh.Trimesh):
        return LoadedMesh(mesh=obj, scene=None, path=path)
    if isinstance(obj, trimesh.Scene):
        return LoadedMesh(mesh=None, scene=obj, path=path)

    raise TypeError(f"Unsupported trimesh load result type: {type(obj)}")


def save_npz(path: str | Path, **arrays: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fixed: dict[str, NDArray[np.generic]] = {}
    for k, v in arrays.items():
        if k in _RESERVED:
            raise ValueError(f"Key '{k}' is reserved. Rename it (e.g. '{k}_arr').")

        fixed[k] = np.asarray(v)

    np.savez_compressed(str(p), **fixed)