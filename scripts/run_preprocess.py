import argparse
from pathlib import Path

import numpy as np
import trimesh

from mpm_explosion.preprocess import load_mesh, normalize_mesh
from mpm_explosion.preprocess.sample_mesh import sample_mesh_volume_grid
from mpm_explosion.preprocess.io import save_npz


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess mesh into MPM particles"
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--target-size", type=float, default=10.0)
    parser.add_argument("--spacing", type=float, default=0.1)
    parser.add_argument("--jitter", type=float, default=0.2)
    parser.add_argument("--density", type=float, default=2000.0)
    parser.add_argument(
        "--export-points",
        action="store_true",
        help="Export particle point cloud as OBJ",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    stem = input_path.stem

    out_dir = args.out_dir.resolve()
    cache_dir = out_dir / "cache"
    debug_dir = out_dir / "debug"

    cache_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / f"preprocess_{stem}.npz"
    particles_path = debug_dir / f"particles_{stem}.obj"

    loaded = load_mesh(input_path, force="scene")
    mesh = loaded.as_single_mesh()

    norm = normalize_mesh(
        mesh,
        mode="target_size",
        target_size=args.target_size,
        center=True,
    )

    sample = sample_mesh_volume_grid(
        norm.mesh,
        spacing=args.spacing,
        jitter=args.jitter,
        density=args.density,
    )

    if args.export_points:
        pc = trimesh.points.PointCloud(sample.positions)
        pc.export(particles_path)

    save_npz(
        npz_path,
        x=sample.positions,
        volume=sample.volume,
        mass=sample.mass if sample.mass is not None else sample.volume * 0,
        center=norm.center,
        scale=np.array([norm.scale], dtype="f4"),
        transform=norm.transform,
    )

    print(f"Done: {sample.positions.shape}")
    print(f"Cache saved to: {npz_path}")
    if args.export_points:
        print(f"Particles saved to: {particles_path}")


if __name__ == "__main__":
    main()