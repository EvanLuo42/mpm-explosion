from mpm_explosion.utils import clear_dir
import argparse
from pathlib import Path

import numpy as np
import trimesh

from mpm_explosion.simulation.mls_mpm import MLSMPM3D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="preprocess_*.npz")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/sim"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dump-every", type=int, default=50)
    ap.add_argument("--dx", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--E", type=float, default=5e4)
    ap.add_argument("--nu", type=float, default=0.2)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    data = np.load(args.input)
    x = data["x"].astype(np.float32)
    mass = data["mass"].astype(np.float32)
    volume = data["volume"].astype(np.float32)

    out_dir = args.out_dir
    clear_dir(out_dir, must_contain="sim")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ymin = float(x[:, 1].min())
    ground_y = ymin - 0.05

    sim = MLSMPM3D(
        x0=x,
        mass=mass,
        volume=volume,
        dx=args.dx,
        dt=args.dt,
        E=args.E,
        nu=args.nu,
        ground_y=ground_y,
        use_gpu=not args.no_gpu,
    )

    for s in range(args.steps):
        sim.step()
        if (s % args.dump_every) == 0:
            pos = sim.get_positions()
            print(f"step {s}, y_min={pos[:,1].min():.4f}, y_mean={pos[:,1].mean():.4f}")
            pc = trimesh.points.PointCloud(pos)
            pc.export(out_dir / f"particles_{s:05d}.obj")
            print("dump", s, pos.shape)

    print("done")


if __name__ == "__main__":
    main()