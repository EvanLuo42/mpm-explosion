
import numpy as np
import trimesh

from mpm_explosion.preprocess.sample_mesh import sample_mesh_volume_grid


def test_volume_sampling_on_unit_cube():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    spacing = 0.1
    res = sample_mesh_volume_grid(mesh, spacing=spacing, jitter=0.0)

    assert 700 <= res.positions.shape[0] <= 1300

    approx_vol = float(res.volume.sum())
    assert abs(approx_vol - 1.0) / 1.0 < 0.25 

    inside = mesh.contains(res.positions)
    assert float(inside.mean()) > 0.99

    com = res.positions.mean(axis=0)
    assert np.linalg.norm(com) < 0.05