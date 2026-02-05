from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import taichi as ti


@ti.data_oriented
class MLSMPM3D:
    """
    Minimal 3D MLS-MPM (Neo-Hookean elastic).
    - particles: x, v, C, F, mass, volume
    - grid: v, mass
    """

    def __init__(
        self,
        x0: np.ndarray,          # (P,3) float32
        mass: np.ndarray,        # (P,) float32
        volume: np.ndarray,      # (P,) float32
        *,
        dx: float = 0.05,
        dt: float = 2e-4,
        gravity=(0.0, -9.8, 0.0),
        E: float = 5e4,
        nu: float = 0.2,
        grid_padding: int = 3,
        ground_y: Optional[float] = 0.0,
        use_gpu: bool = True,
    ):
        arch = ti.gpu if use_gpu else ti.cpu
        ti.init(arch=arch, default_fp=ti.f32, cpu_max_num_threads=0)

        assert x0.ndim == 2 and x0.shape[1] == 3
        P = x0.shape[0]
        self.P = P

        self.dx = float(dx)
        self.inv_dx = 1.0 / float(dx)
        self.dt = float(dt)
        self.gravity = ti.Vector(list(gravity), dt=ti.f32)
        
        if ground_y is None:
            self.has_ground = False
            self.ground_y = 0.0
        else:
            self.has_ground = True
            self.ground_y = float(ground_y)
        
        self.has_ground = ti.static(self.has_ground)

        # Lame parameters
        # mu = E / (2(1+nu)), lambda = E*nu/((1+nu)(1-2nu))
        self.mu = float(E) / (2.0 * (1.0 + float(nu)))
        self.lam = float(E) * float(nu) / ((1.0 + float(nu)) * (1.0 - 2.0 * float(nu)))

        # Decide grid bounds from particles
        bmin = x0.min(axis=0)
        bmax = x0.max(axis=0)
        # pad by a few cells
        pad = grid_padding * self.dx
        self.domain_min = (bmin - pad).astype(np.float32)
        self.domain_max = (bmax + pad).astype(np.float32)

        dims = (self.domain_max - self.domain_min) / self.dx
        nx, ny, nz = np.ceil(dims).astype(int) + 1
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)

        # Particle fields
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=P)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=P)
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=P)  # APIC affine
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=P)  # deformation
        self.mass = ti.field(dtype=ti.f32, shape=P)
        self.volume = ti.field(dtype=ti.f32, shape=P)

        # Grid fields
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))

        # Upload initial data
        self.x.from_numpy(x0.astype(np.float32))
        self.v.fill(0.0)
        self.mass.from_numpy(mass.astype(np.float32))
        self.volume.from_numpy(volume.astype(np.float32))
        self._reset_F_C()

    @ti.kernel
    def _reset_F_C(self):
        for p in range(self.P):
            self.F[p] = ti.Matrix.identity(ti.f32, 3)
            self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.func
    def world_to_grid(self, X) -> ti.Vector:
        # grid coordinate in cell units
        return (X - ti.Vector(self.domain_min)) * self.inv_dx

    @ti.func
    def grid_to_world(self, g) -> ti.Vector:
        return ti.Vector(self.domain_min) + g * self.dx

    @ti.kernel
    def clear_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def p2g(self):
        for p in range(self.P):
            Xp = self.x[p]
            vp = self.v[p]
            Cp = self.C[p]
            Fp = self.F[p]

            # MLS-MPM: update deformation gradient with C
            self.F[p] = (ti.Matrix.identity(ti.f32, 3) + self.dt * Cp) @ Fp
            Fp = self.F[p]

            # Neo-Hookean (stable baseline)
            J = ti.max(0.1, Fp.determinant())
            # First Piola: P = mu(F - F^{-T}) + lambda ln(J) F^{-T}
            FinvT = Fp.inverse().transpose()
            P = self.mu * (Fp - FinvT) + self.lam * ti.log(J) * FinvT
            # Cauchy-ish stress term scaled to grid forces
            stress = (1.0 / J) * P @ Fp.transpose()
            # force term in MLS-MPM: -dt * Vp * 4 * inv_dx^2 * stress
            force = -self.dt * self.volume[p] * 4.0 * (self.inv_dx * self.inv_dx) * stress

            # base node
            base = (self.world_to_grid(Xp) - 0.5).cast(int)
            fx = self.world_to_grid(Xp) - base.cast(ti.f32)

            # Quadratic B-spline weights
            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]

            mp = self.mass[p]
            
            wx = ti.Vector([
                0.5 * (1.5 - fx[0]) ** 2,
                0.75 - (fx[0] - 1.0) ** 2,
                0.5 * (fx[0] - 0.5) ** 2,
            ])
            wy = ti.Vector([
                0.5 * (1.5 - fx[1]) ** 2,
                0.75 - (fx[1] - 1.0) ** 2,
                0.5 * (fx[1] - 0.5) ** 2,
            ])
            wz = ti.Vector([
                0.5 * (1.5 - fx[2]) ** 2,
                0.75 - (fx[2] - 1.0) ** 2,
                0.5 * (fx[2] - 0.5) ** 2,
            ])

            for i, j, k in ti.ndrange(3, 3, 3):
                offset = ti.Vector([i, j, k])
                node = base + offset
                if 0 <= node[0] < self.nx and 0 <= node[1] < self.ny and 0 <= node[2] < self.nz:
                    weight = wx[i] * wy[j] * wz[k]
                    dpos = (offset.cast(ti.f32) - fx) * self.dx
                    momentum = mp * (vp + Cp @ dpos)
                    self.grid_v[node] += weight * (momentum + force @ dpos)
                    self.grid_m[node] += weight * mp

    @ti.kernel
    def grid_op(self):
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v = self.grid_v[I] / m
                # gravity
                v += self.dt * self.gravity

                # ground collision (very simple)
                if ti.static(self.has_ground):
                    X = self.grid_to_world(I.cast(ti.f32))
                    if X[1] < self.ground_y and v[1] < 0:
                        v[1] = 0.0

                self.grid_v[I] = v

    @ti.kernel
    def g2p(self):
        for p in range(self.P):
            Xp = self.x[p]
            base = (self.world_to_grid(Xp) - 0.5).cast(int)
            fx = self.world_to_grid(Xp) - base.cast(ti.f32)

            w = [0.5 * (1.5 - fx) ** 2,
                 0.75 - (fx - 1.0) ** 2,
                 0.5 * (fx - 0.5) ** 2]

            new_v = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            
            wx = ti.Vector([
                0.5 * (1.5 - fx[0]) ** 2,
                0.75 - (fx[0] - 1.0) ** 2,
                0.5 * (fx[0] - 0.5) ** 2,
            ])
            wy = ti.Vector([
                0.5 * (1.5 - fx[1]) ** 2,
                0.75 - (fx[1] - 1.0) ** 2,
                0.5 * (fx[1] - 0.5) ** 2,
            ])
            wz = ti.Vector([
                0.5 * (1.5 - fx[2]) ** 2,
                0.75 - (fx[2] - 1.0) ** 2,
                0.5 * (fx[2] - 0.5) ** 2,
            ])

            for i, j, k in ti.ndrange(3, 3, 3):
                offset = ti.Vector([i, j, k])
                node = base + offset
                if 0 <= node[0] < self.nx and 0 <= node[1] < self.ny and 0 <= node[2] < self.nz:
                    weight = wx[i] * wy[j] * wz[k]
                    dpos = (offset.cast(ti.f32) - fx) * self.dx
                    gv = self.grid_v[node]
                    new_v += weight * gv
                    new_C += 4.0 * self.inv_dx * weight * gv.outer_product(dpos)


            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v

    def step(self):
        self.clear_grid()
        self.p2g()
        self.grid_op()
        self.g2p()

    def get_positions(self) -> np.ndarray:
        return self.x.to_numpy()