import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class Richards3DModifiedPicard:
    def __init__(
        self,
        nx,
        ny,
        nz,
        dx,
        dy,
        dz,
        alpha=0.0335,
        n_vg=2.0,
        theta_r=0.102,
        theta_s=0.368,
        Ks=0.00922,
        S_s=1e-10,
        z_base=0.0,
    ):
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.n = self.nx * self.ny * self.nz

        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.cell_volume = self.dx * self.dy * self.dz

        self.alpha = float(alpha)
        self.n_vg = float(n_vg)
        self.m_vg = 1.0 - 1.0 / self.n_vg

        self.theta_r = self._as_field(theta_r)
        self.theta_s = self._as_field(theta_s)
        self.Ks = self._as_field(Ks)
        self.S_s = self._as_field(S_s)

        self.z_base = self._as_xy_field(z_base)
        self.z_centers = self.z_base[None, :, :] + (np.arange(self.nz)[:, None, None] + 0.5) * self.dz

    def _as_field(self, value):
        if np.isscalar(value):
            return np.full((self.nz, self.ny, self.nx), float(value), dtype=float)

        arr = np.asarray(value, dtype=float)
        if arr.shape != (self.nz, self.ny, self.nx):
            raise ValueError(
                f"Expected shape {(self.nz, self.ny, self.nx)} but got {arr.shape}."
            )
        return arr

    def _as_xy_field(self, value):
        if np.isscalar(value):
            return np.full((self.ny, self.nx), float(value), dtype=float)

        arr = np.asarray(value, dtype=float)
        if arr.shape != (self.ny, self.nx):
            raise ValueError(f"Expected shape {(self.ny, self.nx)} but got {arr.shape}.")
        return arr

    def _idx(self, k, j, i):
        return (k * self.ny + j) * self.nx + i

    def _unflatten(self, h_flat):
        arr = np.asarray(h_flat, dtype=float)
        if arr.size != self.n:
            raise ValueError(f"Expected {self.n} values but got {arr.size}.")
        return arr.reshape((self.nz, self.ny, self.nx))

    def _flatten(self, h_3d):
        return np.asarray(h_3d, dtype=float).reshape(-1)

    def theta(self, h):
        h = np.asarray(h, dtype=float)
        out = np.empty_like(h)

        sat = h >= 0.0
        out[sat] = self.theta_s[sat]

        unsat = ~sat
        abs_h = np.abs(h[unsat])
        out[unsat] = self.theta_r[unsat] + (
            (self.theta_s[unsat] - self.theta_r[unsat])
            / (1.0 + (self.alpha * abs_h) ** self.n_vg) ** self.m_vg
        )
        return out

    def conductivity(self, h):
        h = np.asarray(h, dtype=float)
        out = np.empty_like(h)
        theta_full = self.theta(h)

        sat = h >= 0.0
        out[sat] = self.Ks[sat]

        unsat = ~sat
        se = (theta_full[unsat] - self.theta_r[unsat]) / (
            self.theta_s[unsat] - self.theta_r[unsat] + 1e-30
        )
        se = np.clip(se, 0.0, 1.0)
        out[unsat] = self.Ks[unsat] * se ** 0.5 * (
            1.0 - (1.0 - se ** (1.0 / self.m_vg)) ** self.m_vg
        ) ** 2
        return out

    def capacity(self, h):
        h = np.asarray(h, dtype=float)
        out = np.empty_like(h)

        sat = h >= 0.0
        out[sat] = self.S_s[sat]

        unsat = ~sat
        abs_h = np.abs(h[unsat])
        num = (
            (self.theta_s[unsat] - self.theta_r[unsat])
            * self.m_vg
            * self.n_vg
            * (self.alpha ** self.n_vg)
            * (abs_h ** (self.n_vg - 1.0))
        )
        den = (1.0 + (self.alpha * abs_h) ** self.n_vg) ** (self.m_vg + 1.0)
        out[unsat] = num / (den + 1e-30)
        return out

    @staticmethod
    def _harmonic_mean(a, b):
        return 2.0 * a * b / (a + b + 1e-30)

    def solve_step(self, h_n, dt, top_flux=0.0, max_iter=50, tol=1e-6):
        h_n_3d = self._unflatten(h_n)
        h_m = h_n_3d.copy()

        top_flux_field = np.asarray(top_flux, dtype=float)
        if top_flux_field.ndim == 0:
            top_flux_field = np.full((self.ny, self.nx), float(top_flux_field))
        elif top_flux_field.shape != (self.ny, self.nx):
            raise ValueError(
                f"top_flux must be scalar or shape {(self.ny, self.nx)}, got {top_flux_field.shape}"
            )

        for _ in range(max_iter):
            lhs = lil_matrix((self.n, self.n), dtype=float)
            rhs = np.zeros(self.n, dtype=float)

            theta_m = self.theta(h_m)
            theta_n = self.theta(h_n_3d)
            c_m = self.capacity(h_m)
            k_m = self.conductivity(h_m)

            for k in range(self.nz):
                for j in range(self.ny):
                    for i in range(self.nx):
                        z_i = self.z_centers[k, j, i]
                        p = self._idx(k, j, i)

                        lhs[p, p] += self.cell_volume * c_m[k, j, i] / dt
                        rhs[p] -= self.cell_volume * (theta_m[k, j, i] - theta_n[k, j, i]) / dt

                        if k == self.nz - 1:
                            rhs[p] += top_flux_field[j, i] * self.dx * self.dy

                        neighbors = []
                        if i > 0:
                            neighbors.append((k, j, i - 1, self.dy * self.dz / self.dx))
                        if i < self.nx - 1:
                            neighbors.append((k, j, i + 1, self.dy * self.dz / self.dx))
                        if j > 0:
                            neighbors.append((k, j - 1, i, self.dx * self.dz / self.dy))
                        if j < self.ny - 1:
                            neighbors.append((k, j + 1, i, self.dx * self.dz / self.dy))
                        if k > 0:
                            neighbors.append((k - 1, j, i, self.dx * self.dy / self.dz))
                        if k < self.nz - 1:
                            neighbors.append((k + 1, j, i, self.dx * self.dy / self.dz))

                        for kn, jn, inn, g in neighbors:
                            pn = self._idx(kn, jn, inn)
                            k_face = self._harmonic_mean(k_m[k, j, i], k_m[kn, jn, inn])
                            conductance = g * k_face

                            lhs[p, p] += conductance
                            lhs[p, pn] -= conductance

                            z_n = self.z_centers[kn, jn, inn]
                            total_head_grad = (h_m[kn, jn, inn] - h_m[k, j, i]) + (z_n - z_i)
                            rhs[p] += conductance * total_head_grad

            dh = spsolve(lhs.tocsr(), rhs).reshape((self.nz, self.ny, self.nx))
            h_m += dh

            if np.linalg.norm(dh.ravel(), ord=np.inf) < tol:
                break

        return self._flatten(h_m)

    def solve_transient(self, h0, dt, n_steps, top_flux=0.0, max_iter=50, tol=1e-6):
        h = self._flatten(self._unflatten(h0))
        history = [h.copy()]

        for _ in range(int(n_steps)):
            h = self.solve_step(h, dt, top_flux=top_flux, max_iter=max_iter, tol=tol)
            history.append(h.copy())

        return np.array(history)


if __name__ == "__main__":
    solver = Richards3DModifiedPicard(
        nx=10,
        ny=10,
        nz=12,
        dx=0.1,
        dy=0.1,
        dz=0.05,
        z_base=10.1,
        alpha=0.067,
        n_vg=2.0,
        theta_r=0.075,
        theta_s=0.287,
        Ks=100.0 / 100.0 / 3600.0,
        S_s=1e-10,
    )

    h0 = np.full(solver.n, -1.0)
    dt = 60.0
    top_flux = 1.0e-7

    trajectory = solver.solve_transient(
        h0=h0,
        dt=dt,
        n_steps=10,
        top_flux=top_flux,
        max_iter=40,
        tol=1e-5,
    )

    h_final = trajectory[-1]
    print("Simulation complete")
    print("Min/Max pressure head:", np.min(h_final), np.max(h_final))
