# Richards Solver: Physics and Usage Guide

## Overview

This solver implements a **3D-prism finite-volume style Richards equation model** where:

- Horizontal geometry is represented by prisms (regular or unstructured planform).
- Vertical direction is layered using `dz`.
- Unknown is pressure head `h` in each cell.
- Fluxes are computed between neighboring cells using hydraulic conductivity and total-head gradients.

The main implementation is in `richards_model.py` (`RichardsSolver`).

---

## Governing Physics

The model uses the mixed form of Richards equation:

$$
\frac{\partial \theta(h)}{\partial t} = \nabla \cdot \left(K(h)\nabla(h+z)\right) + q
$$

Where:

- $h$: pressure head [m]
- $z$: elevation head [m]
- $\theta(h)$: volumetric water content [-]
- $K(h)$: unsaturated hydraulic conductivity [m/s]
- $q$: source/sink term [1/s], here represented as flux boundary contribution at the top

### Soil hydraulic model

Van Genuchten-Mualem style constitutive laws are used:

- `get_theta(h, lay, prism)`
- `get_K(h, lay, prism)`
- `get_C(h, lay, prism)`

For `h >= 0` (saturated side):

- `theta = theta_s`
- `K = Ks`
- `C = S_s` (specific storage, configurable)

For `h < 0`: unsaturated relations are used.

---

## Spatial Discretization

## 1) Cell volumes

For prism `i`, layer `k`:

- Horizontal area: `A_ij[i]`
- Layer thickness: `dz[k]`
- Cell volume: `V_ik = A_ij[i] * dz[k]`

## 2) Vertical interfaces

Cells in the same prism are connected by layer adjacency (`k±1`).

- Vertical conductance geometry factor: `G_v = A_ij[i] / dz_dist`
- `dz_dist` is computed from layer-center elevations

## 3) Lateral interfaces (unstructured-compatible)

Neighbor prism connectivity comes from `adj_prisms`.

- Interface width: `W_ij[i][j]`
- Centroid distance: `L_ij[i][j]`
- Lateral geometry factor: `G_lat = (W_ij[i][j] * dz[k]) / L_ij[i][j]`

This works for regular and unstructured grids as long as `adj_prisms`, `W_ij`, `L_ij`, and `A_ij` are consistent.

---

## Flux Formulation (Current Implementation)

For any face between cells `i` and `j` in the same layer or between layers:

$$
q_{i \leftarrow j} = \left(G_{ij} K_{face}\right) \left[(h_j-h_i) + (z_j-z_i)\right]
$$

- Uses **total-head gradient** $\Delta(h+z)$.
- `K_face` uses harmonic averaging of adjacent cell conductivities.

This unifies vertical and lateral fluxes under the same physical form.

---

## Time Integration / Nonlinearity

- Implicit Picard-style iteration inside `solve_step(...)`.
- Mass accumulation term linearized with `C(h_m)` and `theta(h_m)-theta(h_n)`.
- Linear system solved with sparse direct solve (`spsolve`).

Convergence criterion:

- `||dh||_inf < tol`.

---

## Boundary Conditions and Sources

### Top infiltration

`apply_top_boundary(...)` adds top-layer rainfall flux:

- Global uniform rainfall: `rainfall_intensity`
- Optional subset by prism ID: `rainfall_prisms`
- Optional per-prism map/vector: `rainfall_by_prism`

Top inflow for diagnostics is computed by `get_boundary_flux_with_rain(...)`.

### Rain targeting

- `rainfall_prisms=None` => apply rainfall to all top cells
- `rainfall_prisms=[...]` => apply to selected prism IDs only
- Nested iterables and numpy arrays are accepted and normalized

---

## Saturated Storage Parameter `S_s`

`S_s` is optional and physically important for saturated compressibility.

- Scalar form: same value for all cells
- Array form: shape `(n_prisms, n_layers)`

If omitted, default is `1e-10`.

---

## Diagnostics Available

- `calculate_storage(h_array)`
- `get_boundary_flux_with_rain(h_array)`
- `get_lateral_fluxes(h_array, layer=None)`
- `get_total_lateral_exchange(h_array, layer=None)`

These allow mass balance checks and comparison between grid types.

---

## Minimal Usage

```python
from richards_model import RichardsSolver
import numpy as np

solver = RichardsSolver(
    adj_prisms=adj,
    A_ij=areas,
    W_ij=widths,
    L_ij=lengths,
    dz=np.full(5, 0.04),
    base_elevations=np.full(len(areas), 10.1),
    alpha=0.067*100,
    n_vg=2,
    theta_r=np.full((len(areas), 5), 0.075),
    theta_s=np.full((len(areas), 5), 0.287),
    Ks=np.full((len(areas), 5), 100/100/3600),
    S_s=1e-10,
    rainfall_intensity=0.1/100/86400,
    rainfall_prisms=[1, 4, 5]  # optional
)

h = np.full(solver.total_cells, -10.0)
dt = 864
for _ in range(10):
    h = solver.solve_step(h, dt)

print("Storage:", solver.calculate_storage(h))
print("Top inflow rate:", solver.get_boundary_flux_with_rain(h))
print("Total lateral exchange:", solver.get_total_lateral_exchange(h))
```

---

## Notes for Regular vs Unstructured Grids

The physics code is shared. Differences come only from geometry/connectivity inputs:

- Regular: structured neighbor graph and constant interface geometry
- Unstructured: polygon-derived adjacency, shared-edge widths, centroid distances

To compare both experiments fairly, keep these identical:

- `alpha, n_vg, theta_r, theta_s, Ks, S_s`
- `dt`, number of steps, initial condition
- total rainfall volume (and intended wetting pattern)

---

## Known Modeling Simplifications

- Infiltration is imposed as prescribed top flux (no explicit ponding/capacity switch).
- No explicit hysteresis in retention/conductivity.
- No root uptake, evaporation, or bottom drainage boundary options built-in.

These can be added by extending boundary/source handling.
