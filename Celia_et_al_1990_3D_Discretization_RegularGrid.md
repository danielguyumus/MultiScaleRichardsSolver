# 3D Richards Discretization (Celia et al., 1990) and Its Representation in This Solver

This note explains:

1. The mixed-form Richards equation used by Celia et al. (1990).
2. The modified Picard linearization used to robustly solve the nonlinear system.
3. The 3D finite-volume discretization on a Cartesian grid.
4. How each term appears in `solver/richards_3d_modified_picard.py`.

---

## 1) Governing PDE in 3D

In pressure-head form (with elevation head included in total head), the mixed Richards equation is

$$
\frac{\partial \theta(h)}{\partial t}
= \nabla \cdot \left[K(h)\nabla(h+z)\right] + q.
$$

Where:

- $h$ is pressure head [L]
- $z$ is elevation head [L]
- $\theta(h)$ is volumetric water content [-]
- $K(h)$ is hydraulic conductivity [L/T]
- $q$ is a volumetric source/sink [1/T]

In this solver, source/sink behavior is represented through boundary fluxes (not a volumetric internal $q$ term).

---

## 2) Celia et al. (1990): Modified Picard Idea

For time level $n \to n+1$, modified Picard iteration solves for a correction

$$
\delta h = h^{m+1} - h^m,
$$

while linearizing the water-content term as

$$
\theta(h^{m+1}) \approx \theta(h^m) + C(h^m)\,\delta h,
\qquad C(h)=\frac{d\theta}{dh}.
$$

This gives a linear system each iteration. 

$$
\left(\frac{C^m}{\Delta t} - \mathcal{L}[K^m]\right)\delta h = -\frac{\theta^m-\theta^n}{\Delta t} + \nabla\cdot\left(K^m\nabla (h^m+z)\right) + \text{BC/source terms}
$$

Written explicitly for each control volume $P$:

$$
\frac{V C_P^m}{\Delta t}\,\delta h_P
+ \sum_{N\in\mathcal{N}(P)} T_{PN}(\delta h_P-\delta h_N)
= -\frac{V(\theta_P^m-\theta_P^n)}{\Delta t}
+ \sum_{N\in\mathcal{N}(P)} T_{PN}\left[(h_N^m-h_P^m)+(z_N-z_P)\right]
+ B_P
$$

with

$$
T_{PN}=g_{PN}K_{PN}^{\text{face}},
\qquad
K_{PN}^{\text{face}}=\frac{2K_P^mK_N^m}{K_P^m+K_N^m}.
$$

Then update:

$$
h^{m+1}=h^m+\delta h.
$$

This is exactly the structure assembled in the solver:

- LHS contains storage Jacobian ($C^m/\Delta t$) and diffusion-like conductance couplings.
- RHS contains the nonlinear residual from $\theta$ and gravity+pressure-head flux divergence evaluated at iterate $m$.

---

## 3) 3D Control-Volume Discretization

### Grid and indexing

The domain is discretized into $(n_x,n_y,n_z)$ cells with spacings $(\Delta x,\Delta y,\Delta z)$.

- Cell index: $(i,j,k)$
- Volume: $V = \Delta x\,\Delta y\,\Delta z$
- Unknown: $h_{i,j,k}$ at cell center

In code, flattening is done with

$$
p = (k\,n_y + j)\,n_x + i.
$$

### Face fluxes and transmissibility

For any neighboring pair $(P,N)$ sharing one face:

$$
F_{P\leftarrow N} = T_{PN}\left[(h_N-h_P) + (z_N-z_P)\right],
$$

with transmissibility

$$
T_{PN} = g_{PN}\,K_{PN}^{\text{face}},
\qquad
K_{PN}^{\text{face}} = \frac{2K_P K_N}{K_P+K_N}.
$$

$g_{PN}$ is the geometric factor (face area over center distance):

- $x$-neighbor: $g = (\Delta y\Delta z)/\Delta x$
- $y$-neighbor: $g = (\Delta x\Delta z)/\Delta y$
- $z$-neighbor: $g = (\Delta x\Delta y)/\Delta z$

This is a standard finite-volume form of $\nabla\cdot[K\nabla(h+z)]$ in 3D.

### Per-cell linear equation (iteration $m$)

For each cell $P$:

$$
\frac{V C_P^m}{\Delta t}\,\delta h_P
+ \sum_{N\in\mathcal{N}(P)} T_{PN}(\delta h_P-\delta h_N)
= -\frac{V(\theta_P^m-\theta_P^n)}{\Delta t}
+ \sum_{N\in\mathcal{N}(P)} T_{PN}\left[(h_N^m-h_P^m)+(z_N-z_P)\right]
+ B_P,
$$

where $B_P$ is boundary contribution (top flux in this implementation).

---

## 4) Boundary Conditions Used Here

### Top boundary (Neumann flux)

At the top layer ($k=n_z-1$), prescribed infiltration flux is added:

$$
B_P = q_{\text{top}}(i,j)\,\Delta x\,\Delta y.
$$

In code, `top_flux` can be:

- scalar (uniform over top surface), or
- 2D field of shape `(ny, nx)`.

### Other boundaries

Lateral sides and bottom are treated as no-flow in this class by only connecting existing in-domain neighbors.
No ghost-cell or external flux term is added there, so boundary normal flux is zero.

---

## 5) Mapping Equation Terms to `richards_3d_modified_picard.py`

### Constitutive relations

- `theta(h)`: van Genuchten retention (plus saturated cap).
- `conductivity(h)`: Mualem-van Genuchten conductivity.
- `capacity(h)`: $C(h)=d\theta/dh$ for unsaturated region, `S_s` in saturated region.

### Modified Picard loop (`solve_step`)

For each nonlinear iteration:

1. Evaluate
   - `theta_m = theta(h_m)`
   - `theta_n = theta(h_n)`
   - `c_m = capacity(h_m)`
   - `k_m = conductivity(h_m)`
2. Assemble sparse system `lhs * dh = rhs`.
3. Solve `dh` with `spsolve`.
4. Update `h_m += dh`.
5. Stop when `||dh||_inf < tol`.

### Storage term

Code contribution:

```python
lhs[p, p] += cell_volume * c_m[k, j, i] / dt
rhs[p] -= cell_volume * (theta_m[k, j, i] - theta_n[k, j, i]) / dt
```

This is the modified-Picard linearization of $\partial\theta/\partial t$.

### Neighbor coupling and flux divergence

For each neighbor `pn`:

```python
k_face = harmonic_mean(k_m[P], k_m[N])
conductance = g * k_face

lhs[p, p] += conductance
lhs[p, pn] -= conductance

total_head_grad = (h_m[N] - h_m[P]) + (z_N - z_P)
rhs[p] += conductance * total_head_grad
```

Interpretation:

- `lhs` terms represent $\sum T_{PN}(\delta h_P-\delta h_N)$.
- `rhs` term represents divergence of $K\nabla(h+z)$ at iterate $m$.

### Top flux boundary

Code contribution:

```python
if k == nz - 1:
    rhs[p] += top_flux_field[j, i] * dx * dy
```

This is the Neumann flux source term over top-face area.

---

## 6) Why This Matches Celia-Style Mass-Conservative Formulation

- Uses **mixed form** via direct $\theta(h)$ in time term.
- Uses **modified Picard** via $C(h^m)\delta h$ linearization.
- Uses **finite-volume flux balances** across all 3D faces.
- Uses **harmonic averaging** for interface conductivity, standard for heterogeneous media.
- Solves a sparse linear system at each Picard iteration until correction is small.

These are the essential ingredients of robust, mass-conservative Richards solvers in the Celia et al. lineage, extended here to a full 3D Cartesian domain.

---

## 7) Practical Notes for This Implementation

- `z_centers` is constructed from `z_base + (k+0.5)dz`, so gravity is handled through center elevation differences.
- `S_s` prevents zero capacity in saturated cells and stabilizes the saturated regime.
- `solve_transient` repeatedly calls `solve_step` and stores full state history.

If desired, this class can be extended with additional boundary options (free drainage, fixed head, seepage face, evapotranspiration sink) while preserving the same modified-Picard core.