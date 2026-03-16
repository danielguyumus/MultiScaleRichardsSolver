import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from collections.abc import Iterable

class RichardsSolver:
    def __init__(self, **kwargs):

        # --- Domain Definition ---
        self.adj_prisms = kwargs.get('adj_prisms', {
            0: [1, 2, 3, 4, 5, 6], 1: [0, 2], 2: [0, 2, 3], 
            3: [0, 2], 4: [0, 5], 5: [0, 4], 6: [0]
        })
        self.A_ij = kwargs.get('A_ij', {0: 9.75, 1: 3.5, 2: 4, 3: 4, 4: 2, 5: 4, 6: 6.25})
        self.W_ij = kwargs.get('W_ij', {
            0: [0, 1.5, 2, 2, 1, 2, 5.5], 1: [1.5, 0, 2, 0, 0, 0, 0],
            2: [2, 2, 0, 2, 0, 0, 0], 3: [2, 0, 2, 0, 0, 0, 0],
            4: [1, 0, 0, 0, 0, 2, 0], 5: [2, 0, 0, 0, 2, 0, 0],
            6: [5.5, 0, 0, 0, 0, 0, 0]
        })
        self.L_ij = kwargs.get('L_ij', {
            0: [0, 4.42, 3.44, 3.44, 5.87, 3.44, 1.45], 1: [4.42, 0, 1.87, 0, 0, 0, 0],
            2: [3.44, 1.87, 0, 2, 0, 0, 0], 3: [3.44, 0, 2, 0, 0, 0, 0],
            4: [5.87, 0, 0, 0, 0, 2.5, 0], 5: [3.44, 0, 0, 0, 2.5, 0, 0],
            6: [1.45, 0, 0, 0, 0, 0, 0]
        })
        self.dz = kwargs.get('dz', [0.1, 0.1, 0.25, 0.5, 0.15])
        self.n_layers = len(self.dz)
        self.base_elevations = kwargs.get('base_elevations',[10.0, 10.2, 10.5, 10.3, 10.1, 10.1])  # Elevation of the bottom-most face for each prisms

        # --- Derived Domain Properties
        self.n_prisms = len(self.A_ij)
        self.total_cells = self.n_prisms * self.n_layers
        self.rainfall_intensity = kwargs.get('rainfall_intensity', 2e-8)
        self.rainfall_prisms = kwargs.get('rainfall_prisms', None)
        self.rainfall_by_prism = kwargs.get('rainfall_by_prism', None)
        self.rainfall_prism_set = self._normalize_prism_ids(self.rainfall_prisms)
        
        # --- Default Soil Physics (Van Genuchten) ---
        self.alpha = kwargs.get('alpha',0.0335)
        self.n_vg = kwargs.get('n_vg',2.0)
        self.m_vg = 1 - 1 / self.n_vg
        self.theta_r = kwargs.get('theta_r', np.full((self.n_prisms,self.n_layers),0.102))
        self.theta_s = kwargs.get('theta_s', np.full((self.n_prisms,self.n_layers),0.368))
        self.Ks = kwargs.get('Ks', np.full((self.n_prisms,self.n_layers),0.00922))

        S_s_input = kwargs.get('S_s', 1e-10)
        if np.isscalar(S_s_input):
            self.S_s = np.full((self.n_prisms, self.n_layers), float(S_s_input))
        else:
            self.S_s = np.array(S_s_input, dtype=float)
            if self.S_s.shape != (self.n_prisms, self.n_layers):
                raise ValueError(f"S_s must be scalar or shape {(self.n_prisms, self.n_layers)}")
        
    # --- Physics Methods ---
    def get_theta(self, h, lay, prism):
        if h >= 0: return self.theta_s[prism,lay]
        return self.theta_r[prism,lay] + (self.theta_s[prism,lay] - self.theta_r[prism,lay]) / (1 + (self.alpha * abs(h))**self.n_vg)**self.m_vg

    def get_K(self, h, lay, prism):
        if h >= 0: return self.Ks[prism,lay]
        Se = (self.get_theta(h,lay,prism) - self.theta_r[prism,lay]) / (self.theta_s[prism,lay] - self.theta_r[prism,lay])
        return self.Ks[prism,lay] * Se**0.5 * (1 - (1 - Se**(1/self.m_vg))**self.m_vg)**2

    def get_C(self, h, lay, prism):
        if h >= 0: return self.S_s[prism, lay]
        abs_h = abs(h)
        num = (self.theta_s[prism,lay] - self.theta_r[prism,lay]) * self.m_vg * self.n_vg * (self.alpha**self.n_vg) * (abs_h**(self.n_vg-1))
        den = (1 + (self.alpha * abs_h)**self.n_vg)**(self.m_vg + 1)
        return num / den

    def get_G_lateral(self, i, j, lay):
        return (self.W_ij[i][j] * self.dz[lay]) / self.L_ij[i][j]

    def get_total_head_gradient(self, h_i, z_i, h_j, z_j):
        return (h_j - h_i) + (z_j - z_i)

    def _flatten_prism_ids(self, prism_ids):
        if prism_ids is None:
            return

        if isinstance(prism_ids, np.integer):
            yield int(prism_ids)
            return

        if isinstance(prism_ids, np.ndarray):
            for value in prism_ids.ravel().tolist():
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                    yield from self._flatten_prism_ids(value)
                else:
                    yield int(value)
            return

        if isinstance(prism_ids, Iterable) and not isinstance(prism_ids, (str, bytes)):
            for value in prism_ids:
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                    yield from self._flatten_prism_ids(value)
                else:
                    yield int(value)
            return

        yield int(prism_ids)

    def _normalize_prism_ids(self, prism_ids):
        if prism_ids is None:
            return None

        normalized = set(self._flatten_prism_ids(prism_ids))
        for prism in normalized:
            if prism < 0 or prism >= self.n_prisms:
                raise ValueError(f"rainfall_prisms contains invalid prism index: {prism}")

        return normalized

    def get_rainfall_intensity_for_prism(self, prism):
        if self.rainfall_by_prism is not None:
            if isinstance(self.rainfall_by_prism, dict):
                return float(self.rainfall_by_prism.get(prism, 0.0))
            return float(self.rainfall_by_prism[prism])

        if self.rainfall_prism_set is None:
            return self.rainfall_intensity

        return self.rainfall_intensity if prism in self.rainfall_prism_set else 0.0

    # --- Boundary conditions (Infiltration) ---
    def apply_top_boundary(self, R):
        k_top = self.n_layers - 1
        for i in range(self.n_prisms):
            idx = k_top * self.n_prisms + i
            # Calculate flux: Area * Intensity
            Q_rain = self.A_ij[i] * self.get_rainfall_intensity_for_prism(i)
            # Add to the Residual vector for the top cells
            R[idx] += Q_rain
        return R

    # --- Get actual elevation ---
    def get_z_centers(self,base_elev,dz):
        z_centers = []
        current_z = base_elev
        for d in dz:
            z_centers.append(current_z + d/2.0)
            current_z += d
        return z_centers

    # --- Solver ---
    def solve_step(self, h_n, dt, max_iter=100, tol=1e-4):
        h_m = h_n.copy()
        # Create a 2D array [prism_index][layer_index]
        self.Z = np.array([self.get_z_centers(b,self.dz) for b in self.base_elevations])
        
        for iteration in range(max_iter):
            LHS = lil_matrix((self.total_cells, self.total_cells))
            RHS = np.zeros(self.total_cells)
            RHS = self.apply_top_boundary(RHS)
            
            for k in range(self.n_layers):
                for i in range(self.n_prisms):
                    idx_i = k * self.n_prisms + i
                    
                    # 1. Mass Accumulation
                    V_ik = self.A_ij[i] * self.dz[k]
                    Ci = self.get_C(h_m[idx_i],k,i)
                    theta_m = self.get_theta(h_m[idx_i],k,i)
                    theta_n = self.get_theta(h_n[idx_i],k,i)
                    
                    LHS[idx_i, idx_i] += V_ik * Ci / dt
                    RHS[idx_i] -= (V_ik / dt) * (theta_m - theta_n)

                    # 2. Vertical Fluxes
                    for direction in [-1, 1]: 
                        adj_k = k + direction
                        if 0 <= adj_k < self.n_layers:
                            adj_k_idx = adj_k * self.n_prisms + i  
                            K_face = 2 / (1/self.get_K(h_m[idx_i],k, i) + 1/self.get_K(h_m[adj_k_idx],adj_k, i))
                            dz_dist = np.abs(self.Z[i, adj_k] - self.Z[i, k])
                            G_v = self.A_ij[i] / dz_dist #dz_avg
                            #dz_avg = (self.dz[k] + self.dz[adj_k]) / 2
                            #G_v = self.A_ij[i] / dz_avg
                        
                            LHS[idx_i, idx_i] += G_v * K_face
                            LHS[idx_i, adj_k_idx] -= G_v * K_face

                            z_i = self.Z[i, k]
                            z_adj = self.Z[i, adj_k]
                            total_head_grad = self.get_total_head_gradient(h_m[idx_i], z_i, h_m[adj_k_idx], z_adj)
                            RHS[idx_i] += G_v * K_face * total_head_grad

                    # 3. Horizontal Fluxes
                    for j in self.adj_prisms[i]:
                        if i == j: continue
                        idx_j = k * self.n_prisms + j
                        K_face = 2/(1/self.get_K(h_m[idx_i],k,i) + 1/self.get_K(h_m[idx_j],k,j))
                        conductance = self.get_G_lateral(i, j, k) * K_face
                        
                        LHS[idx_i, idx_i] += conductance
                        LHS[idx_i, idx_j] -= conductance
                        z_i = self.Z[i, k]
                        z_j = self.Z[j, k]
                        total_head_grad = self.get_total_head_gradient(h_m[idx_i], z_i, h_m[idx_j], z_j)
                        RHS[idx_i] += conductance * total_head_grad

            dh = spsolve(LHS.tocsr(), RHS)
            h_m += dh
            
            if np.linalg.norm(dh, np.inf) < tol:
                break
        else:
            print("Warning: Max iterations reached without convergence.")
                
        return h_m

    def calculate_storage(self, h_array):
        """Calculates total volume of water in the domain."""
        total_water = 0
        # Flattened index logic: k * n_prisms + i
        for k in range(self.n_layers):
            V_layer = self.dz[k]
            for i in range(self.n_prisms):
                idx = k * self.n_prisms + i
                theta = self.get_theta(h_array[idx],k,i)
                # Area * Thickness * Water Content
                total_water += self.A_ij[i] * V_layer * theta
        return total_water

    def get_boundary_flux_with_rain(self, h_array):
        flux_top = 0
    
        # 1. Top is now just the sum of rain across all prisms
        for i in range(self.n_prisms):
            flux_top += self.A_ij[i] * self.get_rainfall_intensity_for_prism(i)
            
        return flux_top

    def get_lateral_fluxes(self, h_array, layer=None):
        fluxes = {}
        layers = range(self.n_layers) if layer is None else [layer]

        for k in layers:
            for i in range(self.n_prisms):
                idx_i = k * self.n_prisms + i
                for j in self.adj_prisms[i]:
                    if j <= i:
                        continue

                    idx_j = k * self.n_prisms + j
                    K_face = 2 / (1 / self.get_K(h_array[idx_i], k, i) + 1 / self.get_K(h_array[idx_j], k, j))
                    conductance = self.get_G_lateral(i, j, k) * K_face
                    z_i = self.Z[i, k]
                    z_j = self.Z[j, k]
                    total_head_grad = self.get_total_head_gradient(h_array[idx_i], z_i, h_array[idx_j], z_j)
                    q_i_from_j = conductance * total_head_grad
                    fluxes[(k, i, j)] = q_i_from_j

        return fluxes

    def get_total_lateral_exchange(self, h_array, layer=None):
        fluxes = self.get_lateral_fluxes(h_array, layer=layer)
        return sum(abs(q) for q in fluxes.values())