import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class RichardsSolver:
    def __init__(self, **kwargs):
        self.dz = kwargs.get('dz', [0.1, 0.1, 0.25, 0.5, 0.15])
        self.n_layers = len(self.dz)
        # --- Default Soil Physics (Van Genuchten) ---
        self.alpha = kwargs.get('alpha', 0.0335)
        self.n_vg = kwargs.get('n_vg', 2.0)
        self.m_vg = 1 - 1 / self.n_vg
        self.theta_r = kwargs.get('theta_r', np.full(self.n_layers,0.102))
        self.theta_s = kwargs.get('theta_s', np.full(self.n_layers,0.368))
        self.Ks = kwargs.get('Ks', np.full(self.n_layers,0.00922))

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
        
        # Derived Domain Properties
        self.n_prisms = len(self.A_ij)
        self.total_cells = self.n_prisms * self.n_layers
        self.rainfall_intensity = kwargs.get('rainfall_intensity', 2e-8)

    # --- Physics Methods ---
    def get_theta(self, h, lay):
        if h >= 0: return self.theta_s[lay]
        return self.theta_r[lay] + (self.theta_s[lay] - self.theta_r[lay]) / (1 + (self.alpha * abs(h))**self.n_vg)**self.m_vg

    def get_K(self, h, lay):
        if h >= 0: return self.Ks[lay]
        Se = (self.get_theta(h,lay) - self.theta_r[lay]) / (self.theta_s[lay] - self.theta_r[lay])
        return self.Ks[lay] * Se**0.5 * (1 - (1 - Se**(1/self.m_vg))**self.m_vg)**2

    def get_C(self, h,lay):
        if h >= 0: return 1e-10
        abs_h = abs(h)
        num = (self.theta_s[lay] - self.theta_r[lay]) * self.m_vg * self.n_vg * (self.alpha**self.n_vg) * (abs_h**(self.n_vg-1))
        den = (1 + (self.alpha * abs_h)**self.n_vg)**(self.m_vg + 1)
        return num / den

    def get_G_lateral(self, i, j, lay):
        return (self.W_ij[i][j] * self.dz[lay]) / self.L_ij[i][j]

    def apply_top_boundary(self, R):
        k_top = self.n_layers - 1
        for i in range(self.n_prisms):
            idx = k_top * self.n_prisms + i
            # Calculate flux: Area * Intensity
            Q_rain = self.A_ij[i] * self.rainfall_intensity
            # Add to the Residual vector for the top cells
            R[idx] += Q_rain
        return R

    # --- Solver ---
    def solve_step(self, h_n, dt, max_iter=100, tol=1e-4):
        h_m = h_n.copy()
        
        for iteration in range(max_iter):
            LHS = lil_matrix((self.total_cells, self.total_cells))
            RHS = np.zeros(self.total_cells)
            RHS = self.apply_top_boundary(RHS)
            
            for k in range(self.n_layers):
                for i in range(self.n_prisms):
                    idx_i = k * self.n_prisms + i
                    
                    # 1. Mass Accumulation
                    V_ik = self.A_ij[i] * self.dz[k]
                    Ci = self.get_C(h_m[idx_i],k)
                    theta_m = self.get_theta(h_m[idx_i],k)
                    theta_n = self.get_theta(h_n[idx_i],k)
                    
                    LHS[idx_i, idx_i] += V_ik * Ci / dt
                    RHS[idx_i] -= (V_ik / dt) * (theta_m - theta_n)

                    # 2. Vertical Fluxes
                    for direction in [-1, 1]: 
                        adj_k = k + direction
                        if 0 <= adj_k < self.n_layers:
                            adj_k_idx = adj_k * self.n_prisms + i  
                            K_face = 2 / (1/self.get_K(h_m[idx_i],k) + 1/self.get_K(h_m[adj_k_idx],adj_k))
                            dz_avg = (self.dz[k] + self.dz[adj_k]) / 2
                            G_v = self.A_ij[i] / dz_avg
                        
                            LHS[idx_i, idx_i] += G_v * K_face
                            LHS[idx_i, adj_k_idx] -= G_v * K_face
                        
                            grad_h = (h_m[adj_k_idx] - h_m[idx_i])
                            gravity = 1.0 if direction == 1 else -1.0
                            RHS[idx_i] += G_v * K_face * (grad_h + gravity)

                    # 3. Horizontal Fluxes
                    for j in self.adj_prisms[i]:
                        if i == j: continue
                        idx_j = k * self.n_prisms + j
                        K_face = 2/(1/self.get_K(h_m[idx_i],k) + 1/self.get_K(h_m[idx_j],k))
                        conductance = self.get_G_lateral(i, j, k) * K_face
                        
                        LHS[idx_i, idx_i] += conductance
                        LHS[idx_i, idx_j] -= conductance
                        RHS[idx_i] += conductance * (h_m[idx_j] - h_m[idx_i])

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
                theta = self.get_theta(h_array[idx],k)
                # Area * Thickness * Water Content
                total_water += self.A_ij[i] * V_layer * theta
        return total_water

    def get_boundary_flux_with_rain(self, h_array):
        flux_top = 0
    
        # 1. Top is now just the sum of rain across all prisms
        for i in range(self.n_prisms):
            flux_top += self.A_ij[i] * self.rainfall_intensity
            
        return flux_top