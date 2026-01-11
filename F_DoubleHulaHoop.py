import numpy as np
import matplotlib.pyplot as plt

class ColeHopfSolver:
    """
    A solver for Mean-Field Control problems
    """
    def __init__(self, N=100, epsilon=1e-10, T=1.0, dt=0.001):
        self.N = N
        self.epsilon = epsilon
        self.T = T
        self.dt = dt
        self.num_iterations = 1     # Increase for fixed-point convergence of Measure
        self.K_samples = 100        # MC samples for Kernel evaluation
        self.denom_floor = 1e-10    # Numerical stability floor

    def F_function_multidim(self, tau, x_array, cost_func, current_swarm):
        """
        Computes the target position (E[wY] / E[w]) using the Heat Kernel. 
        """
        N, d = x_array.shape
        K = self.K_samples

        # 1. Generate noise for the 'Main' particles
        # Z ~ N(0, 1) -> Y = x + sqrt(tau)*Z
        Z_main = np.random.standard_normal((N, K, d))
        X_plus_W = x_array[:, np.newaxis, :] + Z_main * np.sqrt(tau) # Shape (N, K, d)
        
        # 2. Generate noise for the 'Context' swarm (The Crowd) 
        # Shape: (N_context, d) -> (N_context, d)
        noise_context = np.random.standard_normal(current_swarm.shape) * np.sqrt(tau)
        future_swarm_proxy = current_swarm + noise_context
        
        # 3. Flatten for cost computation
        # We evaluate N*K particles against the N context particles
        flat_XW = X_plus_W.reshape(N*K, d)
        
        # Compute costs G(Y_i, future_swarm)
        # Reshape back to (N, K) to normalize weights per particle
        g_vals = cost_func(flat_XW, context=future_swarm_proxy).reshape(N, K)

        # 4. Log-Sum-Exp Trick for Weights
        # w = exp( - (g - g_min) / eps )
        min_g = np.min(g_vals, axis=1, keepdims=True)
        exponent_scalar = np.exp(-(g_vals - min_g) / self.epsilon)

        # 5. Weighted Average
        # Numerator: Sum( w * Y )
        numerator = np.sum(exponent_scalar[:, :, np.newaxis] * X_plus_W, axis=1)
        # Denominator: Sum( w )
        denominator = np.sum(exponent_scalar, axis=1)

        # Result is the "Optimal Target" (Center of Mass of low cost area)
        return numerator / (denominator[:, np.newaxis] + self.denom_floor)

    def solve_iterative(self, d, cost_function, x0_init=None):
        num_steps = int(self.T / self.dt) + 1
        time_grid = np.linspace(0, self.T, num_steps)
        dt = self.dt

        # Initialize Swarm
        if x0_init is None:
            x0_init = np.random.normal(0, 1.0, (self.N, d))

        X_current = x0_init.copy()
        X_path = np.zeros((self.N, num_steps, d))
        
        print(f"--- Simulating: N={self.N}, eps={self.epsilon}, T={self.T} ---")

        for k in range(self.num_iterations): 
            X_path[:, 0, :] = X_current
            X_t = X_current.copy()

            for i in range(1, num_steps):
                t = time_grid[i-1]
                tau = self.T - t
                
                # Singularity check: If close to T, switch to pure noise or stop drift
                if tau < 1e-5:
                    drift = 0
                else:
                    # CORRECTION 2: Generate Z inside F_function (fresh samples)
                    # CORRECTION 3: Pass current_swarm to project future context
                    target_pos = self.F_function_multidim(tau, X_t, cost_function, current_swarm=X_t)
                    
                    # Compute drift u = (Target - Current) / tau
                    drift = (target_pos - X_t) / tau

                # Update Dynamics: dX = u*dt + dW
                dW = np.random.normal(0, np.sqrt(dt), (self.N, d))
                X_t = X_t + drift * dt + dW
                X_path[:, i, :] = X_t

            # Stats
            final_swarm = X_path[:, -1, :]
            final_cost = np.mean(cost_function(final_swarm, context=final_swarm))
            print(f"Iter {k+1}: Mean Terminal Cost = {final_cost:.4f}")
            X_current = X_path[:, -1, :]    

        return X_path

# ==========================================
# Cost Functions 
# ==========================================

def cost_side_by_side_rings_2d(X, context=None, target_radius=1.0, center_offset=2.0):
    """
    NEW CASE: Two side-by-side separated rings.
    Centers located at (-center_offset, 0) and (+center_offset, 0).
    
    V(x) = min( V_left_ring, V_right_ring )
    W(x-y) = -log|x-y| (Log Repulsion)
    """
    # Default initialization cost (pull to origin if no context)
    if context is None: return np.sum(X**2, axis=1)
    
    # --- 1. Interaction Term (W) (Unchanged) ---
    # Sample context and compute log repulsion
    ctx = context[np.random.choice(context.shape[0], min(400, context.shape[0]), replace=False)]
    diff = X[:, np.newaxis, :] - ctx[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-300)
    W = -0.5 * np.log(dist_sq)

    # --- 2. Confinement Term (V) (Changed) ---
    R2 = target_radius**2
    d = center_offset

    # --- Left Ring Potential ---
    # Calculate squared distance to center C_L = (-d, 0)
    # r_L^2 = (x - (-d))^2 + (y - 0)^2 = (x+d)^2 + y^2
    X_shifted_left = X + np.array([d, 0]) 
    r_sq_L = np.sum(X_shifted_left**2, axis=1)
    V_L = 0.5 * (r_sq_L - R2)**2

    # --- Right Ring Potential ---
    # Calculate squared distance to center C_R = (+d, 0)
    # r_R^2 = (x - d)^2 + (y - 0)^2
    X_shifted_right = X - np.array([d, 0])
    r_sq_R = np.sum(X_shifted_right**2, axis=1)
    V_R = 0.5 * (r_sq_R - R2)**2
    
    # --- Combine ---
    # A particle falls into the well of whichever ring center is closer
    V = np.minimum(V_L, V_R)

    return V + 2 * np.mean(W, axis=1)


def Energy_side_by_side_rings_2d(X, context=None, target_radius=1.0, center_offset=2.0):
    """
    NEW CASE: Two side-by-side separated rings.
    Centers located at (-center_offset, 0) and (+center_offset, 0).
    
    V(x) = min( V_left_ring, V_right_ring )
    W(x-y) = -log|x-y| (Log Repulsion)
    """
    # Default initialization cost (pull to origin if no context)
    if context is None: return np.sum(X**2, axis=1)
    
    # --- 1. Interaction Term (W) (Unchanged) ---
    # Sample context and compute log repulsion
    ctx = context[np.random.choice(context.shape[0], min(400, context.shape[0]), replace=False)]
    diff = X[:, np.newaxis, :] - ctx[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-300)
    W = -0.5 * np.log(dist_sq)

    # --- 2. Confinement Term (V) (Changed) ---
    R2 = target_radius**2
    d = center_offset

    # --- Left Ring Potential ---
    # Calculate squared distance to center C_L = (-d, 0)
    # r_L^2 = (x - (-d))^2 + (y - 0)^2 = (x+d)^2 + y^2
    X_shifted_left = X + np.array([d, 0]) 
    r_sq_L = np.sum(X_shifted_left**2, axis=1)
    V_L = 0.5 * (r_sq_L - R2)**2

    # --- Right Ring Potential ---
    # Calculate squared distance to center C_R = (+d, 0)
    # r_R^2 = (x - d)^2 + (y - 0)^2
    X_shifted_right = X - np.array([d, 0])
    r_sq_R = np.sum(X_shifted_right**2, axis=1)
    V_R = 0.5 * (r_sq_R - R2)**2
    
    # --- Combine ---
    # A particle falls into the well of whichever ring center is closer
    V = np.minimum(V_L, V_R)

    return V + np.mean(W, axis=1)

 

# ==========================================
# Visualization Runner
# ==========================================
def run_example_double_ring():
    print("\n--- Running Double Ring Example ---")
    
    # Epsilon = 0.1 allows for a visible "Gaussian Cloud" equilibrium. 
    solver = ColeHopfSolver(N=400, epsilon=1e-299, T=1.0, dt=0.001)

    # Initial Condition: Two separate blobs to see them merge
    x0_a = np.random.normal(0, 1, (200, 2))
    x0_b = np.random.normal(0, 1, (200, 2))
    x0 = np.vstack([x0_a, x0_b])

    traj = solver.solve_iterative(d=2, cost_function=cost_side_by_side_rings_2d, x0_init=x0)
    
    final_swarm = traj[:, -1, :]
    final_cost = np.mean(Energy_side_by_side_rings_2d(final_swarm, context=final_swarm))
    print(f" Mean Energy = {final_cost:.4f}")


    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    times = [0, int(traj.shape[1]/2), -1]
    titles = ["T=0 (Start)", "T=0.5 (Parting)", "T=1.0 (Equilibrium)"]
    
    lims = [-4, 4]
    
    for i, ax in enumerate(axes):
        t_idx = times[i]
        pts = traj[:, t_idx, :]
        ax.scatter(pts[:, 0], pts[:, 1], c='blue', alpha=0.6, s=15, edgecolor='k', linewidth=0.5)
        ax.set_title(titles[i])
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw Center of Mass
        # com = np.mean(pts, axis=0)
        # ax.scatter(com[0], com[1], c='red', marker='x', s=100, label='Center of Mass')
        # if i==0: ax.legend()

    plt.tight_layout()
    plt.show()



 
if __name__ == "__main__":
    run_example_double_ring()