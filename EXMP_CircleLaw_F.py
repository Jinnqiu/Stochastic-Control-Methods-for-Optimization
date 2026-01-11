import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class ColeHopfSolver:
    """
    A solver for Mean-Field Control problems.
    """
    def __init__(self, N=200, epsilon=1e-299, T=1.0, dt=0.001):
        self.N = N
        self.epsilon = epsilon
        self.T = T
        self.dt = dt
        self.num_iterations = 1
        self.K_samples = 100
        self.denom_floor = 1e-300

    def F_function_multidim(self, tau, x_array, cost_func, current_swarm):
        """
        Calculates the optimal target position (center of mass of low-cost futures).
        """
        N, d = x_array.shape
        K = self.K_samples

        # 1. Generate noise for the 'Main' particles
        # Z ~ N(0, 1) -> Y = x + sqrt(tau)*Z
        Z_main = np.random.standard_normal((N, K, d))
        W_samples = Z_main * np.sqrt(tau)
        X_plus_W = x_array[:, np.newaxis, :] + W_samples

        # 2. Generate noise for the 'Context' swarm (The Crowd)  
        # Shape: (N, d) -> (N, d)
        noise_context = np.random.standard_normal(current_swarm.shape) * np.sqrt(tau)
        future_swarm_proxy = current_swarm + noise_context

        # 3. Flatten for cost evaluation
        flat_XW = X_plus_W.reshape(N*K, d)

        # 4. Compute Costs
        # Note: We pass the 'future_swarm_proxy' as context
        g_vals = cost_func(flat_XW, context=future_swarm_proxy).reshape(N, K)

        # 5. Log-Sum-Exp Trick for Weights
        min_g = np.min(g_vals, axis=1, keepdims=True)
        exponent_scalar = np.exp(-(g_vals - min_g) / self.epsilon)

        numerator = np.mean(exponent_scalar[:, :, np.newaxis] * X_plus_W, axis=1)
        denominator = np.mean(exponent_scalar, axis=1)

        return numerator / (denominator[:, np.newaxis] + self.denom_floor)

    def solve_iterative(self, d, cost_function, x0_init=None):
        num_steps = int(self.T / self.dt) + 1
        time_grid = np.linspace(0, self.T, num_steps)
        dt = self.dt

        if x0_init is None:
            x0_init = np.random.normal(0, 0.5, (self.N, d))

        X_path = np.zeros((self.N, num_steps, d))
        print(f"--- Iterative Sim: N={self.N}, eps={self.epsilon} ---")
        X_current = x0_init.copy()

        for k in range(self.num_iterations):
            # FIX: Finite Horizon Reset
            # Always restart from the initial distribution x0.
            # (Recycling X_T as X_0 is only for Stationary/Ergodic problems)
            X_path[:, 0, :] = X_current
            X_t = X_current.copy()

            for i in range(1, num_steps):
                t = time_grid[i-1]
                tau = self.T - t
                
                # Singularity Cutoff
                if tau < 1e-15:
                    X_path[:, i:, :] = X_t[:, np.newaxis, :]
                    break

                target_pos = self.F_function_multidim(tau, X_t, cost_function, current_swarm=X_t)
                
                drift = (target_pos - X_t) / tau
                
                dW = np.random.normal(0, np.sqrt(dt), (self.N, d))
                X_t = X_t + drift * dt + dW
                X_path[:, i, :] = X_t

            # Calculate Final Cost
            final_swarm = X_path[:, -1, :]
            total_costs = cost_function(final_swarm, context=final_swarm)
            mean_functional = np.mean(total_costs)
            print(f"Iter {k+1}: Mean Final Cost = {mean_functional:.4f}")
            X_current = X_path[:, -1, :]

        return X_path
 
# ==========================================
# Cost Functions
# ========================================== 
 

def cost_newtonian_2d(X, context=None):
    """
    Newtonian Swarm (Circle Law).
    """
    if context is None: return 0
    if context.shape[0] > 400:
        idx = np.random.choice(context.shape[0], 400, replace=False)
        ctx = context[idx]
    else:
        ctx = context
        
    diff = X[:, np.newaxis, :] - ctx[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-10)
    
    # Pairwise: 0.5*r^2 - 0.5*log(r^2)
    pairwise_cost = 0.5 * dist_sq - 0.5 * np.log(dist_sq)
    return np.mean(pairwise_cost, axis=1)

# ==========================================
# Runners
# ==========================================
 

def run_example_8_circle():
    print("\nRunning Example 8: 2D Newtonian Swarm (Circle Law)") 
    solver = ColeHopfSolver(N=200, epsilon=1e-10, T=1.0, dt=0.001)
    
    x0 = np.random.normal(0, 0, (200, 2))
    
    traj = solver.solve_iterative(d=2, cost_function=cost_newtonian_2d, x0_init=x0)
    final = traj[:, -1, :]
    mean_final = np.mean(final, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    lims = [-3, 3]
    axes[0].scatter(x0[:, 0], x0[:, 1], c='steelblue', alpha=0.6, s=15)
    axes[0].set_title("Initial")
    
    axes[1].scatter(final[:, 0], final[:, 1], c='crimson', alpha=0.8, s=15)
    # Expected radius is sqrt(1) = 1 for this specific Newtonian potential scaling
    circle = plt.Circle((mean_final[0], mean_final[1]), 1.0, color='black', fill=False, linewidth=2, linestyle='--')
    axes[1].add_patch(circle)
    axes[1].set_title("Final (Uniform Disk)")
    
    for ax in axes:
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    run_example_8_circle()