import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class ColeHopfSolver:
    """
    A solver for Mean-Field Control problems
    """
    def __init__(self, N=1000, epsilon=1e-299, T=1.0, dt=0.001):
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
            
            X_path[:, 0, :] = X_current
            X_t = X_current.copy()

            for i in range(1, num_steps):
                t = time_grid[i-1]
                tau = self.T - t
                
                # FIX: Singularity Cutoff
                # As tau -> 0, the drift term (Target - X)/tau explodes.
                if tau < 1e-15:
                    X_path[:, i:, :] = X_t[:, np.newaxis, :]
                    break

                target_pos = self.F_function_multidim(tau, X_t, cost_function, current_swarm=X_t)
                
                drift = (target_pos - X_t) / tau
                
                dW = np.random.normal(0, np.sqrt(dt), (self.N, d))
                X_t = X_t + drift * dt + dW
                X_path[:, i, :] = X_t

            # Stats
            final_swarm = X_path[:, -1, :]
            total_costs = cost_function(final_swarm, context=final_swarm)
            mean_functional = np.mean(total_costs)
            print(f"Iter {k+1}: Mean Final Cost = {mean_functional:.4f}")
            X_current = X_path[:, -1, :]

        return X_path


# ==========================================
# Cost Functions
# ==========================================

# def cost_ring_2d(X, context=None, target_radius=1.5):
    # """
    # 2D Ring (Annulus).
    # V(x) = 0.5 * (|x|^2 - R^2)^2 (Radial Double-Well)
    # W(x-y) = -log|x-y| (Log Repulsion)
    # """
    # if context is None: return np.sum(X**2, axis=1)
    
    # # Heuristic subsampling for context
    # if context.shape[0] > 400:
    #     idx = np.random.choice(context.shape[0], 400, replace=False)
    #     ctx = context[idx]
    # else:
    #     ctx = context
        
    # diff = X[:, np.newaxis, :] - ctx[np.newaxis, :, :]
    # dist_sq = np.sum(diff**2, axis=2)
    # dist_sq = np.maximum(dist_sq, 1e-300)
    # W = -0.5 * np.log(dist_sq)
    
    # r_sq = np.sum(X**2, axis=1)
    # V = 0.5 * (r_sq - target_radius**2)**2
    # return V + np.mean(W, axis=1)

def cost_newtonian_2d(X, context=None):
    if context is None: return 0
    if context.shape[0] > 400:
        idx = np.random.choice(context.shape[0], 400, replace=False)
        ctx = context[idx]
    else:
        ctx = context
        
    diff = X[:, np.newaxis, :] - ctx[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-300)
    # W(z) = 0.5 * |z|^2 - 0.5 ln(|z|^2)
    pairwise_cost = 0.5 * dist_sq - 0.5 * np.log(dist_sq)
    return np.mean(pairwise_cost, axis=1)

# ==========================================
# Studies and Runners
# ==========================================

 
def run_N_study():
    print("\nRunning Example 8: 2D Newtonian Swarm (Circle Law)")
    print("--- Running Convergence Study (1/N vs Value Error) ---")

    N_list = [10, 20, 25, 50, 100, 125, 200, 250, 400, 500, 800, 1000]
    inv_N_values = []
    value_error_values = []
    eps_val = 1e-299

    for n_val in N_list:
        solver_n = ColeHopfSolver(N=n_val, epsilon=eps_val, T=1)

        # Manual loop to capture control cost integral
        num_steps = int(solver_n.T / solver_n.dt) + 1
        time_grid = np.linspace(0, solver_n.T, num_steps)
        dt = solver_n.dt

        # REMOVED: Z_samples generation (now handled inside F_function)
        X_t = np.random.normal(0, 0, (n_val, 2))
        running_int = np.zeros(solver_n.N)

        for i in range(1, num_steps):
            t = time_grid[i-1]
            tau = solver_n.T - t
            
            # Singularity cutoff
            if tau < 1e-5:
                break

            # UPDATED CALL: Removed Z_samples
            F_val = solver_n.F_function_multidim(tau, X_t, cost_newtonian_2d, current_swarm=X_t)
            
            drift = (F_val - X_t) / tau
            drift_sq = np.sum(drift**2, axis=1)
            running_int += drift_sq * dt

            dW = np.random.normal(0, np.sqrt(dt), (solver_n.N, 2))
            X_t = X_t + drift * dt + dW

        final_n = X_t
        costs_n = cost_newtonian_2d(final_n, context=final_n)
        mean_terminal_cost = np.mean(costs_n)
        mean_control_cost = np.mean( (eps_val / 2.0) * running_int )

        value_error = mean_terminal_cost + mean_control_cost - 0.75
        inv_N_values.append(1.0 / n_val)
        value_error_values.append(value_error)
        print(f"N={n_val}, 1/N={1.0/n_val:.4f} -> Value Error={value_error:.5f}")

    # Plotting
    slope, intercept = np.polyfit(inv_N_values, value_error_values, 1)
    fit_line = [slope * x + intercept for x in inv_N_values]
    
    plt.figure(figsize=(8, 6))
    plt.plot(inv_N_values, value_error_values, 'o', label='Simulation Data')
    plt.plot(inv_N_values, fit_line, 'r--', label=f'Fit: y={slope:.4f}x + {intercept:.4f}')
    plt.xlabel('Reciprocal of Particle Number (1/N)')
    plt.ylabel('Value Error')
    plt.title('Linear Relationship: 1/N vs Value Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

         
if __name__ == "__main__":
    # Choose which study to run
    run_N_study()