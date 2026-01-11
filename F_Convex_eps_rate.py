import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class ColeHopfSolver:
    """
    A solver for Mean-Field Control problems
    """
    def __init__(self, N=100, epsilon=1e-299, T=1.0, dt=0.001):
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

        # 2. Generate noise for the 'Context' swarm (The Crowd) - 
        # Shape: (N_context, d)
        noise_context = np.random.standard_normal(current_swarm.shape) * np.sqrt(tau)
        future_swarm_proxy = current_swarm + noise_context

        # 3. Flatten for cost evaluation
        flat_XW = X_plus_W.reshape(N*K, d)

        # 4. Compute Costs
        # Pass the 'future_swarm_proxy' as context
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
                if tau < 1e-5:
                    X_path[:, i:, :] = X_t[:, np.newaxis, :]
                    break

                # UPDATED: Removed Z_samples argument (now generated internally)
                target_pos = self.F_function_multidim(tau, X_t, cost_function, current_swarm=X_t)
                
                drift = (target_pos - X_t) / tau
                
                dW = np.random.normal(0, np.sqrt(dt), (self.N, d))
                X_t = X_t + drift * dt + dW
                X_path[:, i, :] = X_t

            # Calculate Mean of the Total Cost Functional
            final_swarm = X_path[:, -1, :]
            total_costs = cost_function(final_swarm, context=final_swarm)
            mean_functional = np.mean(total_costs)
            print(f"Iter {k+1}: Mean Cost Functional = {mean_functional:.4f}")
            X_current = X_path[:, -1, :]

        return X_path

# ==========================================
# Cost Functions (Original + New Spring Swarm)
# ==========================================

def cost_spring_swarm_2d(X, context=None):
    """
    NEW: Spring Swarm (Quadratic Variance Minimization).
    Functional: G(mu) = 0.5 * integral |x-y|^2 dmu(x)dmu(y)
    """
    if context is None: return np.zeros(X.shape[0])
    
    # Subsample context for efficiency if large
    if context.shape[0] > 1000:
        idx = np.random.choice(context.shape[0], 1000, replace=False)
        ctx = context[idx]
    else:
        ctx = context

    # Efficient computation using expansion |x-y|^2 = |x|^2 + |y|^2 - 2<x,y>
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    ctx_sq = np.sum(ctx**2, axis=1, keepdims=True).T
    
    # Pairwise distance squared
    dist_sq = X_sq + ctx_sq - 2 * (X @ ctx.T)

    # Return V(x) = 0.5 * Mean(|x-y|^2)
    return 0.5 * np.mean(dist_sq, axis=1)



 

# ==========================================
# Runners
# ==========================================

def run_example_spring_swarm():
    print("\n--- Running Example: Spring Swarm (Variance Minimization) ---")

    # Use epsilon=0.1 to see the Gaussian equilibrium. 
    # 1e-299 is effectively zero viscosity (swarm collapses to a single point).
    solver = ColeHopfSolver(N=400, epsilon=1e-299, T=1.0, dt=0.01)

    # Start with a WIDE initial condition to see the collapse
    x0 = np.random.normal(0, 1.5, (400, 2))

    traj = solver.solve_iterative(d=2, cost_function=cost_spring_swarm_2d, x0_init=x0)
    final = traj[:, -1, :]

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    lims = [-3, 3]

    axes[0].scatter(x0[:, 0], x0[:, 1], c='gray', alpha=0.5, s=15)
    axes[0].set_title("Initial (Wide Cloud)")

    axes[1].scatter(final[:, 0], final[:, 1], c='green', alpha=0.7, s=15)
    axes[1].set_title("Final (Collapsed Gaussian)")

    for ax in axes:
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    plt.show()

def run_epsilon_study():
    print("\n--- Running Epsilon Convergence Study ---")

    # Define range of epsilon values to study
    eps_values = np.linspace(0.005, 0.2, 41)

    N_val = 400  # Fixed particle number
    mean_costs = []
    neg_eps_log_eps = []

    # Use a fixed initial condition
    np.random.seed(42)
    x0_fixed = np.random.normal(0, 0, (N_val, 2))

    for eps in eps_values:
        val_x = -eps * np.log(eps)
        neg_eps_log_eps.append(val_x)

        solver = ColeHopfSolver(N=N_val, epsilon=eps, T=1.0)

        # Manual Simulation Loop to capture Control Cost
        num_steps = int(solver.T / solver.dt) + 1
        time_grid = np.linspace(0, solver.T, num_steps)
        dt = solver.dt

        # REMOVED: Pre-generated Z_samples (now internal to F_function)
        X_t = x0_fixed.copy()
        X_path = np.zeros((solver.N, num_steps, 2))
        X_path[:, 0, :] = X_t

        running_int = np.zeros(solver.N)

        for i in range(1, num_steps):
            t = time_grid[i-1]
            tau = solver.T - t
            
            # Singularity Cutoff
            if tau < 1e-5:
                X_path[:, i:, :] = X_t[:, np.newaxis, :]
                break

            # UPDATED CALL: Removed Z_samples arg
            # Using Spring Swarm Cost for the study
            F_val = solver.F_function_multidim(tau, X_t, cost_spring_swarm_2d, current_swarm=X_t)
            
            drift = (F_val - X_t) / tau

            drift_sq = np.sum(drift**2, axis=1)
            running_int += drift_sq * dt

            dW = np.random.normal(0, np.sqrt(dt), (solver.N, 2))
            X_t = X_t + drift * dt + dW
            X_path[:, i, :] = X_t

        traj = X_path

        # Compute Value Error
        final_swarm = traj[:, -1, :]
        terminal_costs = cost_spring_swarm_2d(final_swarm, context=final_swarm)
        mean_terminal = np.mean(terminal_costs)
        mean_control = np.mean( (eps / 2.0) * running_int )

        value_error = mean_terminal + mean_control

        mean_costs.append(value_error)
        print(f"eps={eps:.4f}, -eps*ln(eps)={val_x:.4f} -> Value Error={value_error:.5f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mean_costs_arr = np.array(mean_costs)
    neg_eps_log_eps_arr = np.array(neg_eps_log_eps)

    # Plot 1 (Logarithmic Term)
    axes[0].plot(neg_eps_log_eps, mean_costs, 'o-', color='blue', label='Simulation Data')
    slope2, intercept2 = np.polyfit(neg_eps_log_eps_arr, mean_costs_arr, 1)
    fit_line2 = slope2 * neg_eps_log_eps_arr + intercept2
    residuals2 = mean_costs_arr - fit_line2
    rmse2 = np.sqrt(np.mean(residuals2**2))
    axes[0].plot(neg_eps_log_eps, fit_line2, 'r--', label=f'Fit: y={slope2:.3f}x + {intercept2:.3f}\nRMSE: {rmse2:.5f}')
    axes[0].set_xlabel(r'-$\epsilon \ln(\epsilon)$')
    axes[0].set_ylabel('Mean Final Cost')
    axes[0].set_title(r'Relationship: -$\epsilon \ln(\epsilon)$ vs Mean Cost')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Plot 2 (Linear Term)
    axes[1].plot(eps_values, mean_costs, 'o-', color='green', label='Simulation Data')
    slope1, intercept1 = np.polyfit(eps_values, mean_costs_arr, 1)
    fit_line1 = slope1 * eps_values + intercept1
    residuals1 = mean_costs_arr - fit_line1
    rmse1 = np.sqrt(np.mean(residuals1**2))
    axes[1].plot(eps_values, fit_line1, 'r--', label=f'Fit: y={slope1:.3f}x + {intercept1:.3f}\nRMSE: {rmse1:.5f}')
    axes[1].set_xlabel(r'$\epsilon$')
    axes[1].set_ylabel('Mean Final Cost')
    axes[1].set_title(r'Relationship: $\epsilon$ vs Mean Cost')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # run_example_spring_swarm()
    # 1. Run the Epsilon Study
    run_epsilon_study()