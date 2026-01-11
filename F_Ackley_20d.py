import numpy as np
import matplotlib.pyplot as plt
import time

# --- 0. Parameters (Multi-Dimensional) ---
dim = 20 # Dimensionality of the space (d) 
num_paths = 1000   # Number of independent SDE paths (N)
num_steps = 2001   # Number of time steps (including t=0)
T_max = 1.0
num_iterations = 10
error_variance = 9e-10 # Target metric tolerance
dt = T_max / (num_steps - 1)
time_grid = np.linspace(0, T_max, num_steps)
lambda_val = 0.75  # MEAN-FIELD COUPLING WEIGHT Lambda

initial_guess = 5  # Initial guess for all dimensions

 
# CRITICAL STABILITY PARAMETERS
epsilon = 1e-300  # Adjusted epsilon for stability
K_samples = 1000 # Monte Carlo sample size (S)

# Pre-generate Z_samples once outside the loop for efficiency
# Z_samples = np.random.standard_normal((num_paths, K_samples, dim))

# --- 1. Define g(x) and Core Functions ---

def g(x):
     """The Multi-Dimensional Ackley potential function g(x)."""
     x_sq_sum = np.sum(x**2, axis=-1)
     x_cos_sum = np.sum(np.cos(2 * np.pi * x), axis=-1)

     term1 = -20 * np.exp(-0.2 * np.sqrt(x_sq_sum / dim))
     term2 = -np.exp(x_cos_sum / dim)

     return 20 + np.e + term1 + term2

 

def F_function_multidim(tau, x_array, epsilon, Z_samples):
    """
    Computes F(tau, x) (HJB-based term: G + x). Includes the stability trick.
    """
    W_samples = Z_samples * np.sqrt(tau)
    X_plus_W = x_array[:, np.newaxis, :] + W_samples
    g_X_plus_W = g(X_plus_W)

    # Stability trick
    min_g_per_row = np.min(g_X_plus_W, axis=1, keepdims=True)
    exponent_arg = -(g_X_plus_W - min_g_per_row) / epsilon
    exponent_scalar = np.exp(exponent_arg)

    # Numerator E[e^{-g(x+W)} * (x+W)]
    numerator_matrix = exponent_scalar[:, :, np.newaxis] * X_plus_W
    numerator = np.mean(numerator_matrix, axis=1)

    # Denominator E[e^{-g(x+W)}]
    denominator = np.mean(exponent_scalar, axis=1)

    return numerator / (denominator[:, np.newaxis] + 1e-3000)



# --- 2. Iterative SDE Simulation Loop -----------------------------------------------

# Initial State Definition (Constant 5 for k=1)
constant_start_value = initial_guess
initial_X = np.full((num_paths, dim), constant_start_value)
mean_g_initial = np.mean(g(initial_X)) +1e10# Reference cost value

X_current = np.zeros((num_paths, num_steps, dim))
X_current[:, 0, :] = initial_X # Set the starting batch for k=1

results = []
k = 0
total_metric = 1

for k in range(num_iterations):
    start_time = time.time()
    print(f"\n--- Starting Iteration {k+1}/{num_iterations} ---")

    Z_samples = np.random.standard_normal((num_paths, K_samples, dim))

    if k > 0:
        # Particle Recycling: X(0) for iteration k is X(T) from iteration k-1
        X_current[:, 0, :] = lambda_val * mean_XT + (1 - lambda_val) * X_T

    # SDE Integration (Euler-Maruyama)
    for i in range(1, num_steps):
        t = time_grid[i-1]
        tau = T_max - t
        X_prev = X_current[:, i - 1, :]

        if tau <= dt / 2:
            dW_X = np.random.normal(0, np.sqrt(dt), size=(num_paths, dim))
            X_current[:, i, :] = X_prev + dW_X
            break

        # 1. HJB/F-term calculation (Num_paths x Dim)
        F_values = F_function_multidim(tau, X_prev, epsilon, Z_samples)

        numerator = F_values - X_prev

        # Euler-Maruyama Step: b = Numerator / tau
        drift_term = (numerator / tau) * dt

        dW_X = np.random.normal(0, np.sqrt(dt), size=(num_paths, dim))

        X_current[:, i, :] = X_prev + drift_term + dW_X

    # Final positions
    X_T = X_current[:, num_steps - 1, :]

    # --- Check Convergence ---
    mean_XT = np.mean(X_T, axis=0)
    var_XT = np.var(X_T, axis=0)
    max_var = np.max(var_XT)

    g_T = g(X_T)
    g_meanX = g(mean_XT)
    mean_g_diff = np.abs(g_meanX - mean_g_initial)

    mean_g_new = np.mean(g_T)
    # if g_meanX < mean_g_initial:
    # X_current[:, 0, :] = lambda_val * mean_XT + (1 - lambda_val) * X_T
    mean_g_initial = g_meanX
    cost_variance = np.var(g_T)
    total_metric = max_var + mean_g_diff + cost_variance
    elapsed_time = time.time() - start_time
    print(f"Iteration {k+1}: Time: {elapsed_time:.2f}s, Max Var: {max_var:.4e}, Cost Diff: {mean_g_diff:.4e}, Metric: {total_metric:.4e}")
    print(f" g at meanX: {g_meanX}")
    print(f"meanX: {mean_XT}")


    results.append({'k': k + 1, 'mean': mean_XT, 'variance_vec': var_XT, 'max_var': max_var, 'mean_g_diff': mean_g_diff})

    # Stability checks
    if max_var > 1e10:
        print(f"Variance exploded. Stopping at iteration {k+1}.")
        break

    if total_metric < error_variance:
        print(f"Convergence condition met. Stopping at iteration {k+1}.")
        break



# Final path array for plotting
X_final = X_current.copy()

# --- 3. Final Results ---
print(f"\n--- FINAL RESULTS (Iteration {k+1}) ---")
print(f"Dimension (d): {dim}")
print(f"Initial State X(0) for k=1: {constant_start_value} (constant)")
print(f"Final Max Variance: {np.var(X_current[:, -1, :], axis=0)}")
print(f"Final Mean vector E[X({T_max})]: {np.mean(X_current[:, -1, :], axis=0)}")
print(f" g at meanX: {g_meanX}")



# --- Plotting for dim = 1 or 2 ---
if dim == 1:
    # Trajectories (subset)
    plt.figure(figsize=(10,5))
    nplot = min(50, num_paths)
    idx = np.random.choice(num_paths, nplot, replace=False)
    for j in idx:
        plt.plot(time_grid, X_final[j, :, 0], color='red', alpha=0.25)
    # Mean trajectory
    mean_traj = np.mean(X_final[:, :, 0], axis=0)
    plt.plot(time_grid, mean_traj, color='black', lw=2, label='Mean')
    plt.title('Trajectories (dim=1)')
    plt.xlabel('Time')
    plt.ylabel('X(t)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()