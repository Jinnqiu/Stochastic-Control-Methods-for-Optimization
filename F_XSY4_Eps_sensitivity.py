import numpy as np
import matplotlib.pyplot as plt
import time

# --- 0. Parameters (Multi-Dimensional) ---
dim = 1 # Dimensionality of the space (d) work until 100 with g_ mean .65 if num_path=400, it may reduced to .40
num_paths = 1000    # Number of independent SDE paths (N)
num_steps = 4001   # Number of time steps (including t=0)
T_max = 1.0
num_iterations = 1
error_variance = 9e-10 # Target metric tolerance
dt = T_max / (num_steps - 1)
time_grid = np.linspace(0, T_max, num_steps)
lambda_val = 0.5  # MEAN-FIELD COUPLING WEIGHT Lambda

initial_guess = 1  # Initial guess for all dimensions


# CRITICAL STABILITY PARAMETERS
epsilon = 3e-1  # Adjusted epsilon for stability
K_samples = 800 # Monte Carlo sample size (K)

# Pre-generate Z_samples once outside the loop for efficiency
# Z_samples = np.random.standard_normal((num_paths, K_samples, dim))

# --- 1. Define g(x) and Core Functions ---


# add the benchmark functions here Rosenbrock, Ackley, Griewank, Rastrigin, Salomon, Schwefel, Levy, Dixon-Price, Michalewicz, Sphere in the following order



# def g(x):
#      """The Multi-Dimensional Ackley potential function g(x)."""
#      x_sq_sum = np.sum(x**2, axis=-1)
#      x_cos_sum = np.sum(np.cos(2 * np.pi * x), axis=-1)

#      term1 = -20 * np.exp(-0.2 * np.sqrt(x_sq_sum / dim))
#      term2 = -np.exp(x_cos_sum / dim)

#      return 20 + np.e + term1 + term2


# def g(x):
#     """The Multi-Dimensional Griewank function g(x)."""

#     # Compute sum of squares term
#     x_sq_sum = np.sum(x**2, axis=-1)

#     # Compute product of cosines term
#     indices = np.arange(1, dim + 1)
#     x_cos_prod = np.prod(np.cos(x / np.sqrt(indices)), axis=-1)

#     # Griewank function formula
#     return 1 + (x_sq_sum / 4000) - x_cos_prod

 

def g(x):
    """The Multi-Dimensional Xin-She Yang 4 function g(x)."""
    # Global minimum at (1,1,...,1) with g(x)=0
    # $$f(\mathbf{x}) = \left[ \sum_{i=1}^{d} \sin^2(x_i) - \exp\left( -\sum_{i=1}^{d} x_i^2 \right) \right] \cdot \exp\left( -\sum_{i=1}^{d} \sin^2\sqrt{|x_i|} \right)$$
    sin_sq_sum = np.sum(np.sin(x)**2, axis=-1)
    exp_neg_sq_sum = np.exp(-np.sum(x**2, axis=-1))
    exp_neg_sin_sqrt_sum = np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))**2, axis=-1))
    return (sin_sq_sum - exp_neg_sq_sum) * exp_neg_sin_sqrt_sum


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
        X_current[:, 0, :] = mean_XT

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
    print(f"mean g new: {mean_g_new}")

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
print(f" g value: {mean_g_initial}")
print(f"Final mean g value: {mean_g_new}")


# --- 4. Epsilon Sensitivity Study ---
print("\n--- Starting Epsilon Sensitivity Study ---")

# Define range of epsilons to test (avoid 0 to prevent log(0))
# We choose a range around the original epsilon used
epsilon_list = np.linspace(0.0005, 0.12, 21) # Adjusted range for faster execution and to start from a lower value
results_eps = []
x_axis_vals = [] # -epsilon * ln(epsilon)

print(f"Testing epsilons: {epsilon_list}")

for eps_val in epsilon_list:
    # Reset state for this epsilon run
    X_study = np.full((num_paths, dim), constant_start_value)
    X_study_batch = np.zeros((num_paths, num_steps, dim))
    X_study_batch[:, 0, :] = X_study

    mean_XT_study = None

    # Accumulator for the integral term
    integral_sum = np.zeros(num_paths)

    # Run the iterative process (using the same num_iterations as main block)
    for k_sub in range(num_iterations):
        Z_samples_study = np.random.standard_normal((num_paths, K_samples, dim))

        # Reset integral sum for the current iteration path
        integral_sum = np.zeros(num_paths)

        if k_sub > 0 and mean_XT_study is not None:
             X_study_batch[:, 0, :] = mean_XT_study

        for i in range(1, num_steps):
            t = time_grid[i-1]
            tau = T_max - t
            X_prev = X_study_batch[:, i - 1, :]

            if tau <= dt / 2:
                dW_X = np.random.normal(0, np.sqrt(dt), size=(num_paths, dim))
                X_study_batch[:, i, :] = X_prev + dW_X
                break

            # Calculate F with current eps_val
            F_vals = F_function_multidim(tau, X_prev, eps_val, Z_samples_study)

            numerator_term = F_vals - X_prev

            # Calculate integral term: |numerator|^2 / (1-t)^2
            # numerator_term is (num_paths, dim), sum squares over dim
            drift_norm_sq = np.sum(numerator_term**2, axis=1)
            integral_sum += (drift_norm_sq / (tau**2)) * dt

            drift = (numerator_term / tau) * dt
            dW = np.random.normal(0, np.sqrt(dt), size=(num_paths, dim))

            X_study_batch[:, i, :] = X_prev + drift + dW

        # End of time steps for this iteration
        X_T_study = X_study_batch[:, num_steps - 1, :]
        mean_XT_study = np.mean(X_T_study, axis=0)

    # Calculate final metric for this epsilon
    g_T_study = g(X_T_study)
    final_mean_g_val = np.mean(g_T_study)

    # Compute Value_error
    mean_integral = np.mean(integral_sum)
    Value_error = final_mean_g_val + (eps_val / 2) * mean_integral + 1

    results_eps.append(Value_error)
    x_val = -eps_val * np.log(eps_val)
    x_axis_vals.append(x_val)
    print(f"  eps: {eps_val:.3f} | -eps*ln(eps): {x_val:.3f} | Value_error: {Value_error:.4f}")

# Convert to numpy arrays
x_axis_vals = np.array(x_axis_vals)
results_eps = np.array(results_eps)

# Linear Regression: y = m*x + c
A = np.vstack([x_axis_vals, np.ones(len(x_axis_vals))]).T
m, c = np.linalg.lstsq(A, results_eps, rcond=None)[0]

# Compute RMSE for the first regression
y_fit_1 = m * x_axis_vals + c
rmse_1 = np.sqrt(np.mean((results_eps - y_fit_1)**2))

print(f"\nLinear Fit Result 1 (-eps*ln(eps)): Value_error = {m:.4f} * x + {c:.4f}")
print(f"RMSE 1: {rmse_1:.6e}")

# Plotting
plt.figure(figsize=(12, 5))

# Plot 1: Value Error vs eps * ln(eps)
plt.subplot(1, 2, 1)
plt.scatter(x_axis_vals, results_eps, color='blue', label='Simulation Data')
plt.plot(x_axis_vals, y_fit_1, 'r--', label=f'Fit: {m:.2f}x + {c:.2f}\nRMSE: {rmse_1:.2e}')
plt.title(r'Value Error vs $-\epsilon \ln(\epsilon)$')
plt.xlabel(r'$-\epsilon \ln(\epsilon)$')
plt.ylabel(r'Value Error')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Linear Regression for Plot 2: Value Error vs epsilon
B = np.vstack([epsilon_list, np.ones(len(epsilon_list))]).T
m_eps, c_eps = np.linalg.lstsq(B, results_eps, rcond=None)[0]

# Compute RMSE for the second regression
y_fit_2 = m_eps * epsilon_list + c_eps
rmse_2 = np.sqrt(np.mean((results_eps - y_fit_2)**2))

print(f"\nLinear Fit Result 2 (epsilon): Value_error = {m_eps:.4f} * eps + {c_eps:.4f}")
print(f"RMSE 2: {rmse_2:.6e}")

# Plot 2: Value Error vs epsilon
plt.subplot(1, 2, 2)
plt.scatter(epsilon_list, results_eps, color='green', label='Simulation Data')
plt.plot(epsilon_list, y_fit_2, 'r--', label=f'Fit: {m_eps:.2f}x + {c_eps:.2f}\nRMSE: {rmse_2:.2e}')
plt.title(r'Value Error vs $\epsilon$')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Value Error')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()