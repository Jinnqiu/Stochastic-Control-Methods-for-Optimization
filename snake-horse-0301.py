import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os
import torch
import torch.nn.functional as F
import time

# ==========================================
# 1. Image Processing (CPU)
# ==========================================

def get_points_from_image(filename, num_points=30000, scale=3.0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    print(f"Loading image: {full_path}")

    try:
        img = Image.open(full_path).convert('L')
    except FileNotFoundError:
        print(f"ERROR: Could not find '{filename}'. Using random blob.")
        return np.random.normal(0, 0.00000001, (num_points, 2)).astype(np.float32)

    img.thumbnail((1024, 1024)) 
    w, h = img.size
    data = np.array(img)
    y_idxs, x_idxs = np.where(data < 128)
    
    if len(x_idxs) == 0:
        return np.random.normal(0, 0.00000001, (num_points, 2)).astype(np.float32)

    all_points = np.column_stack([x_idxs, y_idxs])
    
    if len(all_points) > num_points:
        indices = np.random.choice(len(all_points), num_points, replace=False)
        sampled_points = all_points[indices]
    else:
        indices = np.random.choice(len(all_points), num_points, replace=True)
        sampled_points = all_points[indices]

    sampled_points = sampled_points.astype(np.float32) # Float32 is better for GPU

    # Normalize
    sampled_points[:, 1] = h - sampled_points[:, 1]
    sampled_points[:, 0] -= w / 2
    sampled_points[:, 1] -= h / 2
    max_dim = max(w, h)
    sampled_points[:, 0] *= (scale / max_dim)
    sampled_points[:, 1] *= (scale / max_dim)
    sampled_points += np.random.normal(0, 0.00000001, sampled_points.shape).astype(np.float32)
    
    return sampled_points

# ==========================================
# 2. Cole-Hopf Solver (GPU Accelerated)
# ==========================================

class ColeHopfSolverGPU:
    def __init__(self, N=100, epsilon=1e-15, T=1.0, dt=0.0001, device='cpu'):
        self.N = N
        self.epsilon = epsilon
        self.T = T
        self.dt = dt
        self.K_samples = 100  # 100 is sufficient with N=100k
        self.denom_floor = 1e-18
        self.device = device

    def F_function_multidim(self, tau, x_params, cost_func, current_swarm):
        # x_params: (N, d)
        N, d = x_params.shape
        K = self.K_samples

        # 1. Project particles forward with noise
        # Z: (N, K, d)
        Z = torch.randn(N, K, d, device=self.device)
        # Expansion: x_params needs to be (N, 1, d) to broadcast against (N, K, d)
        Y_samples = x_params.unsqueeze(1) + Z * np.sqrt(tau)
        
        # 2. Project context (crowd) forward roughly
        noise_context = torch.randn_like(current_swarm) * np.sqrt(tau)
        future_swarm_proxy = current_swarm + noise_context
        
        # 3. Evaluate Cost
        # Flat Y: (N*K, d)
        flat_Y = Y_samples.view(N*K, d)
        
        # g_vals: (N*K) -> reshape to (N, K)
        g_vals = cost_func(flat_Y, context=future_swarm_proxy).view(N, K)

        # 4. Soft-Min (Log-Sum-Exp Trick)
        min_g, _ = torch.min(g_vals, dim=1, keepdim=True)
        weights = torch.exp(-(g_vals - min_g) / self.epsilon)

        # 5. Weighted Average
        # weights: (N, K) -> (N, K, 1)
        # Y_samples: (N, K, d)
        numerator = torch.sum(weights.unsqueeze(2) * Y_samples, dim=1)
        denominator = torch.sum(weights, dim=1)
        
        return numerator / (denominator.unsqueeze(1) + self.denom_floor)

    def solve(self, x0_np, cost_function):
        # Convert initial state to Tensor
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        
        num_steps = int(self.T / self.dt) + 1
        d = x0.shape[1]
        
        # Store path in CPU RAM to save GPU memory, or keep on GPU if small enough
        # We'll keep on CPU for safety
        X_path = torch.zeros((self.N, num_steps, d), device='cpu')
        
        X_current = x0.clone()
        X_path[:, 0, :] = X_current.cpu()

        print(f"--- Simulating on {self.device}: N={self.N}, Steps={num_steps} ---")
        start_time = time.time()

        for i in range(1, num_steps):
            t = (i-1) * self.dt
            tau = self.T - t
            
            if tau < 1e-15:
                drift = 0
            else:
                target_pos = self.F_function_multidim(tau, X_current, cost_function, X_current)
                drift = (target_pos - X_current) / tau

            dW = torch.randn(self.N, d, device=self.device) * np.sqrt(self.dt)
            X_current = X_current + drift * self.dt + dW
            
            # Save to history (move to CPU)
            X_path[:, i, :] = X_current.cpu()
            
            if i % 50 == 0:
                print(f"Step {i}/{num_steps}...", end='\r')
        
        elapsed = time.time() - start_time
        print(f"\nSimulation Done. Time: {elapsed:.2f}s")
        return X_path.numpy()

# ==========================================
# 3. Cost Function (GPU)
# ==========================================

class ImageShapeCostGPU:
    def __init__(self, target_points, device='cpu'):
        # Just store the points themselves (or a subset)
        if isinstance(target_points, np.ndarray):
            self.target_points = torch.tensor(target_points, dtype=torch.float32, device=device)
        else:
            self.target_points = target_points  # Assume it's already a tensor
        self.device = device

    def __call__(self, X, context=None):
        # X shape: (Batch_Size, d) where Batch_Size = N * K (very large)
        # Target shape: (M, d) where M is ~5000 (subsampled target)
        
        batch_size = X.shape[0]
        # Chunk Size 
        
        chunk_size = 40000 
        
        min_dists_sq_list = []
        
        # Check if target points are too large (e.g., > 20k)
        if self.target_points.shape[0] > 20000:
             # For HUGE target (e.g. 500k), we must compute distance in blocks
             # to avoid OOM.
             target_chunk_size = 5000
             
             for i in range(0, batch_size, chunk_size):
                X_chunk = X[i : i+chunk_size]
                
                # Compute min dist to ALL targets by iterating target blocks
                current_min_sq = torch.full((X_chunk.shape[0],), float('inf'), device=self.device)
                
                for j in range(0, self.target_points.shape[0], target_chunk_size):
                    t_chunk = self.target_points[j : j+target_chunk_size]
                    # (Chunk_X, Chunk_Target)
                    dists = torch.cdist(X_chunk, t_chunk)
                    min_vals, _ = torch.min(dists, dim=1) # (Chunk_X)
                    current_min_sq = torch.min(current_min_sq, min_vals.pow(2))
                    
                min_dists_sq_list.append(current_min_sq)
                
        else:
            # Original logic for small targets is faster (vectorized)
            for i in range(0, batch_size, chunk_size):
                X_chunk = X[i : i+chunk_size]
                dists = torch.cdist(X_chunk, self.target_points)
                min_vals, _ = torch.min(dists, dim=1)
                min_dists_sq_list.append(min_vals.pow(2))
            
        V = torch.cat(min_dists_sq_list)

        # 2. Interaction: Log Repulsion (Subsampled)
        W = 0
        if context is not None:
            # VERY Aggressive subsampling for interaction speed
            # We don't need perfect repulsion, just enough to prevent collapse
            if context.shape[0] > 1000:
                idx = torch.randperm(context.shape[0], device=self.device)[:100]
                ctx_subset = context[idx]
            else:
                ctx_subset = context
                
            
            w_list = []
            for i in range(0, batch_size, chunk_size):
                 X_chunk = X[i : i+chunk_size]
                 diff_sq = torch.cdist(X_chunk, ctx_subset, p=2).pow(2)
                 w_val = -0.015 * torch.sum(torch.log(diff_sq + 1e-16), dim=1)
                 w_list.append(w_val)
                 
            W = torch.cat(w_list)

        return V + 0.0000001 * W

# ==========================================
# 4. Animation
# ==========================================

def create_animation(traj, target_points, filename="gpu_image_morph.gif", output_dir=None):
    print(f"Generating Animation -> {filename}")
    
    # --- Subsampling & Pacing ---
    total_steps = traj.shape[1]
    target_active_frames = 150 # Frames for the movement itself
    stride = max(1, total_steps // target_active_frames)
    
    # Create indices for: Start Pause + Movement + End Pause
    pause_frames = 30
    
    start_indices = np.zeros(pause_frames, dtype=int)
    active_indices = np.arange(0, total_steps, stride)
    end_indices = np.full(pause_frames, total_steps - 1, dtype=int)
    
    combined_indices = np.concatenate([start_indices, active_indices, end_indices])
    
    # Apply fancy indexing (ensure integer indices)
    traj_subset = traj[:, combined_indices, :]
    frames_to_animate = traj_subset.shape[1]
    
    print(f"Animation Pacing: {pause_frames} start + {len(active_indices)} move + {pause_frames} end = {frames_to_animate} frames")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.axis('off')

    scat = ax.scatter([], [], c='blue', s=0.1, alpha=0.5)
    
    def update(frame_idx):
        scat.set_offsets(traj_subset[:, frame_idx, :])
        return scat,

    anim = animation.FuncAnimation(fig, update, frames=frames_to_animate, interval=40)
    
    try:
        # Saving
        if output_dir:
            save_path = os.path.join(output_dir, filename)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, filename)

        anim.save(save_path, writer='pillow', fps=25)
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.close(fig)

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    # Settings
    N_particles = 100000  # 

    # Output setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results_H0301")
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Saving results to: {output_dir} ---")
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using Device: {device} ---")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    print("--- 1. Loading Images ---")
    
    # Simulation Resolution (Agents)
    x0 = get_points_from_image("snake_o.png", num_points=N_particles, scale=3.5)
    
    plt.figure(figsize=(6,6))
    plt.scatter(x0[:,0], x0[:,1], c='blue', s=0.1, alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Initial Shape: Snake")
    plt.savefig(os.path.join(output_dir, "gpu_initial_shape_snake.png")) 

    # Ground Truth Resolution (Target)
    print("--- Loading High-Res Target for Fidelity ---")
    target_cloud_full = get_points_from_image("horse.png", num_points=12000, scale=3.5)
    
    # For the main simulation, we can just take a subset or use the full one if needed.
    # We'll use a subset for the "Scatter" visualization to avoid slow plotting.
    target_cloud_display = target_cloud_full[:N_particles] 
    
    plt.figure(figsize=(6,6))
    plt.scatter(target_cloud_display[:,0], target_cloud_display[:,1], c='blue', s=0.1, alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Target Shape: Horse")
    plt.savefig(os.path.join(output_dir, "gpu_target_shape_horse.png"))

    print("--- 2. Solving Control Problem ---")
    
    # Initialize GPU-aware Cost and Solver
    # Cost & Logic Note: 
    # Use 1e-15, which is small enough to act as a "Hard Min"
    
    cost_targets = target_cloud_full
    # Old Way: Pairwise distance to cloud
    cost = ImageShapeCostGPU(cost_targets, device=device)
    
    solver = ColeHopfSolverGPU(N=N_particles, epsilon=1e-15, T=1.0, dt=0.0001, device=device)
    
    traj = solver.solve(x0, cost)
    
    print("--- 3. Animating ---")
    create_animation(traj, target_cloud_display, filename="gpu_image_morph.gif", output_dir=output_dir)
    
    plt.figure(figsize=(6,6))
    plt.scatter(traj[:, -1, 0], traj[:, -1, 1], c='blue', s=0.1, alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Final Shape After Control: Horse")
    plt.savefig(os.path.join(output_dir, "gpu_final_shape_horse.png"))
    #save images when t=0, t=0.25, t=0.5, t=0.75, 0.9, 0.95, 0.995,  t=1.0
    time_points = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.995, 1.0]
    for t in time_points:
        idx = int(t * (traj.shape[1] - 1))
        plt.figure(figsize=(6,6))
        plt.scatter(traj[:, idx, 0], traj[:, idx, 1], c='blue', s=0.1, alpha=0.5)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Shape at Time {t}")
        plt.savefig(os.path.join(output_dir, f"gpu_shape_at_time_{t}.png"))
        plt.close() 
    #refine the final shape by Iter_num iterations with previous step final with smaller dt to capture more details
    Iter_num = 4
    refined_traj = traj.copy()
    for iter in range(Iter_num):
        print(f"Refinement Iteration {iter+1}/{Iter_num}...")
        # Use the last frame as the new initial condition
        x0_refine = refined_traj[:, -1, :]
        # Reduce dt for finer convergence
        solver_refine = ColeHopfSolverGPU(N=N_particles, epsilon=1e-15, T=0.001, dt=0.000001, device=device)
        refined_traj_iter = solver_refine.solve(x0_refine, cost)
        
        # Update the refined trajectory (only keep the final frame of this iteration)
        refined_traj[:, -1, :] = refined_traj_iter[:, -1, :]    
        # save intermediate refined shape after each iteration
        plt.figure(figsize=(6,6))
        plt.scatter(refined_traj[:, -1, 0], refined_traj[:, -1, 1], c='blue', s=0.1, alpha=0.5)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Refined Shape After Iteration {iter+1}")
        plt.savefig(os.path.join(output_dir, f"gpu_refined_shape_iter_{iter+1}.png"))   
        plt.close() 

    # Save the final refined shape
    plt.figure(figsize=(6,6))
    plt.scatter(refined_traj[:, -1, 0], refined_traj[:, -1, 1], c='blue', s=0.1, alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Final Refined Shape After Control: Horse")
    plt.savefig(os.path.join(output_dir, "gpu_final_refined_shape_horse.png"))  
