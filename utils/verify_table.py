"""
Verify OU score table against analytical formulas.
Run this on your machine where PyTorch is installed.

Usage:
    python verify_ou_table.py /path/to/score_table.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python verify_ou_table.py <path_to_score_table.pt>")
    sys.exit(1)

table_path = sys.argv[1]

print("=" * 80)
print("LOADING SCORE TABLE")
print("=" * 80)
data = torch.load(table_path, map_location='cpu')

# Extract config
config = data['config']
alpha = config['alpha']
c_alpha = config['c_alpha']
c_0 = config['c_0']
sigma = config['sigma']
T = config['T']

print(f"Process parameters:")
print(f"  alpha = {alpha}")
print(f"  c_alpha = {c_alpha}")
print(f"  c_0 = {c_0}")
print(f"  sigma = {sigma}")
print(f"  T = {T}")
print()

# Verify it's OU
assert alpha == 1.0 and c_alpha == 0.0, "Not a simple OU process!"
a = c_0  # Drift coefficient
print(f"✓ This is OU with drift coefficient a = {a}")
print(f"  SDE: dX_t = -{a} X_t dt + {sigma} dW_t")
print()

# Extract arrays
S_table = data['S'].numpy()
x0_grid = data['x0_grid'].numpy()
x_grid = data['x_grid'].numpy()
t_grid = data['t_grid'].numpy()

has_density = 'p_table' in data
if has_density:
    p_table = data['p_table'].numpy()
    print("✓ Density table found")
else:
    print("✗ Density table not found")
    
N_x0, N_t, N_x = S_table.shape
print(f"\nGrid: {N_x0} × {N_t} × {N_x}")
print(f"  x0 ∈ [{x0_grid[0]:.2f}, {x0_grid[-1]:.2f}]")
print(f"  x ∈ [{x_grid[0]:.2f}, {x_grid[-1]:.2f}]")
print(f"  t ∈ [{t_grid[0]:.2f}, {t_grid[-1]:.2f}]")
print()

# Analytical formulas
def ou_mean(x0, t):
    """Mean of X_t | X_0 = x0"""
    return x0 * np.exp(-a * t)

def ou_var(t):
    """Variance of X_t | X_0 = x0"""
    return (sigma**2 / (2*a)) * (1 - np.exp(-2*a*t))

def ou_density(x, x0, t):
    """Analytical density p(x,t|x0)"""
    mean = ou_mean(x0, t)
    var = ou_var(t)
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean)**2 / (2 * var))

def ou_score(x, x0, t):
    """Analytical score ∂_x log p(x,t|x0)"""
    mean = ou_mean(x0, t)
    var = ou_var(t)
    return -(x - mean) / var

# Stationary distribution
var_stationary = sigma**2 / (2*a)
std_stationary = np.sqrt(var_stationary)
print(f"Stationary distribution: N(0, {var_stationary:.3f})")
print(f"  std = {std_stationary:.3f}")
print(f"  ±1 data is within {1/std_stationary:.2f}σ")
print()

# At final time T
mean_decay = np.exp(-a * T)
var_at_T = ou_var(T)
print(f"At T = {T}:")
print(f"  Mean decay: {mean_decay:.3f} (retains {mean_decay*100:.1f}% of x0)")
print(f"  Variance: {var_at_T:.3f} ({var_at_T/var_stationary*100:.1f}% of stationary)")
print()

# Compute analytical scores
print("=" * 80)
print("COMPUTING ANALYTICAL SCORES")
print("=" * 80)

S_analytical = np.zeros_like(S_table)
for i in range(N_x0):
    if (i+1) % 5 == 0:
        print(f"  Progress: {i+1}/{N_x0}")
    for j in range(N_t):
        for k in range(N_x):
            S_analytical[i, j, k] = ou_score(x_grid[k], x0_grid[i], t_grid[j])

print("Done.\n")

# Compare scores
print("=" * 80)
print("SCORE COMPARISON")
print("=" * 80)

diff = S_table - S_analytical
abs_diff = np.abs(diff)

print(f"Max absolute error: {abs_diff.max():.6e}")
print(f"Mean absolute error: {abs_diff.mean():.6e}")
print(f"Median absolute error: {np.median(abs_diff):.6e}")
print(f"95th percentile: {np.percentile(abs_diff, 95):.6e}")
print()

# Relative error
mask = np.abs(S_analytical) > 0.1
if mask.sum() > 0:
    rel_error = np.abs((S_table[mask] - S_analytical[mask]) / S_analytical[mask])
    print(f"Relative error (where |S_analytical| > 0.1):")
    print(f"  Mean: {rel_error.mean():.2%}")
    print(f"  Median: {np.median(rel_error):.2%}")
    print(f"  95th percentile: {np.percentile(rel_error, 95):.2%}")
    print()

# Sample test points
print("SAMPLE POINT COMPARISONS")
print("-" * 80)
test_points = [
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.741),
    (1.0, 1.0, 0.0),
    (-1.0, 2.0, -0.5),
    (0.5, T, 0.0),
]

for x0_val, t_val, x_val in test_points:
    x0_idx = np.argmin(np.abs(x0_grid - x0_val))
    t_idx = np.argmin(np.abs(t_grid - t_val))
    x_idx = np.argmin(np.abs(x_grid - x_val))
    
    x0_actual = x0_grid[x0_idx]
    t_actual = t_grid[t_idx]
    x_actual = x_grid[x_idx]
    
    s_table = S_table[x0_idx, t_idx, x_idx]
    s_analytical = ou_score(x_actual, x0_actual, t_actual)
    
    print(f"(x0={x0_actual:.2f}, t={t_actual:.2f}, x={x_actual:.2f})")
    print(f"  Table:      {s_table:+.6f}")
    print(f"  Analytical: {s_analytical:+.6f}")
    print(f"  Error:      {abs(s_table - s_analytical):.6e}")
    print()

# Density comparison if available
if has_density:
    print("=" * 80)
    print("DENSITY COMPARISON")
    print("=" * 80)
    
    # Compute analytical density
    print("Computing analytical density...")
    p_analytical = np.zeros_like(p_table)
    for i in range(N_x0):
        if (i+1) % 5 == 0:
            print(f"  Progress: {i+1}/{N_x0}")
        for j in range(N_t):
            for k in range(N_x):
                p_analytical[i, j, k] = ou_density(x_grid[k], x0_grid[i], t_grid[j])
    
    print()
    diff_p = p_table - p_analytical
    abs_diff_p = np.abs(diff_p)
    
    print(f"Max absolute error: {abs_diff_p.max():.6e}")
    print(f"Mean absolute error: {abs_diff_p.mean():.6e}")
    print(f"Median absolute error: {np.median(abs_diff_p):.6e}")
    print()
    
    # Relative error
    mask_p = p_analytical > 1e-6
    if mask_p.sum() > 0:
        rel_error_p = np.abs((p_table[mask_p] - p_analytical[mask_p]) / p_analytical[mask_p])
        print(f"Relative error (where p > 1e-6):")
        print(f"  Mean: {rel_error_p.mean():.2%}")
        print(f"  Median: {np.median(rel_error_p):.2%}")
        print(f"  95th percentile: {np.percentile(rel_error_p, 95):.2%}")
        print()
    
    # Normalization
    dx = x_grid[1] - x_grid[0]
    masses = p_table.sum(axis=2) * dx
    print(f"Normalization (∫ p dx):")
    print(f"  Min: {masses.min():.6f}")
    print(f"  Max: {masses.max():.6f}")
    print(f"  Mean: {masses.mean():.6f}")
    print(f"  Std: {masses.std():.6f}")
    if 0.99 < masses.min() and masses.max() < 1.01:
        print("  ✓ Well normalized")
    else:
        print("  ✗ Normalization issues")
    print()
    
    # Log-density
    print("Log-density comparison...")
    eps = 1e-12
    log_p_table = np.log(p_table + eps)
    log_p_analytical = np.log(p_analytical + eps)
    
    mask_log = p_analytical > 1e-4
    if mask_log.sum() > 0:
        diff_log = np.abs(log_p_table[mask_log] - log_p_analytical[mask_log])
        print(f"Log-density error (where p > 1e-4):")
        print(f"  Max: {diff_log.max():.6e}")
        print(f"  Mean: {diff_log.mean():.6e}")
        print(f"  Median: {np.median(diff_log):.6e}")
        print()

# Create visualizations
print("=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Choose test points
t_vals = [0.5, 1.0, 2.0, T]
x0_vals = [0.0, 0.5, 1.0]

# Score comparison
fig, axes = plt.subplots(len(t_vals), len(x0_vals), figsize=(15, 12))
fig.suptitle('Score Comparison: Table (blue) vs Analytical (red dashed)', fontsize=14)

for i, t in enumerate(t_vals):
    t_idx = np.argmin(np.abs(t_grid - t))
    t_actual = t_grid[t_idx]
    
    for j, x0 in enumerate(x0_vals):
        x0_idx = np.argmin(np.abs(x0_grid - x0))
        x0_actual = x0_grid[x0_idx]
        
        ax = axes[i, j]
        ax.plot(x_grid, S_table[x0_idx, t_idx, :], 'b-', lw=2, label='Table')
        
        s_ana = ou_score(x_grid, x0_actual, t_actual)
        ax.plot(x_grid, s_ana, 'r--', lw=2, label='Analytical')
        
        ax.set_title(f't={t_actual:.2f}, x₀={x0_actual:.2f}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('score_verification.png', dpi=150)
print("Saved: score_verification.png")

# Density comparison if available
if has_density:
    fig, axes = plt.subplots(len(t_vals), len(x0_vals), figsize=(15, 12))
    fig.suptitle('Density Comparison: Table (blue) vs Analytical (red dashed)', fontsize=14)
    
    for i, t in enumerate(t_vals):
        t_idx = np.argmin(np.abs(t_grid - t))
        t_actual = t_grid[t_idx]
        
        for j, x0 in enumerate(x0_vals):
            x0_idx = np.argmin(np.abs(x0_grid - x0))
            x0_actual = x0_grid[x0_idx]
            
            ax = axes[i, j]
            ax.plot(x_grid, p_table[x0_idx, t_idx, :], 'b-', lw=2, label='Table')
            
            p_ana = ou_density(x_grid, x0_actual, t_actual)
            ax.plot(x_grid, p_ana, 'r--', lw=2, label='Analytical')
            
            ax.set_title(f't={t_actual:.2f}, x₀={x0_actual:.2f}', fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('Density p(x,t|x₀)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('density_verification.png', dpi=150)
    print("Saved: density_verification.png")

# Error heatmaps
fig, axes = plt.subplots(1, 2 if has_density else 1, figsize=(14, 5) if has_density else (7, 5))
if not has_density:
    axes = [axes]

x0_idx = N_x0 // 2
x0_val = x0_grid[x0_idx]

# Score error
ax = axes[0]
error_slice = abs_diff[x0_idx, :, :].T
im = ax.imshow(error_slice, aspect='auto', origin='lower',
               extent=[t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]],
               cmap='hot', interpolation='nearest')
ax.set_xlabel('Time t')
ax.set_ylabel('State x')
ax.set_title(f'|Score Error| at x₀={x0_val:.2f}')
plt.colorbar(im, ax=ax, label='Absolute Error')

# Density error
if has_density:
    ax = axes[1]
    error_slice_p = abs_diff_p[x0_idx, :, :].T
    im = ax.imshow(error_slice_p, aspect='auto', origin='lower',
                   extent=[t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]],
                   cmap='hot', interpolation='nearest')
    ax.set_xlabel('Time t')
    ax.set_ylabel('State x')
    ax.set_title(f'|Density Error| at x₀={x0_val:.2f}')
    plt.colorbar(im, ax=ax, label='Absolute Error')

plt.tight_layout()
plt.savefig('error_heatmaps.png', dpi=150)
print("Saved: error_heatmaps.png")

print()
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print()
print("SUMMARY:")
if abs_diff.max() < 0.1:
    print("✓ Score table matches analytical very well")
elif abs_diff.max() < 0.5:
    print("⚠ Score table has moderate errors (expected for coarse grid)")
else:
    print("✗ Score table has large errors")

if has_density:
    if abs_diff_p.max() < 0.01:
        print("✓ Density table matches analytical very well")
    elif abs_diff_p.max() < 0.05:
        print("⚠ Density table has moderate errors")
    else:
        print("✗ Density table has large errors")
    
    if 0.99 < masses.min() and masses.max() < 1.01:
        print("✓ Density properly normalized")
    else:
        print("✗ Density normalization issues")

print("\nCheck the generated PNG files for visual comparison!")