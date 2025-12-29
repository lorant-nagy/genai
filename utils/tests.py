"""
Test whether backward SDE recovers stationary distribution.

Compares initial stationary samples with final backward samples
using statistical tests and visualizations.
"""

import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import io
import re


def test_stationary_recovery(
    x0: torch.Tensor,
    x_final: torch.Tensor,
    config,
    save_dir: str,
    device: str
) -> bool:
    """
    Test if backward samples recover stationary distribution.
    
    Performs pixel-wise and aggregate statistical tests comparing
    initial stationary samples with final backward samples.
    
    Args:
        x0: Initial stationary samples, shape (n_samples, C, H, W)
        x_final: Final backward samples, shape (n_samples, C, H, W)
        config: Config object
        save_dir: Directory to save plots and logs
        device: Device string
    
    Returns:
        bool: Always returns True (test is informational, doesn't halt)
    """
    # Capture all output for text log
    log_buffer = io.StringIO()
    
    def log_print(msg):
        """Print and capture to log buffer"""
        print(msg)
        # Strip rich formatting for text log
        clean_msg = re.sub(r'\[.*?\]', '', str(msg))
        log_buffer.write(clean_msg + '\n')
    
    log_print("\n" + "="*70)
    log_print("STATIONARY RECOVERY TEST")
    log_print("="*70)
    
    # Extract shapes
    n_samples, C, H, W = x0.shape
    log_print(f"\nData shape: ({n_samples}, {C}, {H}, {W})")
    log_print(f"  n_samples: {n_samples}")
    log_print(f"  C (channels): {C}")
    log_print(f"  H x W: {H} x {W}")
    
    # Convert to numpy and move to CPU
    x0_np = x0.cpu().numpy()
    x_final_np = x_final.cpu().numpy()
    
    # ========================================
    # 1. PIXEL-WISE STATISTICS
    # ========================================
    log_print("\n" + "="*70)
    log_print("PIXEL-WISE STATISTICS (First Pixel: [0, 0, 0])")
    log_print("="*70)
    
    # Extract first pixel (channel 0, position [0,0])
    pixel_ch = 0
    pixel_r = 0
    pixel_c = 0
    
    x0_pixel = x0_np[:, pixel_ch, pixel_r, pixel_c]  # (n_samples,)
    x_final_pixel = x_final_np[:, pixel_ch, pixel_r, pixel_c]  # (n_samples,)
    
    log_print(f"\nFirst Pixel [ch={pixel_ch}, r={pixel_r}, c={pixel_c}]:")
    log_print(f"  Initial (x0):")
    log_print(f"    Mean: {np.mean(x0_pixel):.4f}")
    log_print(f"    Std:  {np.std(x0_pixel):.4f}")
    log_print(f"    Min:  {np.min(x0_pixel):.4f}")
    log_print(f"    Max:  {np.max(x0_pixel):.4f}")
    
    log_print(f"  Final (backward):")
    log_print(f"    Mean: {np.mean(x_final_pixel):.4f}")
    log_print(f"    Std:  {np.std(x_final_pixel):.4f}")
    log_print(f"    Min:  {np.min(x_final_pixel):.4f}")
    log_print(f"    Max:  {np.max(x_final_pixel):.4f}")
    
    # KS test for first pixel
    ks_stat_pixel, p_val_pixel = stats.ks_2samp(x0_pixel, x_final_pixel)
    log_print(f"\n  Kolmogorov-Smirnov Test:")
    log_print(f"    KS statistic: {ks_stat_pixel:.4f}")
    log_print(f"    p-value: {p_val_pixel:.4f}")
    log_print(f"    Interpretation: {'PASS' if p_val_pixel > 0.05 else 'FAIL'} (threshold: p > 0.05)")
    
    # ========================================
    # 2. AGGREGATE STATISTICS (ALL PIXELS)
    # ========================================
    log_print("\n" + "="*70)
    log_print("AGGREGATE STATISTICS (All Pixels)")
    log_print("="*70)
    
    # Flatten all pixels
    x0_flat = x0_np.reshape(n_samples, -1)  # (n_samples, C*H*W)
    x_final_flat = x_final_np.reshape(n_samples, -1)
    
    # Flatten completely for histogram
    x0_all = x0_flat.flatten()
    x_final_all = x_final_flat.flatten()
    
    log_print(f"\nAll Pixels (Total: {x0_all.size}):")
    log_print(f"  Initial (x0):")
    log_print(f"    Mean: {np.mean(x0_all):.4f}")
    log_print(f"    Std:  {np.std(x0_all):.4f}")
    log_print(f"    Min:  {np.min(x0_all):.4f}")
    log_print(f"    Max:  {np.max(x0_all):.4f}")
    
    log_print(f"  Final (backward):")
    log_print(f"    Mean: {np.mean(x_final_all):.4f}")
    log_print(f"    Std:  {np.std(x_final_all):.4f}")
    log_print(f"    Min:  {np.min(x_final_all):.4f}")
    log_print(f"    Max:  {np.max(x_final_all):.4f}")
    
    # KS test for aggregate
    ks_stat_agg, p_val_agg = stats.ks_2samp(x0_all, x_final_all)
    log_print(f"\n  Kolmogorov-Smirnov Test:")
    log_print(f"    KS statistic: {ks_stat_agg:.4f}")
    log_print(f"    p-value: {p_val_agg:.4f}")
    log_print(f"    Interpretation: {'PASS' if p_val_agg > 0.05 else 'FAIL'} (threshold: p > 0.05)")
    
    # ========================================
    # 3. PER-PIXEL MEAN AND STD
    # ========================================
    log_print("\n" + "="*70)
    log_print("PER-PIXEL STATISTICS SUMMARY")
    log_print("="*70)
    
    # Compute mean and std for each pixel across samples
    x0_pixel_means = x0_flat.mean(axis=0)  # (C*H*W,)
    x0_pixel_stds = x0_flat.std(axis=0)
    
    x_final_pixel_means = x_final_flat.mean(axis=0)
    x_final_pixel_stds = x_final_flat.std(axis=0)
    
    log_print(f"\nPer-Pixel Means:")
    log_print(f"  Initial (x0):")
    log_print(f"    Mean of means: {np.mean(x0_pixel_means):.4f}")
    log_print(f"    Std of means:  {np.std(x0_pixel_means):.4f}")
    log_print(f"  Final (backward):")
    log_print(f"    Mean of means: {np.mean(x_final_pixel_means):.4f}")
    log_print(f"    Std of means:  {np.std(x_final_pixel_means):.4f}")
    
    log_print(f"\nPer-Pixel Stds:")
    log_print(f"  Initial (x0):")
    log_print(f"    Mean of stds: {np.mean(x0_pixel_stds):.4f}")
    log_print(f"    Std of stds:  {np.std(x0_pixel_stds):.4f}")
    log_print(f"  Final (backward):")
    log_print(f"    Mean of stds: {np.mean(x_final_pixel_stds):.4f}")
    log_print(f"    Std of stds:  {np.std(x_final_pixel_stds):.4f}")
    
    # ========================================
    # 4. PIXEL-BY-PIXEL KS TESTS
    # ========================================
    log_print("\n" + "="*70)
    log_print("PIXEL-BY-PIXEL KS TESTS")
    log_print("="*70)
    
    # Run KS test for each pixel
    n_pixels = C * H * W
    p_values = []
    
    for i in range(n_pixels):
        x0_pix = x0_flat[:, i]
        x_final_pix = x_final_flat[:, i]
        _, p_val = stats.ks_2samp(x0_pix, x_final_pix)
        p_values.append(p_val)
    
    p_values = np.array(p_values)
    n_pass = np.sum(p_values > 0.05)
    pass_rate = n_pass / n_pixels
    
    log_print(f"\nTotal pixels: {n_pixels}")
    log_print(f"  Passed (p > 0.05): {n_pass} ({pass_rate*100:.1f}%)")
    log_print(f"  Failed (p <= 0.05): {n_pixels - n_pass} ({(1-pass_rate)*100:.1f}%)")
    log_print(f"  Min p-value: {np.min(p_values):.4f}")
    log_print(f"  Max p-value: {np.max(p_values):.4f}")
    log_print(f"  Mean p-value: {np.mean(p_values):.4f}")
    log_print(f"  Median p-value: {np.median(p_values):.4f}")
    
    # ========================================
    # 5. CREATE VISUALIZATIONS
    # ========================================
    log_print("\n" + "="*70)
    log_print("CREATING VISUALIZATIONS")
    log_print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: First pixel histogram
    ax = axes[0, 0]
    bins = 50
    ax.hist(x0_pixel, bins=bins, alpha=0.5, label='Initial (x0)', density=True, color='blue')
    ax.hist(x_final_pixel, bins=bins, alpha=0.5, label='Final (backward)', density=True, color='red')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Density')
    ax.set_title(f'First Pixel [ch={pixel_ch}, r={pixel_r}, c={pixel_c}]\nKS p-value: {p_val_pixel:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Aggregate histogram
    ax = axes[0, 1]
    ax.hist(x0_all, bins=bins, alpha=0.5, label='Initial (x0)', density=True, color='blue')
    ax.hist(x_final_all, bins=bins, alpha=0.5, label='Final (backward)', density=True, color='red')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Density')
    ax.set_title(f'All Pixels (n={x0_all.size})\nKS p-value: {p_val_agg:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Per-pixel mean comparison
    ax = axes[1, 0]
    ax.scatter(x0_pixel_means, x_final_pixel_means, alpha=0.5, s=10)
    lim = max(abs(x0_pixel_means).max(), abs(x_final_pixel_means).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='y=x (perfect match)')
    ax.set_xlabel('Initial Mean')
    ax.set_ylabel('Final Mean')
    ax.set_title('Per-Pixel Means')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    # Plot 4: Per-pixel std comparison
    ax = axes[1, 1]
    ax.scatter(x0_pixel_stds, x_final_pixel_stds, alpha=0.5, s=10)
    lim = max(x0_pixel_stds.max(), x_final_pixel_stds.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'r--', linewidth=2, label='y=x (perfect match)')
    ax.set_xlabel('Initial Std')
    ax.set_ylabel('Final Std')
    ax.set_title('Per-Pixel Standard Deviations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'stationary_recovery_test.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    log_print(f"\nPlot saved: {plot_path}")
    plt.close()
    
    # Save text log
    log_path = os.path.join(save_dir, 'stationary_recovery_test.txt')
    with open(log_path, 'w') as f:
        f.write(log_buffer.getvalue())
    log_print(f"Log saved: {log_path}")
    
    # Upload to wandb
    try:
        import wandb
        wandb.log({
            "stationary_recovery_test": wandb.Image(plot_path),
            "stationary_recovery_log": wandb.Html(f"<pre>{log_buffer.getvalue()}</pre>"),
            "test_ks_first_pixel_pvalue": p_val_pixel,
            "test_ks_aggregate_pvalue": p_val_agg,
            "test_pixel_pass_rate": pass_rate,
        })
        log_print(f"Uploaded to wandb")
    except Exception as e:
        log_print(f"Warning: Could not upload to wandb: {e}")
    
    # Summary
    log_print("\n" + "="*70)
    log_print("TEST SUMMARY")
    log_print("="*70)
    log_print(f"\nFirst Pixel KS Test: {'PASS' if p_val_pixel > 0.05 else 'FAIL'} (p={p_val_pixel:.4f})")
    log_print(f"Aggregate KS Test: {'PASS' if p_val_agg > 0.05 else 'FAIL'} (p={p_val_agg:.4f})")
    log_print(f"Pixel-wise Pass Rate: {pass_rate*100:.1f}%")
    log_print("\n" + "="*70 + "\n")
    
    # Always return True (informational test, doesn't halt)
    return True