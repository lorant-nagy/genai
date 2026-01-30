# utils/eval_plotting.py
from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt


# ------------------------- helpers -------------------------

def _to_numpy(x):
    """Convert torch tensors or array-likes to a CPU numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ------------------------- NEW: Metric Evolution Plotting -------------------------

def plot_metrics_evolution(
    metrics_history: Dict[str, List[float]],
    epochs: List[int],
    save_path: str,
    title: str = "Metrics Evolution During Training"
) -> None:
    """
    Plot evolution of all metrics over training.
    
    Creates separate subplots for each metric type:
    - FID
    - KID (mean ± std)
    - Wasserstein distances (W1, W2 in both spaces)
    
    Args:
        metrics_history: Dict mapping metric name to list of values
        epochs: List of epoch numbers when metrics were computed
        save_path: Path to save the plot
        title: Plot title
    """
    # Determine which metrics we have
    has_fid = 'fid' in metrics_history and len(metrics_history['fid']) > 0
    has_kid = 'kid_mean' in metrics_history and len(metrics_history['kid_mean']) > 0
    has_w_pixel = 'w1_pixel' in metrics_history and len(metrics_history['w1_pixel']) > 0
    has_w_embed = 'w1_embedding' in metrics_history and len(metrics_history['w1_embedding']) > 0
    
    # Count how many subplots we need
    n_plots = sum([has_fid, has_kid, has_w_pixel, has_w_embed])
    
    if n_plots == 0:
        print(f"  [yellow]No metrics to plot[/yellow]")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), squeeze=False)
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot FID
    if has_fid:
        ax = axes[plot_idx]
        ax.plot(epochs, metrics_history['fid'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('FID', fontsize=11)
        ax.set_title('FID Evolution (lower is better)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot KID
    if has_kid:
        ax = axes[plot_idx]
        kid_mean = np.array(metrics_history['kid_mean'])
        kid_std = np.array(metrics_history['kid_std'])
        
        ax.plot(epochs, kid_mean, 'o-', linewidth=2, markersize=6, color='#A23B72', label='KID mean')
        ax.fill_between(epochs, kid_mean - kid_std, kid_mean + kid_std, 
                        alpha=0.2, color='#A23B72', label='± 1 std')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('KID', fontsize=11)
        ax.set_title('KID Evolution (lower is better)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot Wasserstein (pixel space)
    if has_w_pixel:
        ax = axes[plot_idx]
        if 'w1_pixel' in metrics_history:
            ax.plot(epochs, metrics_history['w1_pixel'], 'o-', linewidth=2, 
                   markersize=6, color='#F18F01', label='W1')
        if 'w2_pixel' in metrics_history:
            ax.plot(epochs, metrics_history['w2_pixel'], 's-', linewidth=2, 
                   markersize=6, color='#C73E1D', label='W2')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Wasserstein Distance', fontsize=11)
        ax.set_title('Wasserstein Distance - Pixel Space (lower is better)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot Wasserstein (embedding space)
    if has_w_embed:
        ax = axes[plot_idx]
        if 'w1_embedding' in metrics_history:
            ax.plot(epochs, metrics_history['w1_embedding'], 'o-', linewidth=2, 
                   markersize=6, color='#6A4C93', label='W1')
        if 'w2_embedding' in metrics_history:
            ax.plot(epochs, metrics_history['w2_embedding'], 's-', linewidth=2, 
                   markersize=6, color='#1982C4', label='W2')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Wasserstein Distance', fontsize=11)
        ax.set_title('Wasserstein Distance - Embedding Space (lower is better)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        plot_idx += 1
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [green]✓ Saved metrics evolution plot to {save_path}[/green]")


def plot_single_metric_evolution(
    metric_name: str,
    values: List[float],
    epochs: List[int],
    save_path: str,
    std_values: Optional[List[float]] = None,
    ylabel: str = None,
    title: str = None
) -> None:
    """
    Plot evolution of a single metric.
    
    Args:
        metric_name: Name of the metric (for title)
        values: List of metric values
        epochs: List of epoch numbers
        save_path: Path to save the plot
        std_values: Optional standard deviations (for KID)
        ylabel: Y-axis label (default: metric_name)
        title: Plot title (default: auto-generated)
    """
    if len(values) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    values = np.array(values)
    
    # Plot main line
    ax.plot(epochs, values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    
    # Add error band if std provided
    if std_values is not None:
        std_values = np.array(std_values)
        ax.fill_between(epochs, values - std_values, values + std_values,
                        alpha=0.2, color='#2E86AB', label='± 1 std')
        ax.legend(loc='best', fontsize=11)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel or metric_name.upper(), fontsize=12)
    ax.set_title(title or f'{metric_name.upper()} Evolution During Training', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [green]✓ Saved {metric_name} plot to {save_path}[/green]")


def _as_imshow_img(
    arr: np.ndarray,
    cmap: Optional[str],
    vmin: float,
    vmax: float,
    normalizer=None,  # NEW: optional normalizer for denormalization
) -> Tuple[np.ndarray, dict]:
    """
    Return (img, kwargs) suitable for plt.imshow, handling gray/RGB(A) and (C,H,W)/(H,W,C).
    
    Args:
        arr: Input array (numpy or torch tensor)
        cmap: Colormap for grayscale images
        vmin, vmax: Value range for grayscale images
        normalizer: Optional Normalizer instance for denormalization
    
    Returns:
        Tuple of (image_array, imshow_kwargs)
    """
    arr = _to_numpy(arr)
    
    # Denormalize if normalizer provided
    if normalizer is not None:
        arr_torch = torch.from_numpy(arr)
        arr = normalizer.denormalize(arr_torch).cpu().numpy()

    if arr.ndim == 2:
        return arr, {"cmap": cmap, "vmin": vmin, "vmax": vmax}
    if arr.ndim == 3:
        # (C,H,W)
        if arr.shape[0] in (1, 3, 4):
            if arr.shape[0] == 1:
                return arr[0], {"cmap": cmap, "vmin": vmin, "vmax": vmax}
            # RGB/RGBA: move channels to last dimension
            img = np.moveaxis(arr, 0, -1)
            # Ensure [0,1] range for matplotlib (clip after denormalization)
            img = np.clip(img, 0.0, 1.0)
            return img, {}
        # (H,W,C)
        if arr.shape[-1] in (1, 3, 4):
            if arr.shape[-1] == 1:
                return arr[..., 0], {"cmap": cmap, "vmin": vmin, "vmax": vmax}
            # RGB/RGBA already in correct format
            img = np.clip(arr, 0.0, 1.0)
            return img, {}
    squeezed = np.squeeze(arr)
    if squeezed.ndim in (2, 3):
        return _as_imshow_img(squeezed, cmap, vmin, vmax, normalizer)
    raise ValueError(f"Unsupported image shape for imshow: {arr.shape}")


def _save_samples_strip(samples, out_path, *, cmap, vmin, vmax, dpi, annotate: bool, normalizer=None) -> None:
    """
    Save a 1xN strip (zero margins) with optional index labels.
    Used for BOTH data_samples.png and backward_final_samples.png to ensure identical styling.
    
    Args:
        normalizer: Optional Normalizer instance for denormalization
    """
    n = len(samples)
    if n <= 0:
        return

    fig, axs = plt.subplots(1, n, figsize=(n * 2.5, 2.5), squeeze=False)
    axs = axs[0]  # 1D array of axes length n

    for i in range(n):
        ax = axs[i]
        img_show, imshow_kwargs = _as_imshow_img(samples[i], cmap=cmap, vmin=vmin, vmax=vmax, normalizer=normalizer)
        ax.imshow(img_show, **imshow_kwargs)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
        if annotate:
            ax.text(
                0.02, 0.02, f"{i}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color="white",
                bbox=dict(facecolor="black", alpha=0.35, lw=0, pad=1.0),
            )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    # print(f"Saved {out_path}", flush=True)



def _linspace_indices(T: int, target_cols: int, reverse: bool = False) -> np.ndarray:
    """
    Choose ~target_cols indices spanning [0..T-1]. Uses linspace for even coverage.
    If reverse=True, return them in descending order (for reverse evolution plots).
    """
    target_cols = max(1, int(target_cols))
    if target_cols >= T:
        idx = np.arange(T, dtype=int)
    else:
        idx = np.linspace(0, T - 1, target_cols).round().astype(int)
    return idx[::-1] if reverse else idx


def _plot_grid_with_header(
    images, times, out_path, *, cmap, vmin, vmax, dpi, annotate_first_col: bool, index_fmt: str, normalizer=None
) -> None:
    """
    images: list-of-lists shape [rows][cols], each leaf is (H,W) or (C,H,W) or (H,W,C)
    times:  1D array-like of length cols
    Saves a no-margin grid with a thin header row listing t-values per column.
    
    Args:
        normalizer: Optional Normalizer instance for denormalization
    """
    # Treat images as pure lists to avoid object-dtype arrays.
    rows = len(images)
    assert rows >= 0
    cols = len(images[0]) if rows > 0 else len(np.asarray(times))

    times = _to_numpy(times).reshape(-1)
    assert len(times) == cols, f"times length {len(times)} != cols {cols}"

    height_ratios = [0.12] + [1] * rows  # thin header row
    fig, axs = plt.subplots(
        rows + 1, cols,
        figsize=(cols * 2, max(1, rows) * 2 + 0.3),
        constrained_layout=False,
        gridspec_kw={"wspace": 0, "hspace": 0, "height_ratios": height_ratios},
        squeeze=False,  # always get a 2D array of axes
    )

    header_axes = axs[0, :]
    img_axes = axs[1:, :] if rows > 0 else np.empty((0, cols), dtype=object)

    # Header labels
    for c in range(cols):
        hax = header_axes[c]
        hax.set_axis_off()
        hax.text(0.5, 0.5, f"t={float(times[c]):.2f}", transform=hax.transAxes,
                 ha="center", va="center", fontsize=9)

    # Cells
    for r in range(rows):
        for c in range(cols):
            ax = img_axes[r, c]
            cell = images[r][c]          # keep as list indexing, not numpy object array indexing
            img_show, imshow_kwargs = _as_imshow_img(cell, cmap=cmap, vmin=vmin, vmax=vmax, normalizer=normalizer)
            ax.imshow(img_show, **imshow_kwargs)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
            if annotate_first_col and c == 0:
                ax.text(
                    0.02, 0.02, index_fmt.format(idx=r),
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, lw=0, pad=1.0),
                )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    # print(f"Saved {out_path}", flush=True)


# ------------------------- main entry -------------------------

def plot_score_table_heatmaps(
    score_table_obj,
    process,
    eval_path: str,
    *,
    n_time_points: int = 10,
    dpi: int = 150,
) -> None:
    """
    Visualize score table as heatmaps across time.
    
    Args:
        score_table_obj: ScoreTable instance with .S, .x0_grid, .x_grid, .t_grid
        process: Process instance with parameters (alpha, c_alpha, c_0, sigma, T)
        eval_path: Directory to save the figure
        n_time_points: Number of time slices to show (default 10)
        dpi: Figure DPI
    """
    os.makedirs(eval_path, exist_ok=True)
    
    # Extract data
    S = _to_numpy(score_table_obj.S)  # [N_x0, N_t, N_x]
    x0_grid = _to_numpy(score_table_obj.x0_grid)
    x_grid = _to_numpy(score_table_obj.x_grid)
    t_grid = _to_numpy(score_table_obj.t_grid)
    
    N_x0, N_t, N_x = S.shape
    
    # Select time indices (evenly spaced)
    if n_time_points >= N_t:
        time_indices = np.arange(N_t)
    else:
        time_indices = np.linspace(0, N_t - 1, n_time_points).round().astype(int)
    
    n_plots = len(time_indices)
    
    # Layout: 2 rows if we have many plots, otherwise 1 row
    if n_plots <= 5:
        nrows, ncols = 1, n_plots
    else:
        nrows, ncols = 2, (n_plots + 1) // 2
    
    # Fixed colormap: standard diverging for scores (centered at zero)
    cmap = 'RdBu_r'  # Blue (negative) → White (zero) → Red (positive)
    
    # Create figure
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.2 + 0.8))
    
    # Main title
    title_lines = ["Score Table: ∇ₓ log p(x,t|x₀)"]
    
    if hasattr(process, 'alpha'):
        title_lines.append(
            f"α={process.alpha:.2f}, c_α={process.c_alpha:.2f}, "
            f"c₀={process.c_0:.2f}, σ={process.sigma:.2f}, T={process.T:.2f}"
        )
    elif hasattr(process, 'beta'):
        title_lines.append(f"β={process.beta:.2f}, T={process.T:.2f}")
    
    fig.suptitle('\n'.join(title_lines), fontsize=11)
    
    # Fixed color scale for score visualization
    # Scores are gradients ∇ₓ log p, typically in range [-10, 10] for normalized data
    vmin, vmax = -10.0, 10.0  # Fixed symmetric range
    
    # Create subplots
    for idx, t_idx in enumerate(time_indices):
        ax = plt.subplot(nrows, ncols, idx + 1)
        
        score_slice = S[:, t_idx, :]
        t_val = t_grid[t_idx]
        
        # Use pcolormesh instead of imshow - no aliasing artifacts
        im = ax.pcolormesh(
            x_grid, x0_grid, score_slice,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='nearest',
        )
        
        ax.set_xlabel('x')
        ax.set_ylabel('x₀')
        ax.set_title(f't = {t_val:.3f}')
    
    # Remove empty subplots
    total_subplots = nrows * ncols
    for idx in range(n_plots, total_subplots):
        ax = plt.subplot(nrows, ncols, idx + 1)
        ax.axis('off')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Score', rotation=270, labelpad=15)
    
    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.92, 
                       wspace=0.30, hspace=0.30)
    
    out_path = os.path.join(eval_path, "score_table_heatmaps.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved score table heatmaps to {out_path}", flush=True)


def plot_density_table_heatmaps(
    score_table_obj,
    process,
    eval_path: str,
    *,
    n_time_points: int = 10,
    dpi: int = 150,
    log_scale: bool = True,
) -> None:
    """
    Visualize density table p(x,t|x0) as heatmaps across time.
    
    Args:
        score_table_obj: ScoreTable instance (or dict) with .p_table attribute
        process: Process instance with parameters  
        eval_path: Directory to save the figure
        n_time_points: Number of time slices to show (default 10)
        dpi: Figure DPI
        log_scale: If True, plot log(p+eps) for better visualization (default True)
    """
    os.makedirs(eval_path, exist_ok=True)
    
    # Handle both dict and object inputs
    if isinstance(score_table_obj, dict):
        if 'p_table' not in score_table_obj:
            print("  Density table not found in score_table_obj (score-only mode)", flush=True)
            return
        p_table = _to_numpy(score_table_obj['p_table'])
        x0_grid = _to_numpy(score_table_obj['x0_grid'])
        x_grid = _to_numpy(score_table_obj['x_grid'])
        t_grid = _to_numpy(score_table_obj['t_grid'])
        config = score_table_obj.get('config', {})
        
        if process is None:
            class MockProcess:
                def __init__(self, cfg):
                    self.alpha = cfg.get('alpha')
                    self.c_alpha = cfg.get('c_alpha')
                    self.c_0 = cfg.get('c_0')
                    self.sigma = cfg.get('sigma')
                    self.T = cfg.get('T')
                    self.beta = cfg.get('beta')
            process = MockProcess(config)
    else:
        if not hasattr(score_table_obj, 'p_table') or score_table_obj.p_table is None:
            print("  Density table not available in score_table_obj", flush=True)
            return
        p_table = _to_numpy(score_table_obj.p_table)  # [N_x0, N_t, N_x]
        x0_grid = _to_numpy(score_table_obj.x0_grid)
        x_grid = _to_numpy(score_table_obj.x_grid)
        t_grid = _to_numpy(score_table_obj.t_grid)
    
    N_x0, N_t, N_x = p_table.shape
    
    # Choose what to plot
    if log_scale:
        eps = 1e-12
        plot_data = np.log(p_table + eps)
        data_label = 'log p(x,t|x₀)'
    else:
        plot_data = p_table
        data_label = 'p(x,t|x₀)'
    
    # Select time indices
    if n_time_points >= N_t:
        time_indices = np.arange(N_t)
    else:
        time_indices = np.linspace(0, N_t - 1, n_time_points).round().astype(int)
    
    n_plots = len(time_indices)
    
    # Layout
    if n_plots <= 5:
        nrows, ncols = 1, n_plots
    else:
        nrows, ncols = 2, (n_plots + 1) // 2
    
    # Fixed colormap: standard for probability densities
    cmap = 'viridis'
    
    # Create figure
    fig = plt.figure(figsize=(ncols * 3.5, nrows * 3.2 + 0.8))
    
    # Title
    title_lines = [f"Density Table: {data_label}"]
    
    if process is not None:
        if hasattr(process, 'alpha') and process.alpha is not None:
            title_lines.append(
                f"α={process.alpha:.2f}, c_α={process.c_alpha:.2f}, "
                f"c₀={process.c_0:.2f}, σ={process.sigma:.2f}, T={process.T:.2f}"
            )
        elif hasattr(process, 'beta') and process.beta is not None:
            title_lines.append(f"β={process.beta:.2f}, T={process.T:.2f}")
    
    fig.suptitle('\n'.join(title_lines), fontsize=11)
    
    # Fixed color scale for density visualization
    if log_scale:
        # For log scale: fixed range that works well for log probabilities
        vmin, vmax = -10.0, 0.0
    else:
        # For linear scale: fixed range for probability densities
        vmin, vmax = -0.2, 1.2
    
    # Create subplots
    for idx, t_idx in enumerate(time_indices):
        ax = plt.subplot(nrows, ncols, idx + 1)
        
        density_slice = plot_data[:, t_idx, :]
        t_val = t_grid[t_idx]
        
        # Use pcolormesh instead of imshow - no aliasing artifacts
        im = ax.pcolormesh(
            x_grid, x0_grid, density_slice,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='nearest',
        )
        
        ax.set_xlabel('x')
        ax.set_ylabel('x₀')
        ax.set_title(f't = {t_val:.3f}')
    
    # Remove empty subplots
    total_subplots = nrows * ncols
    for idx in range(n_plots, total_subplots):
        ax = plt.subplot(nrows, ncols, idx + 1)
        ax.axis('off')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(data_label, rotation=270, labelpad=15)
    
    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.92, 
                       wspace=0.30, hspace=0.30)
    
    out_path = os.path.join(eval_path, "density_table_heatmaps.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved density table heatmaps to {out_path}", flush=True)


def plot_corruption_and_samples(
    corruptor,
    batch_of_data: torch.Tensor,
    eval_path: str,
    *,
    normalizer=None,  # NEW: optional normalizer for denormalization
    cmap: str = "gray",
    max_trajectories: int = 5,
    sample_grid_count: int = 9,
    vmin: float = 0.0,
    vmax: float = 1.0,
    dpi: int = 150,
    show_time_header: bool = True,
    mark_t0_indices: bool = True,
    label_data_sample_indices: bool = True,
    # Optional reverse-evolution inputs (precomputed):
    t_grid=None,          # [T] torch.Tensor or np.ndarray
    X=None,               # [T, B, C, H, W] or [T, B, H, W] (torch or np)
    # Grid density:
    n_time_cols: int = 10,  # target number of columns shown in the trajectory grids
) -> None:
    """
    Produce FOUR figures in eval_path:
      1) corruption_trajectories.png   — forward time grid: rows=samples, cols=subsampled time points
      2) data_samples.png              — a 1xN strip of raw data samples
      3) reverse_evolution.png         — reverse-time grid using (t_grid, X) if provided
      4) backward_final_samples.png    — 1xN strip of reverse terminal samples; styled EXACTLY like #2

    Args:
        normalizer: Optional Normalizer instance. If provided, denormalizes data before plotting.
                   Otherwise expects data already in visualization range [vmin, vmax].
    
    Notes
    -----
    * This function performs no heavy computation beyond a single `corruptor(batch_of_data)` call.
    * If you want fewer rows/cols or fewer reverse samples, slice inputs before calling:
         X = X[:, :5]   # only first 5 samples as rows in reverse grids
    """
    os.makedirs(eval_path, exist_ok=True)

    with torch.no_grad():
        corrupted = corruptor(batch_of_data)

    x_traj = _to_numpy(corrupted["x"])              # [T, B, C, H, W] or [T, B, H, W]
    t_vals = _to_numpy(corrupted["t"]).reshape(-1)  # [T]
    if t_vals.ndim != 1:
        raise ValueError(f"Expected times to be 1-D after squeeze; got {t_vals.shape}")

    T_f, B_f = x_traj.shape[0], x_traj.shape[1]
    rows_f = int(min(max_trajectories, B_f))
    col_idx_f = _linspace_indices(T_f, n_time_cols, reverse=False)
    times_f = t_vals[col_idx_f]

    grid_fwd = [[x_traj[t, r] for t in col_idx_f] for r in range(rows_f)]
    _plot_grid_with_header(
        images=grid_fwd, times=times_f,
        out_path=os.path.join(eval_path, "corruption_trajectories.png"),
        cmap=cmap, vmin=vmin, vmax=vmax, dpi=dpi,
        annotate_first_col=mark_t0_indices, index_fmt="{idx}",
        normalizer=normalizer,  # Pass normalizer
    )

    n_show = int(min(sample_grid_count, batch_of_data.shape[0]))
    if n_show > 0:
        batch_np = _to_numpy(batch_of_data)
        samples_fwd = [batch_np[i] for i in range(n_show)]
        _save_samples_strip(
            samples_fwd,
            os.path.join(eval_path, "data_samples.png"),
            cmap=cmap, vmin=vmin, vmax=vmax, dpi=dpi,
            annotate=label_data_sample_indices,
            normalizer=normalizer,  # Pass normalizer
        )

    if (t_grid is not None) and (X is not None):
        t_np = _to_numpy(t_grid).reshape(-1)
        X_np = _to_numpy(X)
        if X_np.shape[0] != t_np.shape[0]:
            raise ValueError(f"First dim of X ({X_np.shape[0]}) must match len(t_grid) ({t_np.shape[0]})")

        T_r, B_r = X_np.shape[0], X_np.shape[1]
        rows_r = int(min(max_trajectories, B_r))

        col_idx_rev = _linspace_indices(T_r, n_time_cols, reverse=True)
        times_rev = t_np[col_idx_rev]

        grid_rev = [[X_np[t, r] for t in col_idx_rev] for r in range(rows_r)]
        _plot_grid_with_header(
            images=grid_rev, times=times_rev,
            out_path=os.path.join(eval_path, "reverse_evolution.png"),
            cmap=cmap, vmin=vmin, vmax=vmax, dpi=dpi,
            annotate_first_col=mark_t0_indices, index_fmt="{idx}",
            normalizer=normalizer,  # Pass normalizer
        )

        earliest_idx = int(np.argmin(t_np))
        latest_idx = int(np.argmax(t_np))
        final_batch = X_np[latest_idx]  # [B, C, H, W] or [B, H, W]
        n_back = int(min(sample_grid_count, final_batch.shape[0]))
        if n_back > 0:
            samples_back = [final_batch[i] for i in range(n_back)]
            _save_samples_strip(
                samples_back,
                os.path.join(eval_path, "backward_final_samples.png"),
                cmap=cmap, vmin=vmin, vmax=vmax, dpi=dpi,
                annotate=label_data_sample_indices,   # identical styling to data_samples.png
                normalizer=normalizer,  # Pass normalizer
            )