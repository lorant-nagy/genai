# utils/eval_plotting.py
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# ------------------------- helpers -------------------------

def _to_numpy(x):
    """Convert torch tensors or array-likes to a CPU numpy array."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_imshow_img(arr: np.ndarray, cmap: Optional[str], vmin: float, vmax: float) -> Tuple[np.ndarray, dict]:
    """
    Return (img, kwargs) suitable for plt.imshow, handling gray/RGB(A) and (C,H,W)/(H,W,C).
    For single-channel images we pass cmap/vmin/vmax; for color images we let matplotlib choose.
    """
    arr = _to_numpy(arr)

    if arr.ndim == 2:
        return arr, {"cmap": cmap, "vmin": vmin, "vmax": vmax}
    if arr.ndim == 3:
        # (C,H,W)
        if arr.shape[0] in (1, 3, 4):
            if arr.shape[0] == 1:
                return arr[0], {"cmap": cmap, "vmin": vmin, "vmax": vmax}
            return np.moveaxis(arr, 0, -1), {}
        # (H,W,C)
        if arr.shape[-1] in (1, 3, 4):
            if arr.shape[-1] == 1:
                return arr[..., 0], {"cmap": cmap, "vmin": vmin, "vmax": vmax}
            return arr, {}
    squeezed = np.squeeze(arr)
    if squeezed.ndim in (2, 3):
        return _as_imshow_img(squeezed, cmap, vmin, vmax)
    raise ValueError(f"Unsupported image shape for imshow: {arr.shape}")


def _save_samples_strip(samples, out_path, *, cmap, vmin, vmax, dpi, annotate: bool) -> None:
    """
    Save a 1xN strip (zero margins) with optional index labels.
    Used for BOTH data_samples.png and backward_final_samples.png to ensure identical styling.
    """
    n = len(samples)
    if n <= 0:
        return

    fig, axs = plt.subplots(1, n, figsize=(n * 2.5, 2.5), squeeze=False)
    axs = axs[0]  # 1D array of axes length n

    for i in range(n):
        ax = axs[i]
        img_show, imshow_kwargs = _as_imshow_img(samples[i], cmap=cmap, vmin=vmin, vmax=vmax)
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
    print(f"Saved {out_path}", flush=True)


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
    images, times, out_path, *, cmap, vmin, vmax, dpi, annotate_first_col: bool, index_fmt: str
) -> None:
    """
    images: list-of-lists shape [rows][cols], each leaf is (H,W) or (C,H,W) or (H,W,C)
    times:  1D array-like of length cols
    Saves a no-margin grid with a thin header row listing t-values per column.
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
            img_show, imshow_kwargs = _as_imshow_img(cell, cmap=cmap, vmin=vmin, vmax=vmax)
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
    print(f"Saved {out_path}", flush=True)


# ------------------------- main entry -------------------------

def plot_corruption_and_samples(
    corruptor,
    batch_of_data: torch.Tensor,
    eval_path: str,
    *,
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
            )
