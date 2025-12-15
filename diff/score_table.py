"""
Precomputed conditional score table for superlinear Langevin processes.

This module implements a PDE-based approach to compute conditional scores
for coordinatewise SDEs. 

The main workflow:

1. Solve 1D Fokker-Planck equation for each initial value x0
2. Compute scores via finite differences on log-density
3. Store in 3D table (x0, t, x)
4. Runtime: vectorized trilinear interpolation
"""

from __future__ import annotations
import torch
import os
from dataclasses import dataclass, asdict
from typing import Callable, Optional, Tuple, Dict, Any
import warnings


class ScoreTableNumericalError(Exception):
    """
    Exception raised when score table building encounters numerical instability.
    This includes NaN values, Inf values, or severe mass loss.
    """
    pass
from rich import print


@dataclass
class ScoreTableConfig:
    """Configuration for score table construction and storage."""
    
    # Grid sizes
    N_x0: int
    N_x: int
    N_t: int
    
    # Domain bounds
    x_min: float
    x_max: float
    x0_min: float = -1.2
    x0_max: float = 1.2
    
    # PDE solver parameters
    initial_gaussian_std: float = 0.01
    density_clamp_eps: float = 1e-12
    boundary_condition: str = "zero_flux"
    time_stepping_scheme: str = "explicit_euler"  # Options: "explicit_euler", "semi_implicit", "tamed_euler"
    
    # Storage
    cache_dir: str = "cache"
    table_filename: str = "superlinear_score_table.pt"
    
    # Process parameters (for validation)
    alpha: float = 3.0
    c_alpha: float = 1.0
    c_0: float = 0.5
    sigma: float = 1.0
    T: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> ScoreTableConfig:
        """Reconstruct from dictionary."""
        return cls(**d)
    
    def validate_process_match(self, process) -> bool:
        """Check if config matches a given process's parameters."""
        return (
            abs(self.alpha - process.alpha) < 1e-6 and
            abs(self.c_alpha - process.c_alpha) < 1e-6 and
            abs(self.c_0 - process.c_0) < 1e-6 and
            abs(self.sigma - process.sigma) < 1e-6 and
            abs(self.T - process.T) < 1e-6
        )


class PDESolver1D:
    """
    Solve the 1D forward Fokker-Planck equation:
    
        ∂_t p(t,x|x0) = -∂_x[g_scal(x)·p(t,x|x0)] + (σ²/2)·∂_x² p(t,x|x0)
    
    Using explicit Euler time-stepping with conservative flux formulation.
    """
    
    def __init__(
        self,
        g_scal: Callable[[torch.Tensor], torch.Tensor],
        sigma: float,
        x_grid: torch.Tensor,
        t_grid: torch.Tensor,
        config: ScoreTableConfig,
        device: str = "cpu",
    ):
        """
        Args:
            g_scal: Scalar drift function x -> g_scal(x)
            sigma: Diffusion coefficient
            x_grid: State grid (N_x,)
            t_grid: Time grid (N_t,)
            config: Score table configuration
            device: torch device
        """
        self.g_scal = g_scal
        self.sigma = sigma
        self.x_grid = x_grid.to(device)
        self.t_grid = t_grid.to(device)
        self.config = config
        self.device = device
        
        # Precompute grid spacing
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dt = self.t_grid[1] - self.t_grid[0]
        
        # Check CFL condition for stability
        max_drift = torch.abs(self.g_scal(self.x_grid)).max().item()
        cfl = (max_drift * self.dt) / self.dx
        if cfl > 0.25:  # Stricter criterion to prevent oscillations
            # warnings.warn(
            #     f"CFL number {cfl:.3f} is too large for stability. "
            #     f"Reduce dt or increase N_x. Recommended: CFL < 0.25"
            # )
            print(f"[red] CFL number: {cfl:.3f} [/red]")
        else:
            print(f"[green]CFL number: {cfl:.3f} [/green]")
    
    def _initialize_delta_approx(self, x0: float) -> torch.Tensor:
        """
        Approximate δ(x - x0) with narrow Gaussian.
        
        Args:
            x0: Initial value (scalar)
            
        Returns:
            p_0: Initial density on x_grid, shape (N_x,)
        """
        std = self.config.initial_gaussian_std
        p_0 = torch.exp(-0.5 * ((self.x_grid - x0) / std) ** 2)
        
        # Check for zero mass (x0 outside grid or too narrow Gaussian)
        total_mass = p_0.sum() * self.dx
        if total_mass < 1e-10:
            raise ScoreTableNumericalError(
                f"Initial condition has zero mass for x0={x0:.4f}. "
                f"x0 is outside grid range [{self.x_grid[0].item():.4f}, {self.x_grid[-1].item():.4f}] "
                f"or initial_gaussian_std={std} is too small."
            )
        
        # Normalize
        p_0 = p_0 / total_mass
        
        return p_0
    
    def _compute_flux(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute conservative flux at cell interfaces.
        
        Flux: F[j+1/2] = g_scal(x[j])·p[j] - (σ²/2)·∂_x p[j]
        
        Args:
            p: Density on x_grid, shape (N_x,)
            
        Returns:
            flux: Flux at interfaces, shape (N_x+1,)
        """
        N_x = len(self.x_grid)
        flux = torch.zeros(N_x + 1, device=self.device)
        
        # Check input
        if torch.isinf(p).any():
            raise ScoreTableNumericalError("Inf in density before flux computation")
        if torch.isnan(p).any():
            raise ScoreTableNumericalError("NaN in density before flux computation")
        
        # Compute drift at grid points
        g_vals = self.g_scal(self.x_grid)
        
        # Check drift function
        if torch.isinf(g_vals).any():
            bad_idx = torch.where(torch.isinf(g_vals))[0][0]
            x_bad = self.x_grid[bad_idx].item()
            raise ScoreTableNumericalError(
                f"Inf in drift function g(x) at x={x_bad:.4f}. "
                f"Grid range: [{self.x_grid[0].item():.4f}, {self.x_grid[-1].item():.4f}]"
            )
        
        # Drift contribution (at cell centers, then interpolate to faces)
        drift_term = g_vals * p
        
        # Check drift term
        if torch.isinf(drift_term).any():
            bad_idx = torch.where(torch.isinf(drift_term))[0][0]
            raise ScoreTableNumericalError(
                f"Inf in drift_term at index {bad_idx}: "
                f"g={g_vals[bad_idx].item():.4e}, p={p[bad_idx].item():.4e}"
            )
        
        # Diffusion coefficient
        diff_coeff = 0.5 * self.sigma ** 2
        
        # Interior fluxes (j=1 to N_x-1)
        for j in range(1, N_x):
            # Drift: average of neighbors
            drift_flux = 0.5 * (drift_term[j-1] + drift_term[j])
            
            # Diffusion: central difference
            diff_flux = -diff_coeff * (p[j] - p[j-1]) / self.dx
            
            flux[j] = drift_flux + diff_flux
        
        # Boundary fluxes
        if self.config.boundary_condition == "zero_flux":
            flux[0] = 0.0
            flux[N_x] = 0.0
        elif self.config.boundary_condition == "absorbing":
            # Absorbing: flux removes probability
            flux[0] = drift_term[0] - diff_coeff * p[0] / self.dx
            flux[N_x] = drift_term[-1] + diff_coeff * p[-1] / self.dx
        else:
            raise ValueError(f"Unknown boundary condition: {self.config.boundary_condition}")
        
        return flux
    
    def _time_step(self, p: torch.Tensor) -> torch.Tensor:
        """
        Advance density by one time step.
        
        Dispatches to appropriate scheme based on config.
        
        Args:
            p: Current density, shape (N_x,)
            
        Returns:
            p_next: Density at next time step, shape (N_x,)
        """
        scheme = self.config.time_stepping_scheme
        
        if scheme == "explicit_euler":
            return self._time_step_explicit_euler(p)
        elif scheme == "semi_implicit":
            return self._time_step_semi_implicit(p)
        elif scheme == "tamed_euler":
            return self._time_step_tamed_euler(p)
        else:
            raise ValueError(f"Unknown time stepping scheme: {scheme}")
    
    def _time_step_explicit_euler(self, p: torch.Tensor) -> torch.Tensor:
        """
        Explicit Euler time step (original method).
        
        Pros: Simple, conservative
        Cons: Unstable for superlinear drift, requires tiny dt
        """
        flux = self._compute_flux(p)
        
        # Check flux for problems
        if torch.isinf(flux).any():
            raise ScoreTableNumericalError(
                "Inf detected in flux computation. "
                "Likely cause: drift values too large (check CFL condition)"
            )
        
        # Conservative update
        p_next = p - (self.dt / self.dx) * (flux[1:] - flux[:-1])
        
        # Check for numerical issues
        if torch.isnan(p_next).any():
            raise ScoreTableNumericalError("NaN detected in density during time evolution")
        
        if torch.isinf(p_next).any():
            max_p = p.max().item()
            max_flux_diff = (flux[1:] - flux[:-1]).abs().max().item()
            raise ScoreTableNumericalError(
                f"Inf detected in density during time evolution. "
                f"max(p)={max_p:.2e}, max(flux_diff)={max_flux_diff:.2e}, "
                f"CFL too large - numerical instability"
            )
        
        # Ensure non-negativity
        p_next = torch.clamp(p_next, min=0.0)
        
        return p_next
    
    def _time_step_semi_implicit(self, p: torch.Tensor) -> torch.Tensor:
        """
        Semi-implicit (IMEX) time step.
        
        Drift (advection): explicit
        Diffusion: implicit (Crank-Nicolson)
        
        Solves: (I - 0.5*dt*D) p^{n+1} = (I + 0.5*dt*D) p^n - dt * div(g*p^n)
        
        Pros: Much more stable, allows larger dt
        Cons: Requires tridiagonal solve
        """
        N_x = len(self.x_grid)
        
        # Explicit drift term
        g_vals = self.g_scal(self.x_grid)
        drift_term = g_vals * p
        
        # Compute drift flux divergence: -d/dx(g*p)
        drift_div = torch.zeros(N_x, device=self.device)
        drift_div[1:-1] = -(drift_term[2:] - drift_term[:-2]) / (2 * self.dx)
        drift_div[0] = -(drift_term[1] - drift_term[0]) / self.dx
        drift_div[-1] = -(drift_term[-1] - drift_term[-2]) / self.dx
        
        # Diffusion coefficient
        D = 0.5 * self.sigma ** 2
        
        # Build tridiagonal system for implicit diffusion
        # (I - 0.5*dt*D*Laplacian) p^{n+1} = RHS
        
        alpha = 0.5 * D * self.dt / (self.dx ** 2)
        
        # Tridiagonal matrix coefficients
        lower = -alpha * torch.ones(N_x - 1, device=self.device)
        diag = (1 + 2 * alpha) * torch.ones(N_x, device=self.device)
        upper = -alpha * torch.ones(N_x - 1, device=self.device)
        
        # Boundary conditions
        if self.config.boundary_condition == "zero_flux":
            # Neumann: no flux at boundaries
            diag[0] = 1 + alpha
            diag[-1] = 1 + alpha
        elif self.config.boundary_condition == "absorbing":
            # Dirichlet: p = 0 at boundaries
            diag[0] = 1
            diag[-1] = 1
            upper[0] = 0
            lower[-1] = 0
        
        # RHS: explicit diffusion + drift
        # (I + 0.5*dt*D*Laplacian) p^n + dt * drift_div
        rhs = p.clone()
        rhs[1:-1] += alpha * (p[2:] - 2 * p[1:-1] + p[:-2])
        
        if self.config.boundary_condition == "zero_flux":
            rhs[0] += alpha * (p[1] - p[0])
            rhs[-1] += alpha * (p[-2] - p[-1])
        
        rhs += self.dt * drift_div
        
        # Solve tridiagonal system
        p_next = self._solve_tridiagonal(lower, diag, upper, rhs)
        
        # Check for issues
        if torch.isnan(p_next).any():
            raise ScoreTableNumericalError("NaN in semi-implicit solver")
        if torch.isinf(p_next).any():
            raise ScoreTableNumericalError("Inf in semi-implicit solver")
        
        # Ensure non-negativity
        p_next = torch.clamp(p_next, min=0.0)
        
        return p_next
    
    def _time_step_tamed_euler(self, p: torch.Tensor) -> torch.Tensor:
        """
        Tamed Euler time step.
        
        Caps the drift when it becomes too large to ensure stability.
        
        g_tamed = g / (1 + dt * |g| / dx)
        
        Pros: Explicit, provably stable
        Cons: Adds artificial damping, less accurate
        """
        # Compute drift
        g_vals = self.g_scal(self.x_grid)
        
        # Taming factor
        taming_factor = 1.0 / (1.0 + self.dt * torch.abs(g_vals) / self.dx)
        g_tamed = g_vals * taming_factor
        
        # Use tamed drift in flux
        drift_term = g_tamed * p
        
        # Diffusion
        diff_coeff = 0.5 * self.sigma ** 2
        
        # Conservative flux update
        flux = torch.zeros(len(self.x_grid) + 1, device=self.device)
        
        for j in range(1, len(self.x_grid)):
            drift_flux = 0.5 * (drift_term[j-1] + drift_term[j])
            diff_flux = -diff_coeff * (p[j] - p[j-1]) / self.dx
            flux[j] = drift_flux + diff_flux
        
        # Boundary fluxes (same as before)
        if self.config.boundary_condition == "zero_flux":
            flux[0] = 0.0
            flux[-1] = 0.0
        
        p_next = p - (self.dt / self.dx) * (flux[1:] - flux[:-1])
        
        # Check
        if torch.isnan(p_next).any():
            raise ScoreTableNumericalError("NaN in tamed Euler")
        if torch.isinf(p_next).any():
            raise ScoreTableNumericalError("Inf in tamed Euler")
        
        p_next = torch.clamp(p_next, min=0.0)
        
        return p_next
    
    def _solve_tridiagonal(self, lower: torch.Tensor, diag: torch.Tensor, 
                          upper: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """
        Solve tridiagonal system: A x = rhs
        
        Uses Thomas algorithm (stable for diagonally dominant matrices).
        
        Args:
            lower: Lower diagonal, length N-1
            diag: Main diagonal, length N
            upper: Upper diagonal, length N-1
            rhs: Right hand side, length N
            
        Returns:
            x: Solution, length N
        """
        N = len(diag)
        
        # Forward sweep
        c = torch.zeros(N - 1, device=self.device)
        d = torch.zeros(N, device=self.device)
        
        c[0] = upper[0] / diag[0]
        d[0] = rhs[0] / diag[0]
        
        for i in range(1, N - 1):
            denom = diag[i] - lower[i-1] * c[i-1]
            c[i] = upper[i] / denom
            d[i] = (rhs[i] - lower[i-1] * d[i-1]) / denom
        
        d[-1] = (rhs[-1] - lower[-1] * d[-2]) / (diag[-1] - lower[-1] * c[-2])
        
        # Backward sweep
        x = torch.zeros(N, device=self.device)
        x[-1] = d[-1]
        
        for i in range(N - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        
        return x
    
    def solve(self, x0: float) -> torch.Tensor:
        """
        Solve forward equation from initial condition at x0.
        
        Args:
            x0: Initial value (scalar in [x0_min, x0_max])
            
        Returns:
            p_history: Density evolution, shape (N_t, N_x)
        """
        N_t = len(self.t_grid)
        N_x = len(self.x_grid)
        
        p_history = torch.zeros(N_t, N_x, device=self.device)
        
        # Initialize
        p = self._initialize_delta_approx(x0)
        p_history[0] = p
        
        # Time evolution
        for i in range(1, N_t):
            p = self._time_step(p)
            p_history[i] = p
        
        return p_history


class ScoreTable:
    """
    Build and store precomputed conditional score table.
    
    Table S[k,i,j] ≈ ∂_x log p(t_i, x_j | x0_k)
    """
    
    def __init__(
        self,
        process,  # SuperlinearLangevin instance
        config: ScoreTableConfig,
        device: str = "cpu",
    ):
        """
        Args:
            process: ItoProcess with drift, diffusion, alpha, c_alpha, c_0, sigma
            config: Score table configuration
            device: torch device
        """
        self.process = process
        self.config = config
        self.device = device
        
        # Will be populated by build()
        self.S = None
        self.p_table = None  # Density table p(x,t|x0) - optional, for analytics
        self.x0_grid = None
        self.x_grid = None
        self.t_grid = None
        self.interpolator = None
    
    def _build_grids(self):
        """Create uniform grids for x0, x, and t."""
        self.x0_grid = torch.linspace(
            self.config.x0_min,
            self.config.x0_max,
            self.config.N_x0,
            device=self.device
        )
        
        self.x_grid = torch.linspace(
            self.config.x_min,
            self.config.x_max,
            self.config.N_x,
            device=self.device
        )
        
        self.t_grid = torch.linspace(
            0.0,
            self.config.T,
            self.config.N_t,
            device=self.device
        )
    
    def _extract_g_scal(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Extract scalar drift function from process.
        
        Returns:
            g_scal: Function x -> -(c_α |x|^α sign(x) + c_0 x)
        """
        alpha = self.process.alpha
        c_alpha = self.process.c_alpha
        c_0 = self.process.c_0
        
        def g_scal(x: torch.Tensor) -> torch.Tensor:
            power_term = torch.pow(torch.abs(x), alpha) * torch.sign(x)
            return -(c_alpha * power_term + c_0 * x)
        
        return g_scal
    
    def _compute_score_table(self, p_table: torch.Tensor) -> torch.Tensor:
        """
        Compute score via finite differences on log-density.
        
        Args:
            p_table: Density table, shape (N_x0, N_t, N_x)
            
        Returns:
            S: Score table, shape (N_x0, N_t, N_x)
        """
        eps = self.config.density_clamp_eps
        dx = self.x_grid[1] - self.x_grid[0]
        
        # Clamp for numerical stability
        p_clamp = torch.clamp(p_table, min=eps)
        log_p = torch.log(p_clamp)
        
        N_x0, N_t, N_x = p_table.shape
        S = torch.zeros_like(p_table)
        
        # Central differences for interior points
        S[:, :, 1:-1] = (log_p[:, :, 2:] - log_p[:, :, :-2]) / (2 * dx)
        
        # One-sided differences for boundaries
        S[:, :, 0] = (log_p[:, :, 1] - log_p[:, :, 0]) / dx
        S[:, :, -1] = (log_p[:, :, -1] - log_p[:, :, -2]) / dx
        
        return S
    
    def build(self):
        """
        Build the score table by solving PDE for each x0.
        
        This is the main computational routine. Progress is printed.
        """
        print(f"Building score table:")
        print(f"  Grid: {self.config.N_x0} x {self.config.N_t} x {self.config.N_x}")
        print(f"  Domain: x0 ∈ [{self.config.x0_min}, {self.config.x0_max}]")
        print(f"           x ∈ [{self.config.x_min}, {self.config.x_max}]")
        print(f"           t ∈ [0, {self.config.T}]")
        
        # Build grids
        self._build_grids()
        
        # Extract drift function
        g_scal = self._extract_g_scal()
        
        # Initialize storage
        p_table = torch.zeros(
            self.config.N_x0,
            self.config.N_t,
            self.config.N_x,
            device=self.device
        )
        
        # Build PDE solver
        pde_solver = PDESolver1D(
            g_scal=g_scal,
            sigma=self.process.sigma,
            x_grid=self.x_grid,
            t_grid=self.t_grid,
            config=self.config,
            device=self.device,
        )
        
        # Solve for each x0
        for k, x0_k in enumerate(self.x0_grid):
            if (k + 1) % max(1, self.config.N_x0 // 10) == 0:
                print(f"  Solving for x0[{k+1}/{self.config.N_x0}]...", flush=True)
            
            p_table[k] = pde_solver.solve(x0_k.item())
        
        # Check density table
        if torch.isnan(p_table).any():
            raise ScoreTableNumericalError("NaN detected in density table")
        
        if torch.isinf(p_table).any():
            raise ScoreTableNumericalError("Inf detected in density table")
        
        print("  Computing scores from densities...")
        
        # Store density table for analytics
        self.p_table = p_table.clone()
        
        # Compute scores
        self.S = self._compute_score_table(p_table)
        
        # Check score table
        if torch.isnan(self.S).any():
            raise ScoreTableNumericalError("NaN detected in score table")
        
        if torch.isinf(self.S).any():
            raise ScoreTableNumericalError("Inf detected in score table")
        
        print("  Building interpolator...")
        
        # Build interpolator
        self.interpolator = ScoreInterpolator(self)
        
        print("Score table built successfully.")
    
    def save(self, path: str, save_density: bool = True):
        """
        Save score table to disk.
        
        Args:
            path: File path (will create parent directories)
            save_density: If True, also save the density table p(x,t|x0) (default: True)
                         This adds memory but enables analytics on the density.
        """
        if self.S is None:
            raise RuntimeError("Score table not built. Call build() first.")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        save_dict = {
            'S': self.S.cpu(),
            'x0_grid': self.x0_grid.cpu(),
            'x_grid': self.x_grid.cpu(),
            't_grid': self.t_grid.cpu(),
            'config': self.config.to_dict(),
        }
        
        # Optionally include density table
        if save_density and self.p_table is not None:
            save_dict['p_table'] = self.p_table.cpu()
            size_mb = self.p_table.numel() * 4 / 1e6  # float32 = 4 bytes
            print(f"  Including density table in save (adds ~{size_mb:.1f} MB)")
        
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str, process) -> ScoreTable:
        """
        Load score table from disk.
        
        Args:
            path: File path
            process: Process instance (for validation)
            
        Returns:
            Loaded ScoreTable instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Score table not found at {path}")
        
        data = torch.load(path)
        
        # Reconstruct config
        config = ScoreTableConfig.from_dict(data['config'])
        
        # Validate process match
        if not config.validate_process_match(process):
            raise ValueError(
                f"Score table at {path} was built with different process parameters. "
                f"Expected: alpha={process.alpha}, c_alpha={process.c_alpha}, "
                f"c_0={process.c_0}, sigma={process.sigma}, T={process.T}. "
                f"Got: alpha={config.alpha}, c_alpha={config.c_alpha}, "
                f"c_0={config.c_0}, sigma={config.sigma}, T={config.T}. "
                f"Rebuild the table or use matching parameters."
            )
        
        # Create instance
        table = cls(process, config, device=process.device)
        
        # Load data
        table.S = data['S'].to(process.device)
        table.x0_grid = data['x0_grid'].to(process.device)
        table.x_grid = data['x_grid'].to(process.device)
        table.t_grid = data['t_grid'].to(process.device)
        
        # Load density table if it was saved
        if 'p_table' in data:
            table.p_table = data['p_table'].to(process.device)
            size_mb = table.p_table.numel() * 4 / 1e6
            print(f"  Loaded density table (~{size_mb:.1f} MB)")
        else:
            print(f"  Note: Density table not found in saved file (score-only mode)")
        
        # Build interpolator
        table.interpolator = ScoreInterpolator(table)
        
        return table


class ScoreInterpolator:
    """
    Vectorized trilinear interpolation for score lookups.
    
    Query: (x, t, x0) -> score, all vectorized over (B, d) pixels.
    """
    
    def __init__(self, score_table: ScoreTable):
        """
        Args:
            score_table: Built ScoreTable instance
        """
        self.S = score_table.S
        self.x0_grid = score_table.x0_grid
        self.x_grid = score_table.x_grid
        self.t_grid = score_table.t_grid
        self.device = score_table.device
        
        # Precompute grid info
        self.N_x0 = len(self.x0_grid)
        self.N_x = len(self.x_grid)
        self.N_t = len(self.t_grid)
        
        self.x0_min = self.x0_grid[0].item()
        self.x0_max = self.x0_grid[-1].item()
        self.x_min = self.x_grid[0].item()
        self.x_max = self.x_grid[-1].item()
        self.t_min = self.t_grid[0].item()
        self.t_max = self.t_grid[-1].item()
    
    def _find_indices(
        self, 
        values: torch.Tensor, 
        grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find grid indices and interpolation weights.
        
        Args:
            values: Query values, shape (N,)
            grid: Grid points, shape (M,)
            
        Returns:
            indices: Lower grid indices, shape (N,), clamped to [0, M-2]
            weights: Interpolation weights in [0, 1], shape (N,)
        """
        # Clamp values to grid range
        grid_min = grid[0]
        grid_max = grid[-1]
        values_clamp = torch.clamp(values, grid_min, grid_max)
        
        # Use searchsorted to find indices
        indices = torch.searchsorted(grid, values_clamp) - 1
        indices = torch.clamp(indices, 0, len(grid) - 2)
        
        # Compute interpolation weights
        lower_vals = grid[indices]
        upper_vals = grid[indices + 1]
        weights = (values_clamp - lower_vals) / (upper_vals - lower_vals + 1e-10)
        weights = torch.clamp(weights, 0.0, 1.0)
        
        return indices, weights
    
    def _trilinear_interp(
        self,
        k_idx: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        k_w: torch.Tensor,
        i_w: torch.Tensor,
        j_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Trilinear interpolation in 3D table.
        
        Args:
            k_idx, i_idx, j_idx: Lower indices in each dimension, shape (N,)
            k_w, i_w, j_w: Interpolation weights, shape (N,)
            
        Returns:
            Interpolated values, shape (N,)
        """
        # Get 8 corner values
        c000 = self.S[k_idx, i_idx, j_idx]
        c001 = self.S[k_idx, i_idx, j_idx + 1]
        c010 = self.S[k_idx, i_idx + 1, j_idx]
        c011 = self.S[k_idx, i_idx + 1, j_idx + 1]
        c100 = self.S[k_idx + 1, i_idx, j_idx]
        c101 = self.S[k_idx + 1, i_idx, j_idx + 1]
        c110 = self.S[k_idx + 1, i_idx + 1, j_idx]
        c111 = self.S[k_idx + 1, i_idx + 1, j_idx + 1]
        
        # Trilinear interpolation formula
        c00 = c000 * (1 - j_w) + c001 * j_w
        c01 = c010 * (1 - j_w) + c011 * j_w
        c10 = c100 * (1 - j_w) + c101 * j_w
        c11 = c110 * (1 - j_w) + c111 * j_w
        
        c0 = c00 * (1 - i_w) + c01 * i_w
        c1 = c10 * (1 - i_w) + c11 * i_w
        
        result = c0 * (1 - k_w) + c1 * k_w
        
        return result
    
    def query(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        x0: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized score lookup.
        
        Args:
            x: Noisy pixel values, shape (B, d)
            t: Time values, shape (B,)
            x0: Clean pixel values, shape (B, d)
            
        Returns:
            Conditional scores, shape (B, d)
        """
        B, d = x.shape
        
        # Flatten to (B*d,) for vectorized processing
        x_flat = x.reshape(-1)
        x0_flat = x0.reshape(-1)
        t_flat = t.unsqueeze(1).expand(B, d).reshape(-1)
        
        # Find indices and weights
        k_idx, k_w = self._find_indices(x0_flat, self.x0_grid)
        i_idx, i_w = self._find_indices(t_flat, self.t_grid)
        j_idx, j_w = self._find_indices(x_flat, self.x_grid)
        
        # Interpolate
        scores_flat = self._trilinear_interp(k_idx, i_idx, j_idx, k_w, i_w, j_w)
        
        # Reshape back
        return scores_flat.reshape(B, d)


def build_score_table(
    process,
    config: ScoreTableConfig,
    save_path: Optional[str] = None,
    save_density: bool = True
) -> ScoreTable:
    
    table = ScoreTable(process, config, device=process.device)
    table.build()
    
    if save_path:
        table.save(save_path, save_density=save_density)
    
    return table