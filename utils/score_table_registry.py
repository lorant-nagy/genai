"""
Score Table Registry System

Manages score tables with unique IDs based on their parameters.
Automatically checks if a table exists before creating a new one.

Dependencies tracked:
1. Process parameters: alpha, c_alpha, c_0, sigma, T, t0
2. Grid discretization: N_x0, N_x, N_t, x_min, x_max, x0_min, x0_max
3. PDE solver parameters: initial_gaussian_std, density_clamp_eps, 
   boundary_condition, time_stepping_scheme
"""

import os
import json
import hashlib
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# Make rich optional for standalone use
try:
    from rich import print
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


@dataclass
class ScoreTableIdentity:
    """
    Complete set of parameters that uniquely identify a score table.
    """
    # Process parameters
    alpha: float
    c_alpha: float
    c_0: float
    sigma: float
    T: float
    t0: float = 0.0
    
    # Grid discretization
    N_x0: int = 400
    N_x: int = 200
    N_t: int = 800
    x_min: float = -4.5
    x_max: float = 4.5
    x0_min: float = -1.2
    x0_max: float = 1.2
    
    # PDE solver parameters
    initial_gaussian_std: float = 0.03
    density_clamp_eps: float = 1e-12
    boundary_condition: str = "zero_flux"
    time_stepping_scheme: str = "semi_implicit"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ScoreTableIdentity':
        """Reconstruct from dictionary."""
        return cls(**d)
    
    @classmethod
    def from_process_and_config(cls, process, score_table_params: Dict) -> 'ScoreTableIdentity':
        """
        Extract identity from a process instance and score_table_params dict.
        
        Args:
            process: SuperlinearLangevin instance
            score_table_params: Dict with grid/solver parameters
            
        Note: n_workers is excluded as it's a runtime parameter, not part of table identity
        """
        return cls(
            # Process parameters
            alpha=process.alpha,
            c_alpha=process.c_alpha,
            c_0=process.c_0,
            sigma=process.sigma,
            T=process.T,
            t0=process.t0,
            # Grid parameters (exclude n_workers)
            N_x0=score_table_params.get('N_x0', 400),
            N_x=score_table_params.get('N_x', 200),
            N_t=score_table_params.get('N_t', 800),
            x_min=score_table_params.get('x_min', -4.5),
            x_max=score_table_params.get('x_max', 4.5),
            x0_min=score_table_params.get('x0_min', -1.2),
            x0_max=score_table_params.get('x0_max', 1.2),
            # PDE solver parameters
            initial_gaussian_std=score_table_params.get('initial_gaussian_std', 0.03),
            density_clamp_eps=score_table_params.get('density_clamp_eps', 1e-12),
            boundary_condition=score_table_params.get('boundary_condition', 'zero_flux'),
            time_stepping_scheme=score_table_params.get('time_stepping_scheme', 'semi_implicit'),
        )
    
    def compute_hash(self) -> str:
        """
        Compute deterministic hash of all parameters.
        Uses MD5 for reasonable length and collision resistance.
        """
        # Convert to JSON string with sorted keys for determinism
        json_str = json.dumps(self.to_dict(), sort_keys=True)
        hash_obj = hashlib.md5(json_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for readability
    
    def matches(self, other: 'ScoreTableIdentity', tolerance: float = 1e-6) -> bool:
        """
        Check if two identities match within tolerance.
        
        Args:
            other: Another ScoreTableIdentity
            tolerance: Numerical tolerance for float comparisons
        """
        # Check integers (must match exactly)
        if (self.N_x0 != other.N_x0 or 
            self.N_x != other.N_x or 
            self.N_t != other.N_t):
            return False
        
        # Check strings (must match exactly)
        if (self.boundary_condition != other.boundary_condition or
            self.time_stepping_scheme != other.time_stepping_scheme):
            return False
        
        # Check floats (within tolerance)
        float_params = [
            'alpha', 'c_alpha', 'c_0', 'sigma', 'T', 't0',
            'x_min', 'x_max', 'x0_min', 'x0_max',
            'initial_gaussian_std', 'density_clamp_eps'
        ]
        
        for param in float_params:
            if abs(getattr(self, param) - getattr(other, param)) > tolerance:
                return False
        
        return True


class ScoreTableRegistry:
    """
    Manages a registry of score tables with their parameters and file locations.
    Registry is stored as a JSON file.
    """
    
    def __init__(self, registry_dir: str):
        """
        Args:
            registry_dir: Directory to store registry.json and score tables
        """
        # Set umask to ensure all created files are deletable
        os.umask(0o002)
        
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.tables_dir = self.registry_dir / "tables"
        self.tables_dir.mkdir(exist_ok=True)
        
        # Fix permissions on directories
        try:
            os.chmod(self.registry_dir, 0o775)
            os.chmod(self.tables_dir, 0o775)
        except Exception:
            pass
        
        # Load existing registry or create new one
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load registry from file or create empty one."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, sort_keys=True)
        
        # Fix permissions so file is deletable
        try:
            os.chmod(self.registry_file, 0o664)  # rw-rw-r--
            os.chmod(self.registry_dir, 0o775)   # rwxrwxr-x
            os.chmod(self.tables_dir, 0o775)     # rwxrwxr-x
        except Exception:
            pass
    
    def get_table_path(self, table_id: str) -> Path:
        """Get the file path for a given table ID."""
        return self.tables_dir / f"score_table_{table_id}.pt"
    
    def find_matching_table(
        self, 
        identity: ScoreTableIdentity,
        tolerance: float = 1e-6
    ) -> Optional[Tuple[str, Path]]:
        """
        Find an existing table that matches the given identity.
        
        Args:
            identity: ScoreTableIdentity to search for
            tolerance: Numerical tolerance for float comparisons
            
        Returns:
            (table_id, file_path) if found, None otherwise
        """
        for table_id, entry in self.registry.items():
            stored_identity = ScoreTableIdentity.from_dict(entry['identity'])
            if identity.matches(stored_identity, tolerance=tolerance):
                table_path = self.get_table_path(table_id)
                if table_path.exists():
                    return table_id, table_path
                else:
                    print(f"[yellow]Warning: Registry entry {table_id} exists but file missing[/yellow]")
        return None
    
    def register_table(
        self,
        identity: ScoreTableIdentity,
        description: str = ""
    ) -> Tuple[str, Path]:
        """
        Register a new score table in the registry.
        
        Args:
            identity: Complete parameter identity
            description: Optional human-readable description
            
        Returns:
            (table_id, file_path) for the new table
        """
        # Generate ID from hash
        table_id = identity.compute_hash()
        
        # Check if already exists
        if table_id in self.registry:
            print(f"[yellow]Table ID {table_id} already registered[/yellow]")
            return table_id, self.get_table_path(table_id)
        
        # Create registry entry
        entry = {
            'identity': identity.to_dict(),
            'description': description,
            'created_at': datetime.now().isoformat(),
        }
        
        self.registry[table_id] = entry
        self._save_registry()
        
        return table_id, self.get_table_path(table_id)
    
    def get_or_create_table(
        self,
        process,
        score_table_params: Dict,
        description: str = "",
        tolerance: float = 1e-6
    ) -> Tuple[str, Path, bool]:
        """
        Get existing table or register a new one.
        
        Args:
            process: SuperlinearLangevin process instance
            score_table_params: Dict with score table configuration
            description: Optional description for new tables
            tolerance: Numerical tolerance for matching
            
        Returns:
            (table_id, file_path, is_new) where is_new=True if table needs building
        """
        # Extract complete identity
        identity = ScoreTableIdentity.from_process_and_config(
            process, score_table_params
        )
        
        # Try to find existing table
        match = self.find_matching_table(identity, tolerance=tolerance)
        
        if match is not None:
            table_id, table_path = match
            print(f"[green]✓ Found existing score table: {table_id}[/green]")
            return table_id, table_path, False
        
        # Register new table
        table_id, table_path = self.register_table(identity, description)
        print(f"[blue]→ Registered new score table: {table_id}[/blue]")
        return table_id, table_path, True
    
    def list_tables(self) -> None:
        """Print a summary of all registered tables."""
        if not self.registry:
            print("[yellow]No tables registered[/yellow]")
            return
        
        print(f"[bold green]Score Table Registry[/bold green]")
        print(f"Location: {self.registry_dir}")
        print(f"\nRegistered tables: {len(self.registry)}")
        print("-" * 80)
        
        for table_id, entry in self.registry.items():
            identity = entry['identity']
            exists = self.get_table_path(table_id).exists()
            status = "[green]✓[/green]" if exists else "[red]✗[/red]"
            
            print(f"\n{status} ID: [cyan]{table_id}[/cyan]")
            
            if entry.get('description'):
                print(f"   Description: {entry['description']}")
            
            # Show key parameters
            print(f"   Process: α={identity['alpha']}, c_α={identity['c_alpha']}, "
                  f"c_0={identity['c_0']}, σ={identity['sigma']}, T={identity['T']}")
            print(f"   Grid: N_x0={identity['N_x0']}, N_x={identity['N_x']}, N_t={identity['N_t']}")
            print(f"   Solver: {identity['time_stepping_scheme']}, BC={identity['boundary_condition']}")
            
            if exists:
                # Get file size
                size_mb = self.get_table_path(table_id).stat().st_size / 1e6
                print(f"   File: {size_mb:.1f} MB")
    
    def remove_table(self, table_id: str, delete_file: bool = False) -> bool:
        """
        Remove a table from the registry.
        
        Args:
            table_id: Table ID to remove
            delete_file: If True, also delete the .pt file
            
        Returns:
            True if removed, False if not found
        """
        if table_id not in self.registry:
            print(f"[yellow]Table {table_id} not found in registry[/yellow]")
            return False
        
        # Remove from registry
        del self.registry[table_id]
        self._save_registry()
        
        # Optionally delete file
        if delete_file:
            table_path = self.get_table_path(table_id)
            if table_path.exists():
                table_path.unlink()
                print(f"[green]✓ Deleted file: {table_path}[/green]")
        
        print(f"[green]✓ Removed table {table_id} from registry[/green]")
        return True


def create_registry_from_config(results_dir: str) -> ScoreTableRegistry:
    """
    Convenience function to create a registry in the standard location.
    
    Args:
        results_dir: Base results directory (from config.env.results_dir)
    
    Returns:
        ScoreTableRegistry instance
    """
    registry_dir = os.path.join(results_dir, "score_tables")
    return ScoreTableRegistry(registry_dir)


# Example usage and tests
if __name__ == "__main__":
    # Test the registry system
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ScoreTableRegistry(tmpdir)
        
        # Create a mock identity
        identity1 = ScoreTableIdentity(
            alpha=1.2, c_alpha=0.5, c_0=0.0, sigma=1.0, T=5.5,
            N_x0=400, N_x=200, N_t=800,
            time_stepping_scheme="semi_implicit"
        )
        
        # Register it
        table_id1, path1 = registry.register_table(
            identity1, 
            description="Test table for alpha=1.2"
        )
        print(f"Registered: {table_id1}")
        
        # Try to find it
        match = registry.find_matching_table(identity1)
        if match:
            print(f"Found matching table: {match[0]}")
        
        # Create a slightly different identity (should not match)
        identity2 = ScoreTableIdentity(
            alpha=1.3,  # Different!
            c_alpha=0.5, c_0=0.0, sigma=1.0, T=5.5,
            N_x0=400, N_x=200, N_t=800,
            time_stepping_scheme="semi_implicit"
        )
        
        match2 = registry.find_matching_table(identity2)
        print(f"Match for different params: {match2}")  # Should be None
        
        # List all tables
        registry.list_tables()