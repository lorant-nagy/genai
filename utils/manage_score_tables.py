#!/usr/bin/env python3
"""
Score Table Registry Manager

Command-line utility to inspect and manage score tables.

Usage:
    python manage_score_tables.py list [--registry-dir PATH]
    python manage_score_tables.py info TABLE_ID [--registry-dir PATH]
    python manage_score_tables.py remove TABLE_ID [--delete-file] [--registry-dir PATH]
    python manage_score_tables.py clean [--dry-run] [--registry-dir PATH]

Examples:
    # List all registered tables
    python manage_score_tables.py list --registry-dir /run_logs/score_tables
    
    # Show details of a specific table
    python manage_score_tables.py info a1b2c3d4e5f6g7h8
    
    # Remove a table from registry (keeps file)
    python manage_score_tables.py remove a1b2c3d4e5f6g7h8
    
    # Remove a table and delete its file
    python manage_score_tables.py remove a1b2c3d4e5f6g7h8 --delete-file
    
    # Clean registry (remove entries for missing files)
    python manage_score_tables.py clean --registry-dir /run_logs/score_tables
"""

import argparse
import sys
from pathlib import Path
from rich import print
from rich.table import Table
from rich.console import Console

# Assuming this script is in the same directory as score_table_registry.py
# or that score_table_registry is importable
try:
    from score_table_registry import ScoreTableRegistry, ScoreTableIdentity
except ImportError:
    # If running as standalone script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    from score_table_registry import ScoreTableRegistry, ScoreTableIdentity


def cmd_list(args):
    """List all registered score tables."""
    registry = ScoreTableRegistry(args.registry_dir)
    
    if not registry.registry:
        print("[yellow]No tables registered[/yellow]")
        return
    
    console = Console()
    table = Table(title=f"Score Table Registry: {args.registry_dir}")
    
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Process Params", style="blue")
    table.add_column("Grid", style="magenta")
    
    for table_id, entry in registry.registry.items():
        identity = entry['identity']
        table_path = registry.get_table_path(table_id)
        exists = table_path.exists()
        
        status = "✓" if exists else "✗"
        size = f"{table_path.stat().st_size / 1e6:.1f} MB" if exists else "N/A"
        
        process_str = (f"α={identity['alpha']}, c_α={identity['c_alpha']}, "
                      f"c_0={identity['c_0']}, σ={identity['sigma']}, T={identity['T']}")
        
        grid_str = (f"N_x0={identity['N_x0']}, N_x={identity['N_x']}, "
                   f"N_t={identity['N_t']}")
        
        table.add_row(table_id, status, size, process_str, grid_str)
    
    console.print(table)
    print(f"\nTotal: {len(registry.registry)} tables registered")


def cmd_info(args):
    """Show detailed information about a specific table."""
    registry = ScoreTableRegistry(args.registry_dir)
    
    if args.table_id not in registry.registry:
        print(f"[red]Error: Table {args.table_id} not found in registry[/red]")
        sys.exit(1)
    
    entry = registry.registry[args.table_id]
    identity = ScoreTableIdentity.from_dict(entry['identity'])
    table_path = registry.get_table_path(args.table_id)
    
    print(f"\n[bold cyan]Score Table: {args.table_id}[/bold cyan]")
    print("=" * 80)
    
    # Status
    exists = table_path.exists()
    status_color = "green" if exists else "red"
    status_text = "EXISTS" if exists else "MISSING"
    print(f"Status: [{status_color}]{status_text}[/{status_color}]")
    print(f"Path: {table_path}")
    
    if exists:
        size_mb = table_path.stat().st_size / 1e6
        print(f"Size: {size_mb:.2f} MB")
    
    # Description
    if entry.get('description'):
        print(f"\nDescription: {entry['description']}")
    
    # Process parameters
    print("\n[bold]Process Parameters:[/bold]")
    print(f"  alpha (α):      {identity.alpha}")
    print(f"  c_alpha (c_α):  {identity.c_alpha}")
    print(f"  c_0:            {identity.c_0}")
    print(f"  sigma (σ):      {identity.sigma}")
    print(f"  T:              {identity.T}")
    print(f"  t0:             {identity.t0}")
    
    # Grid parameters
    print("\n[bold]Grid Discretization:[/bold]")
    print(f"  N_x0:           {identity.N_x0}")
    print(f"  N_x:            {identity.N_x}")
    print(f"  N_t:            {identity.N_t}")
    print(f"  x0 range:       [{identity.x0_min}, {identity.x0_max}]")
    print(f"  x range:        [{identity.x_min}, {identity.x_max}]")
    
    # PDE solver parameters
    print("\n[bold]PDE Solver Parameters:[/bold]")
    print(f"  Initial Gaussian std:  {identity.initial_gaussian_std}")
    print(f"  Density clamp eps:     {identity.density_clamp_eps}")
    print(f"  Boundary condition:    {identity.boundary_condition}")
    print(f"  Time stepping scheme:  {identity.time_stepping_scheme}")
    
    print("=" * 80 + "\n")


def cmd_remove(args):
    """Remove a table from the registry."""
    registry = ScoreTableRegistry(args.registry_dir)
    
    if args.table_id not in registry.registry:
        print(f"[red]Error: Table {args.table_id} not found in registry[/red]")
        sys.exit(1)
    
    # Confirm deletion if file will be deleted
    if args.delete_file:
        table_path = registry.get_table_path(args.table_id)
        if table_path.exists():
            print(f"[yellow]Warning: This will DELETE the file at {table_path}[/yellow]")
            confirm = input("Are you sure? (yes/no): ")
            if confirm.lower() != 'yes':
                print("[blue]Cancelled[/blue]")
                return
    
    # Remove
    registry.remove_table(args.table_id, delete_file=args.delete_file)


def cmd_clean(args):
    """Remove registry entries for missing files."""
    registry = ScoreTableRegistry(args.registry_dir)
    
    missing = []
    for table_id in registry.registry:
        table_path = registry.get_table_path(table_id)
        if not table_path.exists():
            missing.append(table_id)
    
    if not missing:
        print("[green]✓ All registered tables have files present[/green]")
        return
    
    print(f"[yellow]Found {len(missing)} entries with missing files:[/yellow]")
    for table_id in missing:
        print(f"  - {table_id}")
    
    if args.dry_run:
        print("\n[blue]Dry run - no changes made[/blue]")
        return
    
    confirm = input(f"\nRemove {len(missing)} entries from registry? (yes/no): ")
    if confirm.lower() != 'yes':
        print("[blue]Cancelled[/blue]")
        return
    
    for table_id in missing:
        registry.remove_table(table_id, delete_file=False)
    
    print(f"[green]✓ Cleaned {len(missing)} entries[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Manage score table registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--registry-dir',
        type=str,
        default='/run_logs/score_tables',
        help='Path to score table registry directory (default: /run_logs/score_tables)'
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # List command
    parser_list = subparsers.add_parser('list', help='List all registered tables')
    parser_list.set_defaults(func=cmd_list)
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Show detailed info about a table')
    parser_info.add_argument('table_id', help='Table ID (hash)')
    parser_info.set_defaults(func=cmd_info)
    
    # Remove command
    parser_remove = subparsers.add_parser('remove', help='Remove a table from registry')
    parser_remove.add_argument('table_id', help='Table ID (hash)')
    parser_remove.add_argument('--delete-file', action='store_true',
                              help='Also delete the .pt file')
    parser_remove.set_defaults(func=cmd_remove)
    
    # Clean command
    parser_clean = subparsers.add_parser('clean', help='Remove entries for missing files')
    parser_clean.add_argument('--dry-run', action='store_true',
                             help='Show what would be cleaned without making changes')
    parser_clean.set_defaults(func=cmd_clean)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()