"""
WandB utilities for automatic login and initialization.

This module handles:
- Automatic login detection and authentication
- WandB initialization with config
- Clean API for train.py to use

All functions assume wandb is always enabled.
"""

import os
import subprocess
import wandb
from rich import print


def check_and_login():
    """
    Check if logged in to WandB, and attempt to login if not.
    
    Tries to find API key in the following order:
    1. WANDB_API_KEY environment variable
    2. ~/.wandb_key file
    3. Interactive prompt
    
    Returns:
        bool: True if successfully logged in, False otherwise
    """
    try:
        # Check if already logged in
        result = subprocess.run(['wandb', 'status'], capture_output=True, text=True)
        if 'Logged in?' in result.stdout and 'True' in result.stdout:
            # Extract username if available
            for line in result.stdout.split('\n'):
                if 'Current user:' in line:
                    username = line.split(':')[-1].strip()
                    print(f"[green]✅ Logged in to WandB as: {username}[/green]")
                    break
            return True
    except Exception as e:
        print(f"[yellow]⚠️  Could not check WandB status: {e}[/yellow]")
    
    # Not logged in, try to login
    print("[yellow]⚠️  Not logged in to WandB. Attempting login...[/yellow]")
    
    # Try environment variable first
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if wandb_api_key:
        print("[blue]🔑 Found API key in WANDB_API_KEY environment variable[/blue]")
        try:
            wandb.login(key=wandb_api_key)
            print("[green]✅ Successfully logged in to WandB![/green]")
            return True
        except Exception as e:
            print(f"[red]❌ Login failed: {e}[/red]")
            return False
    
    # Try ~/.wandb_key file
    wandb_key_file = os.path.expanduser('~/.wandb_key')
    if os.path.exists(wandb_key_file):
        print(f"[blue]🔑 Found API key in {wandb_key_file}[/blue]")
        try:
            with open(wandb_key_file, 'r') as f:
                wandb_api_key = f.read().strip()
            wandb.login(key=wandb_api_key)
            print("[green]✅ Successfully logged in to WandB![/green]")
            return True
        except Exception as e:
            print(f"[red]❌ Login failed: {e}[/red]")
            return False
    
    # No automatic method found, prompt user
    print("[yellow]No API key found in environment or ~/.wandb_key[/yellow]")
    print("[yellow]Please login to WandB:[/yellow]")
    print("  1. Get your API key from: [blue]https://wandb.ai/authorize[/blue]")
    print("  2. Paste it when prompted below")
    print("")
    
    try:
        wandb.login()
        print("[green]✅ Successfully logged in to WandB![/green]")
        return True
    except Exception as e:
        print(f"[red]❌ Login failed: {e}[/red]")
        return False


def initialize_wandb(config, run_dir, petname_str):
    """
    Initialize WandB with the given configuration.
    
    Args:
        config: Configuration object with wandb section
        run_dir: Directory to save wandb files
        petname_str: Human-readable run name
    
    Raises:
        RuntimeError: If wandb initialization fails
    """
    # Try to login if needed
    if not check_and_login():
        raise RuntimeError("WandB login failed. Cannot continue without WandB.")
    
    # Get wandb settings from config (handle both dict and Cfg object)
    if hasattr(config, 'wandb'):
        wandb_config = config.wandb
        wandb_project = getattr(wandb_config, 'project', 'diffusion-training')
        wandb_entity = getattr(wandb_config, 'entity', None)
        wandb_mode = getattr(wandb_config, 'mode', 'online')
        wandb_notes = getattr(wandb_config, 'notes', '')
        custom_tags = getattr(wandb_config, 'tags', [])
    else:
        # Default values if no wandb section in config
        wandb_project = 'diffusion-training'
        wandb_entity = None
        wandb_mode = 'online'
        wandb_notes = ''
        custom_tags = []
    
    # Auto-generate tags from process and model
    auto_tags = []
    if hasattr(config, 'corruption') and hasattr(config.corruption, 'process_cls'):
        auto_tags.append(config.corruption.process_cls)
    if hasattr(config, 'model') and hasattr(config.model, 'cls'):
        auto_tags.append(config.model.cls)
    
    all_tags = auto_tags + custom_tags
    
    # Initialize wandb
    try:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=petname_str,
            config=config.to_dict(),
            dir=run_dir,
            tags=all_tags,
            notes=wandb_notes,
            mode=wandb_mode
        )
        print(f"[green]✅ WandB initialized: {wandb.run.url}[/green]")
    except Exception as e:
        raise RuntimeError(f"WandB initialization failed: {e}")


def log_metrics(metrics_dict):
    """
    Log metrics to WandB.
    
    Args:
        metrics_dict: Dictionary of metrics to log
    """
    if wandb.run is not None:
        wandb.log(metrics_dict)


def log_images(images_dict):
    """
    Log images to WandB.
    
    Args:
        images_dict: Dictionary of {name: image_path or wandb.Image}
    """
    if wandb.run is not None:
        wandb.log(images_dict)


def watch_model(model, log="all", log_freq=100):
    """
    Watch model gradients and parameters in WandB.
    
    Args:
        model: PyTorch model
        log: What to log ("gradients", "parameters", "all", or None)
        log_freq: Logging frequency
    """
    if wandb.run is not None:
        wandb.watch(model, log=log, log_freq=log_freq)


def save_file(filepath):
    """
    Save a file to WandB.
    
    Args:
        filepath: Path to file to save
    """
    if wandb.run is not None:
        wandb.save(filepath)


def finish_run():
    """
    Finish the WandB run.
    """
    if wandb.run is not None:
        wandb.finish()