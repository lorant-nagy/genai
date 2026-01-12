# config_factory.py
import yaml
import argparse
from pathlib import Path
from itertools import product
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import hashlib

def is_sweep_value(value: Any) -> bool:
    """Check if a value is a list meant for sweeping (not a config list)."""
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        return False
    # If it's a list of lists, it's a sweep over list values
    # e.g., [[1,2,3], [1,2,3,4]] means sweep over two channel_mult options
    if all(isinstance(x, list) for x in value):
        return True
    # If it's a list of dicts, it's a sweep over dict values
    if all(isinstance(x, dict) for x in value):
        return True
    # If it's a list of primitives/strings, it's a sweep
    if all(isinstance(x, (int, float, str, bool, type(None))) for x in value):
        return True
    return False

def find_sweep_params(config: Dict, prefix: str = "") -> List[Tuple[str, List]]:
    """
    Recursively find all parameters that should be swept.
    Returns list of (dotted_path, values) tuples.
    """
    sweep_params = []
    
    for key, value in config.items():
        current_path = f"{prefix}.{key}" if prefix else key
        
        if is_sweep_value(value):
            sweep_params.append((current_path, value))
        elif isinstance(value, dict):
            # Recurse into nested dicts
            sweep_params.extend(find_sweep_params(value, current_path))
    
    return sweep_params

def set_nested_value(config: Dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dotted path notation."""
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value

def get_nested_value(config: Dict, path: str) -> Any:
    """Get a value from a nested dict using dotted path notation."""
    keys = path.split('.')
    current = config
    for key in keys:
        current = current[key]
    return current

def generate_config_name(sweep_params: List[Tuple[str, Any]]) -> str:
    """
    Generate a unique, readable config name from sweep parameters.
    """
    parts = []
    for path, value in sweep_params:
        # Simplify the path for readability
        key = path.split('.')[-1]
        
        # Format the value
        if isinstance(value, list):
            # For lists (like channel_mult), join with underscores
            value_str = '_'.join(map(str, value))
        elif isinstance(value, float):
            value_str = f"{value:.4g}".replace('.', 'p')
        elif isinstance(value, str):
            value_str = value.replace('/', '_').replace(' ', '_')
        else:
            value_str = str(value)
        
        parts.append(f"{key}_{value_str}")
    
    # Join with double underscores for clarity
    name = "__".join(parts)
    
    # If name is too long, add a hash
    if len(name) > 200:
        hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
        name = name[:180] + "__" + hash_suffix
    
    return name

def create_configs(factory_config: Dict) -> List[Tuple[str, Dict]]:
    """
    Generate all config combinations from a factory config.
    Returns list of (config_name, config_dict) tuples.
    """
    # Find all parameters to sweep
    sweep_params = find_sweep_params(factory_config)
    
    if not sweep_params:
        # No sweep parameters, return single config
        return [("config_0", factory_config)]
    
    # Generate all combinations
    param_names = [path for path, _ in sweep_params]
    param_values = [values for _, values in sweep_params]
    
    configs = []
    for i, combination in enumerate(product(*param_values)):
        # Deep copy the base config
        config = deepcopy(factory_config)
        
        # Set all swept parameters
        sweep_settings = []
        for path, value in zip(param_names, combination):
            set_nested_value(config, path, value)
            sweep_settings.append((path, value))
        
        # Generate name
        config_name = generate_config_name(sweep_settings)
        configs.append((config_name, config))
    
    return configs

def save_configs(configs: List[Tuple[str, Dict]], output_dir: Path) -> None:
    """Save all generated configs to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a manifest file
    manifest = {
        'n_configs': len(configs),
        'configs': [name for name, _ in configs]
    }
    
    with open(output_dir / 'manifest.yml', 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    
    # Save each config
    for config_name, config in configs:
        config_path = output_dir / f"{config_name}.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created: {config_path}")
    
    print(f"\nTotal configs generated: {len(configs)}")
    print(f"Manifest saved to: {output_dir / 'manifest.yml'}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate config files from a factory config with parameter sweeps'
    )
    parser.add_argument(
        'factory_config',
        type=str,
        help='Path to factory config YAML file'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for generated configs'
    )
    
    args = parser.parse_args()
    
    # Load factory config
    factory_path = Path(args.factory_config)
    if not factory_path.exists():
        raise FileNotFoundError(f"Factory config not found: {factory_path}")
    
    with open(factory_path, 'r') as f:
        factory_config = yaml.safe_load(f)
    
    # Generate configs
    print(f"Loading factory config from: {factory_path}")
    configs = create_configs(factory_config)
    
    # Save configs
    output_dir = Path(args.output_dir)
    save_configs(configs, output_dir)

if __name__ == "__main__":
    main()