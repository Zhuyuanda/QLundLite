import yaml
from typing import Any

def recursive_update(d: dict, u: dict) -> dict:
    """
    Recursively update dictionary `d` with values from dictionary `u`.
    If a value is a dictionary in both `d` and `u`, update recursively.
    Otherwise, overwrite the value in `d` with the value from `u`.

    Args:
        d: The dictionary to update.
        u: The dictionary with updates.

    Returns:
        The updated dictionary `d`.
    """
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = recursive_update(d[k], v)
        else:
            d[k] = v
    return d

def parse_dot_args(dot_args: list[str]) -> dict[str, Any]:
    """
    Parse a list of override arguments in the form 'key.subkey=value' into a nested dictionary.

    Args:
        dot_args: List of strings, each in the form 'key.subkey=value'.

    Returns:
        A nested dictionary representing the overrides.
    """
    overrides = {}
    for arg in dot_args:
        key, value = arg.split('=', 1)
        parts = [yaml.safe_load(p) for p in key.split('.')]
        current = overrides
        for p in parts[:-1]:
            current = current.setdefault(p, {})
        # Use yaml.safe_load to parse value into correct type (int, float, bool, etc.)
        current[parts[-1]] = yaml.safe_load(value)
    return overrides