import os
import re

from mlebench.utils import get_logger

logger = get_logger(__name__)

# Mapping from kwarg keys to the env vars they should auto-sync.
# config.yaml uses YAML anchors to set both kwargs (e.g. agent.steps) and env_vars
# (e.g. STEP_LIMIT) from the same anchor, but anchors resolve at parse time. When
# kwargs are overridden via CLI, the corresponding env_vars must be updated too
# because start.sh uses them (e.g. TIME_LIMIT_SECS for the timeout command).
KWARG_TO_ENV = {
    "agent.steps": "STEP_LIMIT",
    "agent.time_limit": "TIME_LIMIT_SECS",
}

_SECRET_PATTERN = re.compile(r"\$\{\{\s*secrets\.(\w+)\s*\}\}")


def parse_env_var_values(dictionary: dict) -> dict:
    """Replace ${{ secrets.NAME }} placeholders with actual environment variable values."""
    for key, value in dictionary.items():
        if not isinstance(value, str):
            continue
        match = _SECRET_PATTERN.match(value)
        if not match:
            continue
        env_var = match.group(1)
        if os.getenv(env_var) is None:
            raise ValueError(f"Environment variable `{env_var}` is not set!")
        dictionary[key] = os.getenv(env_var)
    return dictionary


def _coerce_value(value: str):
    """Try to convert a string to bool, int, or float; return as-is on failure."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_kv_pairs(pairs: list[str], coerce_types: bool = False) -> dict:
    """Parse a list of 'key=value' strings into a dict.

    If coerce_types is True, values are converted to bool/int/float when possible.
    """
    result = {}
    for kv in pairs:
        if "=" not in kv:
            logger.warning(f"Invalid key=value format, skipping: {kv}")
            continue
        key, value = kv.split("=", 1)
        result[key] = _coerce_value(value) if coerce_types else value
    return result


def apply_cli_overrides(agent, raw_kwargs: list[str], raw_env_vars: list[str]):
    """Merge --kwargs and --env-vars on top of config.yaml values.

    Returns (merged_kwargs, merged_env_vars).
    """
    extra_kwargs = parse_kv_pairs(raw_kwargs, coerce_types=True)
    extra_env_vars = parse_kv_pairs(raw_env_vars)

    if not extra_kwargs and not extra_env_vars:
        return agent.kwargs, agent.env_vars

    merged_kwargs = {**agent.kwargs, **extra_kwargs}
    merged_env_vars = {**agent.env_vars}

    # Auto-sync env_vars that mirror overridden kwargs (command line overrides config.yaml)
    for kwarg_key, env_key in KWARG_TO_ENV.items():
        if kwarg_key in extra_kwargs and env_key in merged_env_vars:
            logger.info(f"Syncing env_var {env_key}: {merged_env_vars[env_key]} -> {extra_kwargs[kwarg_key]}")
            merged_env_vars[env_key] = extra_kwargs[kwarg_key]

    # Explicit --env-vars take highest priority.
    merged_env_vars.update(extra_env_vars)

    return merged_kwargs, merged_env_vars
