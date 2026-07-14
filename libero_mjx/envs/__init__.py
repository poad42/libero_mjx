"""Environment registry for LIBERO MJX port."""

_ENV_REGISTRY: dict[str, type] = {}


def register_env(name: str):
    def decorator(cls):
        _ENV_REGISTRY[name] = cls
        return cls
    return decorator


def load_env(name: str, **kwargs):
    if name not in _ENV_REGISTRY:
        raise KeyError(
            f"Env '{name}' not registered. Available: {list(_ENV_REGISTRY)}"
        )
    return _ENV_REGISTRY[name](**kwargs)


def available_envs() -> list[str]:
    return list(_ENV_REGISTRY)