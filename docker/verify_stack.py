#!/usr/bin/env python3
"""Verify the libero-mjx container stack.

Two modes:

  python /opt/verify_stack.py --build    # inside `docker build`, no GPU passed
  python /opt/verify_stack.py            # at run time, with --device=/dev/kfd

The build mode checks that the pinned packages import and that the asset roots
exist. The run mode additionally checks that a GPU backend is live for JAX and
that MJX resolves to the Warp implementation.

Exits non-zero on the first failed assertion.
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys

# Make the repo importable when the script is run from a directory other than the
# repo root (the image also sets PYTHONPATH).
_REPO = os.environ.get("REPO_PATH", "/workspace/libero-mjx")
if os.path.isdir(_REPO) and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FAIL = 0


def check(label: str, fn) -> None:
    global FAIL
    try:
        detail = fn()
        print(f"PASS  {label:<44} {detail}")
    except Exception as exc:  # noqa: BLE001
        FAIL = 1
        print(f"FAIL  {label:<44} {type(exc).__name__}: {exc}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true", help="build-time checks (no GPU)")
    args = p.parse_args()

    print("=== libero-mjx stack ===")

    def python_version():
        return sys.version.split()[0]

    check("python", python_version)

    def jax_version():
        import jax
        return f"{jax.__version__} backend={jax.default_backend()}"

    check("jax", jax_version)

    def mujoco_version():
        import mujoco
        return mujoco.__version__

    check("mujoco", mujoco_version)

    def warp_version():
        import warp
        return f"{warp.__version__} {warp.__file__}"

    check("warp", warp_version)

    def mjx_warp():
        import mujoco.mjx.warp as w
        assert w.WARP_INSTALLED, "MJX is not on the Warp implementation"
        return "WARP_INSTALLED=True"

    check("mjx warp backend", mjx_warp)

    def torch_version():
        import torch
        return f"{torch.__version__} hip={torch.version.hip}"

    check("torch", torch_version)

    def robosuite_version():
        import robosuite
        return robosuite.__version__

    check("robosuite", robosuite_version)

    def libero_paths():
        root = pathlib.Path(os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"))
        assets = root / "libero" / "libero" / "assets"
        assert assets.is_dir(), f"missing {assets}"
        return str(assets)

    check("libero assets", libero_paths)

    def robosuite_assets():
        root = pathlib.Path(os.environ.get("ROBOSUITE_ASSETS_ROOT", "/opt/robosuite_assets"))
        assert (root / "base.xml").is_file(), f"missing {root}/base.xml"
        return str(root)

    check("robosuite assets", robosuite_assets)

    # The task XMLs live in the repository, which is mounted at run time and not
    # copied into the image, so this check only runs when the repo is importable.
    try:
        import libero_mjx  # noqa: F401
        _has_repo = True
    except Exception:  # noqa: BLE001
        _has_repo = False

    if _has_repo:
        def xml_count():
            import libero_mjx
            xdir = pathlib.Path(libero_mjx.__file__).parent / "assets" / "xml"
            n = len(list(xdir.glob("*.xml")))
            assert n == 131, f"expected 131 task XMLs, found {n}"
            return f"{n} task XMLs"

        check("task xmls", xml_count)
    else:
        print("SKIP  task xmls (repository not mounted)")

    if not args.build:
        def jax_gpu():
            import jax
            devs = [d for d in jax.devices() if d.platform != "cpu"]
            assert devs, "no GPU device visible to JAX"
            return str(devs)

        check("jax gpu device", jax_gpu)

        def torch_gpu():
            import torch
            assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
            return torch.cuda.get_device_name(0)

        check("torch gpu device", torch_gpu)

        def warp_gpu():
            import warp
            warp.init()
            devs = warp.get_devices()
            assert devs, "no warp device"
            return ", ".join(str(d) for d in devs)

        check("warp gpu device", warp_gpu)

        def env_smoke():
            import jax
            import jax.numpy as jp
            from libero_mjx.envs.libero import LiberoEnv
            env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=1)
            state = env.reset(jax.random.PRNGKey(0))
            state = env.step(state, jp.zeros(7))
            jax.block_until_ready(state.data.qpos)
            return "reset+step ok"

        check("libero env smoke", env_smoke)

    print("=== " + ("ALL CHECKS PASSED" if FAIL == 0 else "CHECKS FAILED") + " ===")
    return FAIL


if __name__ == "__main__":
    sys.exit(main())
