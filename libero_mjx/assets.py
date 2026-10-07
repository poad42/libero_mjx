"""Asset-root resolution for the checked-in task XMLs.

The 131 task XMLs were extracted from robosuite/LIBERO and therefore carry
absolute asset paths from the machine they were extracted on. Those paths fall
into three roots:

  /workspace/libero_basil/libero/libero/assets   LIBERO meshes and textures
  /opt/robosuite_assets                          robosuite model assets
  /opt/venv/lib/python3.12/site-packages/robosuite/models/assets
                                                 robosuite model assets (pip)

Instead of shipping 131 rewritten XMLs, the loader maps every known root onto a
single configurable root per source. The defaults reproduce the canonical
container layout, so an image built by ``Dockerfile`` works with no environment
set. A different layout only needs::

    export LIBERO_ASSETS_ROOT=/data/libero/assets
    export ROBOSUITE_ASSETS_ROOT=/data/robosuite/assets

No file in the repository is modified at run time.
"""
from __future__ import annotations

import os

LIBERO_ROOT_DEFAULT = "/workspace/libero_basil/libero/libero/assets"
ROBOSUITE_ROOT_DEFAULT = "/opt/robosuite_assets"

_LIBERO_KNOWN = (
    "/workspace/libero_basil/libero/libero/assets",
)

_ROBOSUITE_KNOWN = (
    "/opt/robosuite_assets",
    "/opt/venv/lib/python3.12/site-packages/robosuite/models/assets",
)


def libero_assets_root() -> str:
    return os.environ.get("LIBERO_ASSETS_ROOT", LIBERO_ROOT_DEFAULT)


def robosuite_assets_root() -> str:
    return os.environ.get("ROBOSUITE_ASSETS_ROOT", ROBOSUITE_ROOT_DEFAULT)


def resolve_asset_roots(xml: str) -> str:
    """Rewrite known absolute asset roots in an MJCF string to the configured roots."""
    libero = libero_assets_root()
    robosuite = robosuite_assets_root()

    for known in _LIBERO_KNOWN:
        if known != libero:
            xml = xml.replace(known, libero)
    for known in _ROBOSUITE_KNOWN:
        if known != robosuite:
            xml = xml.replace(known, robosuite)
    return xml
