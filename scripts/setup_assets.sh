#!/usr/bin/env bash
# Fetch and link the 3D assets the task XMLs reference.
#
# The 131 task XMLs in libero_mjx/assets/xml/ contain scene structure only. Every
# mesh and texture is referenced by absolute path and comes from two upstream
# projects, neither of which is redistributed here:
#
#   LIBERO    https://github.com/Lifelong-Robot-Learning/LIBERO
#             libero/libero/assets  (meshes, textures, object models)
#   robosuite https://github.com/ARISE-Initiative/robosuite
#             robosuite/models/assets  (Panda meshes, grippers, bases, textures)
#
# This script materialises both at the canonical paths the XMLs use:
#
#   $LIBERO_BASIL_PATH/libero/libero/assets   (default: ~/workspace/libero_basil)
#   /opt/robosuite_assets                     (symlink to the installed package)
#
# A non-canonical layout is also supported: set LIBERO_ASSETS_ROOT and
# ROBOSUITE_ASSETS_ROOT and the loader rewrites the roots in memory.
#
# Usage:
#   ./scripts/setup_assets.sh
#   LIBERO_BASIL_PATH=/data/libero ./scripts/setup_assets.sh
#   ./scripts/setup_assets.sh --no-link       # skip the /opt symlink (no sudo)
set -euo pipefail

LIBERO_REPO="${LIBERO_REPO:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_COMMIT="${LIBERO_COMMIT:-master}"
LIBERO_BASIL_PATH="${LIBERO_BASIL_PATH:-$HOME/workspace/libero_basil}"
LINK=1

for arg in "$@"; do
  case "$arg" in
    --no-link) LINK=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

echo "[assets] LIBERO -> ${LIBERO_BASIL_PATH}"
if [ ! -d "${LIBERO_BASIL_PATH}/.git" ]; then
  git clone "${LIBERO_REPO}" "${LIBERO_BASIL_PATH}"
fi
git -C "${LIBERO_BASIL_PATH}" fetch --all --quiet
git -C "${LIBERO_BASIL_PATH}" checkout --quiet "${LIBERO_COMMIT}"
test -d "${LIBERO_BASIL_PATH}/libero/libero/assets" \
  || { echo "[assets] LIBERO assets missing after checkout" >&2; exit 1; }
echo "[assets]   $(du -sh "${LIBERO_BASIL_PATH}/libero/libero/assets" | cut -f1)"

echo "[assets] robosuite"
RS_ASSETS="$(python3 - <<'PY'
import os
try:
    import robosuite
except ImportError:
    raise SystemExit("")
print(os.path.join(os.path.dirname(robosuite.__file__), "models", "assets"))
PY
)"
if [ -z "${RS_ASSETS}" ] || [ ! -d "${RS_ASSETS}" ]; then
  echo "[assets] robosuite not importable; install it with 'pip install robosuite==1.5.1'"
  exit 1
fi
echo "[assets]   ${RS_ASSETS} ($(du -sh "${RS_ASSETS}" | cut -f1))"

if [ "${LINK}" = "1" ]; then
  if [ -e /opt/robosuite_assets ] && [ ! -L /opt/robosuite_assets ]; then
    echo "[assets] /opt/robosuite_assets exists and is not a symlink; leaving it"
  else
    ln -sfn "${RS_ASSETS}" /opt/robosuite_assets 2>/dev/null \
      || sudo ln -sfn "${RS_ASSETS}" /opt/robosuite_assets \
      || echo "[assets] could not link /opt/robosuite_assets; set ROBOSUITE_ASSETS_ROOT instead"
  fi
fi

echo "[assets] verify"
LIBERO_ASSETS_ROOT="${LIBERO_ASSETS_ROOT:-${LIBERO_BASIL_PATH}/libero/libero/assets}" \
ROBOSUITE_ASSETS_ROOT="${ROBOSUITE_ASSETS_ROOT:-/opt/robosuite_assets}" \
python3 - <<'PY'
import os
from libero_mjx.assets import resolve_asset_roots, libero_assets_root, robosuite_assets_root
import libero_mjx, pathlib
xml = pathlib.Path(libero_mjx.__file__).parent / "assets" / "xml" / "libero_spatial_task0.xml"
text = resolve_asset_roots(xml.read_text())
assert libero_assets_root() in text
assert robosuite_assets_root() in text
missing = []
for line in text.splitlines():
    if 'file="' in line:
        path = line.split('file="', 1)[1].split('"', 1)[0]
        if path.startswith("/") and not os.path.exists(path):
            missing.append(path)
print(f"  libero root:    {libero_assets_root()}")
print(f"  robosuite root: {robosuite_assets_root()}")
print(f"  unresolved:     {len(missing)}")
for m in missing[:5]:
    print(f"    {m}")
raise SystemExit(1 if missing else 0)
PY
echo "[assets] OK"
