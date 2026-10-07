# Assets

The 131 task XMLs in `libero_mjx/assets/xml/` describe scene structure only:
object placement, robot configuration, contact parameters. Every mesh, texture
and 3D model is referenced by absolute path and is **not** redistributed here.

## Where the assets come from

| Source | URL | License | Provides |
|---|---|---|---|
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT (code) | `libero/libero/assets` — object meshes, textures, articulated fixtures, scene XML fragments |
| robosuite | https://github.com/ARISE-Initiative/robosuite | MIT | `robosuite/models/assets` — Panda arm and gripper meshes, bases, arena textures |

Both are installed by the project itself: LIBERO is cloned and robosuite is a pip
dependency. Nothing is vendored into this repository.

## The three roots in the XMLs

The XMLs were extracted from robosuite on a workstation, so they contain the
absolute paths that machine used:

| Root in the XML | Files using it | Resolved to |
|---|---|---|
| `/workspace/libero_basil/libero/libero/assets` | 130 | LIBERO assets |
| `/opt/robosuite_assets` | 10 | robosuite assets |
| `/opt/venv/lib/python3.12/site-packages/robosuite/models/assets` | 120 | robosuite assets |

`libero_mjx/assets.py` rewrites all three roots in memory at load time, so the
checked-in XMLs are never modified. The defaults are the canonical container
layout; a different layout only needs two environment variables:

```bash
export LIBERO_ASSETS_ROOT=/data/libero/assets
export ROBOSUITE_ASSETS_ROOT=/data/robosuite/assets
```

The mapping is applied in `LiberoMjxEnv.__init__` before
`mujoco.MjModel.from_xml_string`, so it covers every entry point.

## Canonical layout

```
/workspace/libero_basil/            # LIBERO checkout
  libero/
    libero/
      assets/                       # LIBERO meshes + textures
      bddl_files/                   # task goal definitions
      init_files/                   # initial states per task
    configs/                        # hydra configs used by train/eval
    datasets/                       # demo HDF5 (see docs/datasets.md)

/opt/robosuite_assets -> <site-packages>/robosuite/models/assets
```

## Set it up on a host

```bash
./scripts/setup_assets.sh
```

The script clones LIBERO at a pinned ref into `$LIBERO_BASIL_PATH`
(`~/workspace/libero_basil` by default), locates the robosuite assets from the
installed package, symlinks `/opt/robosuite_assets` (skip with `--no-link`), and
verifies that every `file=` path in a sample task XML resolves.

To install robosuite itself:

```bash
pip install robosuite==1.5.1
```

## In the container

The `Dockerfile` performs the same steps at build time:

- clones LIBERO into `/workspace/libero_basil`,
- installs `robosuite==1.5.1` and links `/opt/robosuite_assets` to its assets.

So an image built from this repository needs no host assets. To use a host copy
instead, mount it (see `scripts/docker_run.sh`):

```bash
LIBERO_BASIL_PATH=~/workspace/libero_basil \
ROBOSUITE_ASSETS=~/workspace/robosuite_assets \
./scripts/docker_run.sh python tests/test_all_suites.py
```

## Verify

```bash
python -c "
from libero_mjx.assets import resolve_asset_roots, libero_assets_root, robosuite_assets_root
import libero_mjx, pathlib, os
xml = pathlib.Path(libero_mjx.__file__).parent/'assets'/'xml'/'libero_spatial_task0.xml'
text = resolve_asset_roots(xml.read_text())
missing = [l.split('file=\"')[1].split('\"')[0] for l in text.splitlines()
           if 'file=\"' in l and not os.path.exists(l.split('file=\"')[1].split('\"')[0])]
print('roots:', libero_assets_root(), robosuite_assets_root())
print('unresolved:', len(missing))
"
```

`tests/test_assets.py` does the same over all 131 XMLs.
