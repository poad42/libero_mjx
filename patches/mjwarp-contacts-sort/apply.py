"""Backport main's canonical contact sort into a mujoco_warp 4.0.x install.

mujoco_warp 4.0.1 ships DeterminismType.CONTACTS but `_sort_contacts` is a stub
(`# TODO(team): Implementation contact sorting.`). main implements it in
`_src/sort.py`. Copy that module in and point the stub at it.

Usage: python apply.py <site-packages>/mujoco_warp
"""
import shutil
import sys
from pathlib import Path

pkg = Path(sys.argv[1])
sort_py = Path(__file__).with_name("sort.py")
driver = pkg / "_src" / "collision_driver.py"

shutil.copyfile(sort_py, pkg / "_src" / "sort.py")

src = driver.read_text()
stub = '''def _sort_contacts(m: Model, d: Data):
  """Deterministic contact sort."""
  # TODO(team): Implementation contact sorting.
  pass'''
replacement = '''def _sort_contacts(m: Model, d: Data):
  """Deterministic contact sort."""
  from mujoco_warp._src import sort

  sort.contacts(d)'''
if stub not in src:
    if "sort.contacts(d)" in src:
        print("already patched")
        sys.exit(0)
    raise SystemExit("stub not found in collision_driver.py")
driver.write_text(src.replace(stub, replacement, 1))
print("patched", driver)
