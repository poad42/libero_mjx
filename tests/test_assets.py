"""Every asset path referenced by a task XML must resolve.

This test needs the LIBERO and robosuite assets. It skips (rather than fails)
when they are absent, so it is safe to run anywhere; inside the container or
after ``scripts/setup_assets.sh`` it runs the full check over all 131 XMLs.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from libero_mjx.assets import (
    libero_assets_root,
    resolve_asset_roots,
    robosuite_assets_root,
)

XML_DIR = pathlib.Path(__file__).resolve().parent.parent / "libero_mjx" / "assets" / "xml"
XMLS = sorted(XML_DIR.glob("*.xml"))


def _missing_in(xml_path: pathlib.Path) -> list[str]:
    text = resolve_asset_roots(xml_path.read_text())
    missing = []
    for line in text.splitlines():
        if 'file="' not in line:
            continue
        path = line.split('file="', 1)[1].split('"', 1)[0]
        if path.startswith("/") and not os.path.exists(path):
            missing.append(path)
    return missing


def _assets_present() -> bool:
    return os.path.isdir(libero_assets_root()) and os.path.isdir(robosuite_assets_root())


@pytest.mark.skipif(not XMLS, reason="no task XMLs found")
def test_xml_count():
    assert len(XMLS) == 131, f"expected 131 task XMLs, found {len(XMLS)}"


@pytest.mark.skipif(not _assets_present(), reason="LIBERO/robosuite assets not installed")
@pytest.mark.parametrize("xml_path", XMLS, ids=lambda p: p.stem)
def test_asset_paths_resolve(xml_path: pathlib.Path):
    missing = _missing_in(xml_path)
    assert not missing, f"{xml_path.name}: {len(missing)} unresolved, first: {missing[:3]}"


def test_roots_are_configurable(monkeypatch):
    monkeypatch.setenv("LIBERO_ASSETS_ROOT", "/tmp/libero_x")
    monkeypatch.setenv("ROBOSUITE_ASSETS_ROOT", "/tmp/robosuite_x")
    out = resolve_asset_roots(
        'file="/workspace/libero_basil/libero/libero/assets/a.png" '
        'file="/opt/venv/lib/python3.12/site-packages/robosuite/models/assets/b.stl"'
    )
    assert "/tmp/libero_x/a.png" in out
    assert "/tmp/robosuite_x/b.stl" in out
