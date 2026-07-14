"""Robosuite compatibility patches for LIBERO suite extraction.

Fixes `robot_base_factory` returning strings instead of classes for
unknown robot base names (e.g. 'NullBase'). This allows loading all 5
LIBERO suites (spatial, object, goal, scene10, scene90) from robosuite
without needing a specific robot base registration.

Usage:
    from libero_mjx.robosuite_patch import patch_robosuite
    patch_robosuite()
    # Now all LIBERO suites can be loaded via OffScreenRenderEnv
"""
from __future__ import annotations


def patch_robosuite():
    """Monkey-patch robosuite's robot_base_factory to handle unknown bases.

    The issue: robosuite 1.5.1's BASE_MAPPING doesn't include 'NullBase'
    (used by LIBERO's Floor/Kitchen scene types). When robot_base_factory
    can't find the base name, it returns an error string instead of raising,
    causing `TypeError: 'str' object is not callable` downstream.

    This patch makes unknown base names fall back to NullMount.
    """
    import robosuite.models.bases as bases

    def patched_factory(name, idn=0):
        if name in bases.BASE_MAPPING:
            return bases.BASE_MAPPING[name](idn=idn)
        return bases.BASE_MAPPING["NullMount"](idn=idn)

    # Patch in all locations where robot_base_factory is imported
    bases.robot_base_factory = patched_factory
    try:
        import robosuite.environments.robot_env as re
        re.robot_base_factory = patched_factory
    except ImportError:
        pass
    try:
        import robosuite.robots.fixed_base_robot as fbr
        fbr.robot_base_factory = patched_factory
    except ImportError:
        pass
    try:
        import robosuite.robots.robot as rbt
        rbt.robot_base_factory = patched_factory
    except ImportError:
        pass

    # Patch ManipulatorModel.add_base to catch string bases
    try:
        from robosuite.models.robots import ManipulatorModel, RobotModel
        _orig_add = ManipulatorModel.add_base
        def _patched_add(self, base=None):
            if base is None or isinstance(base, str):
                base = bases.BASE_MAPPING["NullMount"]()
            return _orig_add(self, base)
        ManipulatorModel.add_base = _patched_add
        RobotModel.add_base = _patched_add
    except ImportError:
        pass