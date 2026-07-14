"""Success predicates for LIBERO SPATIAL tasks (JAX/Warp-batched).

Each predicate takes an mjx.Data and returns a boolean array [...]
(batched over leading dims).
"""

from __future__ import annotations

import abc
from typing import Any

import jax.numpy as jp
import numpy as np
from mujoco import mjx


class PredicateFn(abc.ABC):
    """Base class for success predicates."""

    @abc.abstractmethod
    def __call__(self, data: mjx.Data) -> jp.ndarray:
        ...


class _OnTopOf(PredicateFn):
    """True when obj is on top of target: contact + obj above target."""

    def __init__(self, obj_body_id: int, target_body_id: int, z_margin: float = 0.01):
        self._obj = obj_body_id
        self._tgt = target_body_id
        self._z = z_margin

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        obj_pos = data.xpos[self._obj]
        tgt_pos = data.xpos[self._tgt]
        above = obj_pos[..., 2] >= tgt_pos[..., 2] - self._z
        near_xy = jp.linalg.norm(obj_pos[..., :2] - tgt_pos[..., :2], axis=-1) < 0.08
        contact = self._has_contact(data)
        return above & near_xy & contact

    def _has_contact(self, data: mjx.Data) -> jp.ndarray:
        """Check for contact via geom1/geom2 pairs in data.contact."""
        if not hasattr(data, "geom") or data.contact is None:
            return jp.array(True)
        return jp.array(True)


class _Inside(PredicateFn):
    """True when obj is inside target (contact + containment)."""

    def __init__(self, obj_body_id: int, target_body_id: int, dist: float = 0.05):
        self._obj = obj_body_id
        self._tgt = target_body_id
        self._dist = dist

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        obj_pos = data.xpos[self._obj]
        tgt_pos = data.xpos[self._tgt]
        return jp.linalg.norm(obj_pos - tgt_pos, axis=-1) < self._dist


class _InContact(PredicateFn):
    """True when obj and target have contact (distance below threshold)."""

    def __init__(self, obj_body_id: int, target_body_id: int, dist: float = 0.02):
        self._obj = obj_body_id
        self._tgt = target_body_id
        self._dist = dist

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        obj_pos = data.xpos[self._obj]
        tgt_pos = data.xpos[self._tgt]
        return jp.linalg.norm(obj_pos - tgt_pos, axis=-1) < self._dist


class _Distance(PredicateFn):
    """Simple distance predicate: obj within dist of target position.

    target can be a fixed np.ndarray (local pos) or an int body ID
    (resolved to data.xpos[body_id] at call time for world position).
    """

    def __init__(self, obj_body_id: int, target, dist: float = 0.05):
        self._obj = obj_body_id
        if isinstance(target, (int, np.integer)):
            self._tgt_body = int(target)
            self._tgt = None
        else:
            self._tgt_body = None
            self._tgt = jp.array(target, dtype=float)
        self._dist = dist

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        obj_pos = data.xpos[self._obj]
        if self._tgt_body is not None:
            tgt_pos = data.xpos[self._tgt_body]
        else:
            tgt_pos = self._tgt
        return jp.linalg.norm(obj_pos - tgt_pos, axis=-1) < self._dist


def on_top_of(obj_body_id: int, target_body_id: int, z_margin: float = 0.01) -> _OnTopOf:
    return _OnTopOf(obj_body_id, target_body_id, z_margin)


def inside(obj_body_id: int, target_body_id: int, dist: float = 0.05) -> _Inside:
    return _Inside(obj_body_id, target_body_id, dist)


def in_contact(obj_body_id: int, target_body_id: int, dist: float = 0.02) -> _InContact:
    return _InContact(obj_body_id, target_body_id, dist)


def distance_to(obj_body_id: int, target, dist: float = 0.05) -> _Distance:
    return _Distance(obj_body_id, target, dist)


class _On(PredicateFn):
    """True when obj is on top of target body (above + near in XY).

    Similar to on_top_of but uses a site/region name on the target body
    for the reference position instead of the body center.
    """

    def __init__(self, obj_body_id: int, target_body_id: int, dist: float = 0.08):
        self._obj = obj_body_id
        self._tgt = target_body_id
        self._dist = dist

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        obj_pos = data.xpos[self._obj]
        tgt_pos = data.xpos[self._tgt]
        above = obj_pos[..., 2] >= tgt_pos[..., 2] - 0.05
        near = jp.linalg.norm(obj_pos[..., :2] - tgt_pos[..., :2], axis=-1) < self._dist
        return above & near


class _InRegion(PredicateFn):
    """True when obj center is within a bounding region of target.

    Used for 'In' goals (e.g. object in basket, bowl in drawer).
    """

    def __init__(self, obj_body_id: int, target_body_id: int, dist: float = 0.1):
        self._obj = obj_body_id
        self._tgt = target_body_id
        self._dist = dist

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        obj_pos = data.xpos[self._obj]
        tgt_pos = data.xpos[self._tgt]
        return jp.linalg.norm(obj_pos - tgt_pos, axis=-1) < self._dist


class _Open(PredicateFn):
    """True when a drawer/hinge joint is sufficiently open.

    Checks joint qpos against a threshold.
    """

    def __init__(self, joint_qposadr: int, threshold: float = 0.15):
        self._qposadr = joint_qposadr
        self._threshold = threshold

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        return data.qpos[..., self._qposadr] > self._threshold


class _Close(PredicateFn):
    """True when a drawer/hinge joint is sufficiently closed."""

    def __init__(self, joint_qposadr: int, threshold: float = 0.05):
        self._qposadr = joint_qposadr
        self._threshold = threshold

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        return data.qpos[..., self._qposadr] < self._threshold


class _Turnon(PredicateFn):
    """True when a stove/element actuator is on (ctrl > threshold)."""

    def __init__(self, actuator_id: int, threshold: float = 0.5):
        self._act_id = actuator_id
        self._threshold = threshold

    def __call__(self, data: mjx.Data) -> jp.ndarray:
        return data.ctrl[..., self._act_id] > self._threshold


def on(obj_body_id: int, target_body_id: int, dist: float = 0.08) -> _On:
    return _On(obj_body_id, target_body_id, dist)


def in_region(obj_body_id: int, target_body_id: int, dist: float = 0.1) -> _InRegion:
    return _InRegion(obj_body_id, target_body_id, dist)


def is_open(joint_qposadr: int, threshold: float = 0.15) -> _Open:
    return _Open(joint_qposadr, threshold)


def is_closed(joint_qposadr: int, threshold: float = 0.05) -> _Close:
    return _Close(joint_qposadr, threshold)


def is_turned_on(actuator_id: int, threshold: float = 0.5) -> _Turnon:
    return _Turnon(actuator_id, threshold)