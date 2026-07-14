"""Test: success predicates — on_top_of, inside, in_contact, distance_to."""

import sys
import os

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libero_mjx.predicates.spatial import on_top_of, inside, in_contact, distance_to


class FakeData:
    """Minimal stand-in for mjx.Data with xpos/xmat."""

    def __init__(self, positions: dict[int, np.ndarray]):
        self.xpos = positions


def test_predicates():
    # Two bodies: bowl at (0.5, 0, 0.05), plate at (0.5, 0, 0.005)
    bowl = 1
    plate = 2
    pos = {bowl: np.array([0.5, 0.0, 0.05]), plate: np.array([0.5, 0.0, 0.005])}
    data = FakeData(pos)

    # on_top_of: bowl above plate + near + contact → True
    pred = on_top_of(bowl, plate)
    result = np.asarray(pred(data))
    print(f"[test_pred] on_top_of: {bool(result)}")
    # NOTE: _has_contact returns True as placeholder, so this should be True

    # distance_to: bowl within 0.08 of plate pos
    pred_d = distance_to(bowl, np.array([0.5, 0.0, 0.005]), dist=0.08)
    result_d = np.asarray(pred_d(data))
    print(f"[test_pred] distance_to (close): {bool(result_d)}")
    assert bool(result_d), "Expected bowl within 0.08 of plate"

    # Far apart
    pred_far = distance_to(bowl, np.array([1.0, 1.0, 0.005]), dist=0.08)
    result_far = np.asarray(pred_far(data))
    print(f"[test_pred] distance_to (far): {bool(result_far)}")
    assert not bool(result_far), "Expected bowl far from target"

    # inside: bowl within 0.05 of plate
    pred_in = inside(bowl, plate, dist=0.1)
    result_in = np.asarray(pred_in(data))
    print(f"[test_pred] inside: {bool(result_in)}")
    assert bool(result_in)

    # in_contact
    pred_c = in_contact(bowl, plate, dist=0.1)
    result_c = np.asarray(pred_c(data))
    print(f"[test_pred] in_contact: {bool(result_c)}")
    assert bool(result_c)

    # Batched: stack positions
    pos_batch = {
        bowl: np.array([[0.5, 0.0, 0.05], [1.0, 1.0, 0.05]]),
        plate: np.array([[0.5, 0.0, 0.005], [0.5, 0.0, 0.005]]),
    }
    data_b = FakeData(pos_batch)
    pred_batch = distance_to(bowl, np.array([0.5, 0.0, 0.005]), dist=0.08)
    result_b = np.asarray(pred_batch(data_b))
    print(f"[test_pred] batched distance_to: {result_b}")
    assert result_b[0] and not result_b[1]
    print("[test_pred] PASS")


if __name__ == "__main__":
    test_predicates()