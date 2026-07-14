"""Test: OSC controller — verify torque computation runs and matches robosuite structure."""

import sys
import os

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mujoco_playground._src.manipulation.franka_emika_panda import panda
from libero_mjx.controllers.osc import OscController


def test_osc_smoke():
    """OSC controller produces torques of correct shape."""
    from etils import epath
    xml = epath.Path(__file__).parent.parent / "libero_mjx" / "assets" / "xml" / "libero_bowl_plate.xml"
    assets = panda.get_assets()
    import mujoco
    from mujoco import mjx
    mj_model = mujoco.MjModel.from_xml_string(xml.read_text(), assets=assets)
    mj_model.opt.timestep = 0.002
    mjx_model = mjx.put_model(mj_model, impl="warp")

    osc = OscController.from_model(mj_model, kp=150.0, damping_ratio=1.0)
    osc.set_model(mjx_model)
    print(f"[test_osc] controller created, nu={osc._nu}")

    data = mjx.make_data(mj_model, impl="warp")
    # Forward to populate site_xpos / jacobians
    data = mjx.forward(mjx_model, data)

    delta = jp.zeros(6)
    torques = osc.compute_torques(data, delta)
    print(f"[test_osc] torques shape={torques.shape} dtype={torques.dtype}")
    assert torques.shape[-1] == mj_model.nu, f"Expected nu={mj_model.nu}, got {torques.shape[-1]}"

    # Non-zero action should produce non-zero arm torques
    delta_up = jp.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    torques_up = osc.compute_torques(data, delta_up)
    arm_idx = osc._arm_act_idx
    arm_t = np.asarray(torques_up[arm_idx])
    nz = np.abs(arm_t).sum()
    print(f"[test_osc] arm torque mag for z+ action: {nz:.6f}")
    assert nz > 1e-6, "Expected non-zero arm torques for non-zero action"
    print("[test_osc] PASS")


if __name__ == "__main__":
    test_osc_smoke()