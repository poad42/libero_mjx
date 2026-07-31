"""Profile Warp eval: per-phase timing breakdown.

Measures wall-clock time for each phase of the eval loop:
  render: WarpRenderer.render() (forward kinematics + ray tracing)
  obs: DLPack transfer + obs dict construction
  policy: BC transformer inference
  step: JAX vstep (OSC controller + physics substeps)

Usage:
  bash scripts/docker_run.sh python scripts/prof_eval.py
  bash scripts/docker_run.sh python scripts/prof_eval.py --n-envs 50 --steps 100
"""
import os, sys, time, argparse, numpy as np, torch, importlib.util

spec = importlib.util.spec_from_file_location("rkp", "libero_mjx/render_kernel_patch.py")
_m = importlib.util.module_from_spec(spec); spec.loader.exec_module(_m); _m.patch_render_kernel()

import jax, jax.numpy as jnp, mujoco.mjx.warp as mjwarp
sys.path.insert(0, ".")
from libero_mjx.warp_gpu_patch import patch_warp_to_gpu; patch_warp_to_gpu()
from libero_mjx import texture_patch
from libero_mjx.envs.libero import LiberoEnv
from libero_mjx.envs.base import LiberoState
from libero_mjx.render import WarpRenderer
from libero_mjx.robosuite_patch import patch_robosuite; patch_robosuite()

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from easydict import EasyDict
import yaml
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.utils import torch_load_model, safe_device, get_task_embs, control_seed
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import get_dataset
from mujoco import mjx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--ckpt", default="checkpoints/task0_model_50ep.pth")
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-dt", type=float, default=0.002, help="Physics timestep (0.002=default, 0.01=fast)")
    args = p.parse_args()

    control_seed(args.seed)
    torch.distributions.Distribution.set_default_validate_args(False)
    cd = os.path.join(os.environ["LIBERO_BASIL_PATH"], "libero/configs")
    benchmark_name = {"spatial":"libero_spatial","object":"libero_object",
        "goal":"libero_goal","scene10":"libero_10","scene90":"libero_90"}[args.suite]
    with initialize_config_dir(version_base=None, config_dir=cd):
        cfg = compose(config_name="config", overrides=[
            f"seed={args.seed}", f"benchmark_name={benchmark_name}",
            "policy=bc_transformer_policy", "lifelong=single_task",
            f"data.task_order_index={args.task_id}",
            "train.num_workers=4","train.n_epochs=50","train.batch_size=32",
            "eval.use_mp=false","eval.num_procs=1","eval.eval=false","use_wandb=false"])
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(cfg)))
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    bm = get_benchmark(benchmark_name)(args.task_id)
    task = bm.get_task(args.task_id)
    init_states = torch.load(os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file), weights_only=False)
    demo = os.path.join(cfg.folder, bm.get_task_demonstration(args.task_id))
    _, shape_meta = get_dataset(dataset_path=demo, obs_modality=cfg.data.obs.modality, initialize_obs_utils=True, seq_len=cfg.data.seq_len)
    cfg.shape_meta = shape_meta
    descs = [bm.get_task(i).language for i in range(bm.n_tasks)]
    embs = get_task_embs(cfg, descs); bm.set_task_embs(embs)
    task_emb = bm.get_task_emb(args.task_id)
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(1, cfg), "cuda")
    sd,_,pv = torch_load_model(args.ckpt, map_location="cuda")
    algo.policy.load_state_dict(sd); algo.policy.previous_mask=pv; algo.eval()

    env = LiberoEnv(suite=args.suite, task_id=args.task_id, impl="warp", n_envs=args.n_envs, optimize_physics=False)
    env.load_init_states(args.task_id)
    if args.sim_dt != 0.002:
        env._sim_dt = args.sim_dt
    print(f"n_substeps = {env.n_substeps} (sim_dt={env.sim_dt}, ctrl_dt={env.dt})")

    m = env._mj_model; nq,nv = m.nq,m.nv
    renderer = WarpRenderer(m, n_envs=args.n_envs, brightness_boost=1.15)
    ep_idx = [i % init_states.shape[0] for i in range(args.n_envs)]
    qp = jnp.array([init_states[i][1:1+nq] for i in ep_idx], dtype=jnp.float32)
    qv = jnp.array([init_states[i][1+nq:1+nq+nv] for i in ep_idx], dtype=jnp.float32)
    def mk_state(qpos, qvel, rng):
        d = mjx.make_data(m, impl="warp", naconmax=env._naconmax, njmax=env._njmax)
        d = d.replace(qpos=qpos, qvel=qvel); d = mjx.forward(env._mjx_model, d)
        info = {"rng":rng,"step":jnp.array(0,dtype=jnp.int32),"gripper_current_action":jnp.zeros(2)}
        obs = env._get_obs(d, info)
        metrics = {k: jnp.array(0.0) for k in env._reward_keys()}; metrics["success"]=jnp.array(0.0)
        return LiberoState(d, obs, jnp.array(0.0), jnp.array(0.0), metrics, info)
    vstep = jax.jit(jax.vmap(env.step))
    state = jax.jit(jax.vmap(mk_state))(qp, qv, jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs))
    for _ in range(5): state = vstep(state, jnp.zeros((args.n_envs,7)))
    jax.block_until_ready(state.data.qpos)

    _p = algo.policy
    def _ga(data):
        _p.eval()
        with torch.no_grad():
            d = _p.preprocess_input(data, train_mode=False)
            x = _p.spatial_encode(d); _p.latent_queue.append(x)
            if len(_p.latent_queue) > _p.max_seq_len: _p.latent_queue.pop(0)
            x = torch.cat(_p.latent_queue, dim=1); x = _p.temporal_encode(x)
            dist = _p.policy_head(x[:, -1])
        return dist.sample().detach().view(-1, 7)
    _p.get_action = _ga; _p.reset()
    _ = renderer.render(state_data=state.data)
    jax.block_until_ready(state.data.qpos)

    times = {"render": [], "obs": [], "policy": [], "step": []}
    for s in range(args.steps):
        t0 = time.perf_counter()
        images = renderer.render(state_data=state.data); torch.cuda.synchronize()
        times["render"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        qpos_t = torch.utils.dlpack.from_dlpack(state.data.qpos.__dlpack__())
        obs = {"obs": {"agentview_rgb": images["agentview_rgb"], "eye_in_hand_rgb": images["eye_in_hand_rgb"],
            "joint_states": qpos_t[:, env._robot_arm_qposadr].to("cuda"), "gripper_states": qpos_t[:, 7:9].to("cuda")},
            "task_emb": task_emb.unsqueeze(0).repeat(args.n_envs, 1).to("cuda")}
        torch.cuda.synchronize(); times["obs"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        action = _p.get_action(obs); torch.cuda.synchronize()
        times["policy"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        state = vstep(state, jnp.from_dlpack(action))
        jax.block_until_ready(state.data.qpos)
        times["step"].append(time.perf_counter() - t0)

    print(f"\nProfiling {args.steps} steps, {args.n_envs} envs, n_substeps={env.n_substeps}:\n")
    for k, v in times.items():
        arr = np.array(v) * 1000
        print(f"  {k:10s}: {arr.mean():7.1f} ms avg  ({arr[5:].mean():7.1f} ms excl warmup)")
    total = sum(np.array(v).mean() for v in times.values()) * 1000
    print(f"  {'total':10s}: {total:7.1f} ms/step  ({args.n_envs/total*1000:.1f} env-steps/s)")


if __name__ == "__main__":
    main()