#!/usr/bin/env python3
"""Extract MJCF XMLs for all LIBERO suites from robosuite.

LIBERO has 5 suites:
  - libero_spatial: 10 tasks, pick bowl → plate (same objects, different positions)
  - libero_object: 10 tasks, pick food item → basket (same scene, different objects)
  - libero_goal: 10 tasks, different goals with same objects (open drawer, put on stove, etc.)
  - libero_10: 10 tasks, multi-step kitchen tasks (turn on stove + put pot, etc.)
  - libero_90: 90 tasks, longer horizon kitchen tasks

All use the same Panda robot. The scenes differ in objects and fixtures.

Usage:
    python scripts/extract_all_xmls.py --output-dir libero_mjx/assets/xml/
"""
import os, sys, argparse, json

def extract_suite(benchmark_name, task_order_index, output_dir):
    """Extract XML for one task from robosuite LIBERO."""
    sys.path.insert(0, os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"))
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(benchmark_name)(task_order_index)
    n_tasks = benchmark.n_tasks

    for task_id in range(n_tasks):
        task = benchmark.get_task(task_id)
        bddl_path = os.path.join(
            os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"),
            "libero/libero/bddl_files", task.problem_folder, task.bddl_file,
        )

        # Create env to get the MJCF model
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=96, camera_widths=96,
        )
        env.reset()

        # Save MJCF XML from the mujoco model
        import tempfile
        xml_path_tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
        xml_path_tmp.close()
        env.sim.model.save_xml(xml_path_tmp.name)
        with open(xml_path_tmp.name) as f:
            xml_str = f.read()
        os.unlink(xml_path_tmp.name)
        env.close()

        # Save XML
        suite_short = benchmark_name.replace("libero_", "")
        xml_name = f"libero_{suite_short}_task{task_id}.xml"
        xml_path = os.path.join(output_dir, xml_name)
        with open(xml_path, "w") as f:
            f.write(xml_str)
        print(f"  [{benchmark_name}] task {task_id}: {xml_name} ({len(xml_str)} bytes)")

    return n_tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="libero_mjx/assets/xml")
    p.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    for suite in args.suites:
        print(f"\n=== {suite} ===")
        try:
            n = extract_suite(suite, 0, args.output_dir)
            total += n
            print(f"  {suite}: {n} tasks extracted")
        except Exception as e:
            print(f"  {suite}: FAILED - {e}")

    print(f"\nTotal: {total} XMLs extracted to {args.output_dir}")


if __name__ == "__main__":
    main()