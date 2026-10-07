# Datasets

Training data is not redistributed with this repository. The LIBERO authors
publish human teleoperation demonstrations for all five suites under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This page records
exactly where they come from, how to fetch them, and how to verify the result.

## Where they live

| Source | URL | Notes |
|---|---|---|
| HuggingFace (recommended) | https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets | Reproducible, no expiry, revision-pinnable |
| LIBERO project page | https://libero-project.github.io/datasets.html | Canonical description |
| Original Box links | see `scripts/download_datasets.py --source box` | Expire; fallback only |

The HuggingFace revision this repository was tested against is
`f13aa24a3da8c43c7225569f28c562979fa0e35a`.

## Suites

| Suite | HuggingFace directory | Tasks | Files | Size |
|---|---|---|---|---|
| `spatial` | `libero_spatial` | 10 | 10 | 6.24 GB |
| `object` | `libero_object` | 10 | 10 | 7.44 GB |
| `goal` | `libero_goal` | 10 | 10 | 6.37 GB |
| `scene10` | `libero_10` | 10 | 10 | 13.73 GB |
| `scene90` | `libero_90` | 90 | 90 | 66.66 GB |

Each task is one HDF5 file named after the task, for example
`libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5`.
The full set is about 100 GB; `scene90` alone is two thirds of that.

## Download

```bash
# one suite
python scripts/download_datasets.py --suite spatial

# everything (~100 GB)
python scripts/download_datasets.py --suite all

# to a specific location
python scripts/download_datasets.py --suite scene90 --out-dir /data/libero_datasets

# check an existing copy without downloading
python scripts/download_datasets.py --suite all --verify-only
```

By default the datasets land in `$LIBERO_BASIL_PATH/libero/datasets`, which is
where LIBERO's `get_libero_path("datasets")` looks. The container run wrapper
mounts a host directory there with `LIBERO_DATASETS=/path/to/datasets`.

## Verify

The downloader checks that each suite directory holds the expected number of
`.hdf5` files. For a content check, load one episode:

```python
import h5py
p = "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
with h5py.File(p, "r") as f:
    print(list(f.keys()))          # data, env_args, ...
    print(f["data"].attrs["num_demos"])
```

## Using them

`scripts/train_bc.py` and `scripts/eval_bc.py` / `scripts/eval_warp_only.py`
read the demo path from the LIBERO benchmark registry, so no flag is needed once
the files are in place:

```bash
python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 \
    --save checkpoints/spatial_task0.pth
```

The `scene10` and `scene90` suites map to the LIBERO benchmark names
`LIBERO_10` and `LIBERO_90`; the `libero_100` name used by LIBERO's own
downloader is the union of the two.
