"""Write the two untrained presets of the plate-with-hole experiment into runs/.

A preset is a run directory holding only ``specs.json`` and ``metadata.json``:
the Train tab's "All hyperparameters... > Start from > run: <name>" loads it,
and nothing has been trained yet. ``runs/`` is gitignored and the specs carry
absolute paths, so this script is how the presets are regenerated on another
machine, after the two datasets have been built::

    uv run python -m datagen.make_plate_hole --dim 2 --radius-only \\
        --n-uniform 25000 --n-band 25000 --name plate_hole_2d_r_only --plot
    uv run python -m datagen.make_plate_hole --dim 2 \\
        --n-uniform 25000 --n-band 25000 --name plate_hole_2d_xyr --plot
    uv run python experiments/make_plate_presets.py

The two families
----------------
================================  ======================  ===  =================
preset                            dataset                 d    sigma
================================  ======================  ===  =================
``preset_plate2d_r_only_d1``      ``plate_hole_2d_r_only``  1  sqrt(0.01 * 1)
``preset_plate2d_xyr_d3``         ``plate_hole_2d_xyr``     3  sqrt(0.01 * 3)
================================  ======================  ===  =================

The latent dimension equals the number of generating parameters in each case:
one plate family whose hole only grows, one whose hole also moves. Everything
else is the ``preset_plate2d_30min`` recipe (the spreadsheet's "Quick setup"
scaled to a 30-minute CPU run): 4 x 64 ReLU, no dropout, 4096 samples per
shape, 5 shapes per batch, Adam 5e-4 / 1e-3 halved at epochs 500 and 1000,
clamped L1 at 0.1, 1500 epochs.

``sigma`` follows the spreadsheet's "variance 0.01" the way the earlier preset
did: the trainer draws each code from N(0, (sigma / sqrt(d))^2), so
``sigma = sqrt(0.01 * d)`` gives every component variance 0.01.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

from structsept.app import datasets, hyperparams, training

ROOT = pathlib.Path(__file__).resolve().parents[1]

COMMON = {
    "n_layers": 4,
    "width": 64,
    "latent_in": [2],
    "dropout_layers": "none",
    "dropout_prob": 0.0,
    "num_epochs": 1500,
    "samples_per_scene": 4096,
    "scenes_per_batch": 5,
    "loss_function": "clampedL1",
    "clamping_distance": 0.1,
    "seed": 42,
}

PRESETS = [
    {
        "run": "preset_plate2d_r_only_d1",
        "dataset": "plate_hole_2d_r_only",
        "latent_dim": 1,
        "description": (
            "Preset, untrained. Plate with a CENTRED hole, only the radius "
            "varies (40 radii, r 0.07-0.45): one generating parameter, latent "
            "d=1. 4x64, 1500 epochs, lr halved at 500 and 1000, 4096 samples "
            "per shape, 5 shapes per batch (8 batches/epoch, ~8 min on CPU). "
            "Pick dataset plate_hole_2d_r_only. Afterwards compare the "
            "learned code with params.csv column r."
        ),
    },
    {
        "run": "preset_plate2d_xyr_d3",
        "dataset": "plate_hole_2d_xyr",
        "latent_dim": 3,
        "description": (
            "Preset, untrained. Plate with a hole whose centre AND radius vary "
            "(134 plates: Sobol 128 + 6 extremes, margin 0.05): three "
            "generating parameters, latent d=3 (the d=2 runs so far, on the "
            "25-plate set, did not hold x_c). 4x64, 1500 epochs, lr halved at "
            "500 and 1000, 4096 samples per shape, 5 shapes per batch (~28 min "
            "headless, ~33 min in the GUI). Pick dataset plate_hole_2d_xyr. "
            "Afterwards compare the three code components with params.csv "
            "columns x_c, y_c, r."
        ),
    },
]


def write_preset(preset, data_root, runs_dir, log=print):
    """Validate one preset against its dataset and write its two files.

    Returns the path of the ``specs.json`` written. Exits with a message when
    the dataset is missing or the hyperparameter set is not trainable.
    """
    rows = {d["name"]: d for d in datasets.list_datasets(data_root)}
    row = rows.get(preset["dataset"])
    if row is None or not row["split"]:
        sys.exit(
            f"dataset {preset['dataset']} not found under {data_root}; "
            "build it first (see the module docstring)"
        )
    geom = row["geom_dimension"]
    d = preset["latent_dim"]
    hp = hyperparams.defaults()
    hp.update(COMMON)
    hp.update(
        {
            "latent_dim": d,
            "code_init_std": math.sqrt(0.01 * d),
            "description": preset["description"],
        }
    )
    issues = hyperparams.validate(hp, row["n_instances"], geom)
    for issue in issues:
        log(f"  [{issue.level}] {issue.key}: {issue.message}")
    if hyperparams.errors(issues):
        sys.exit(f"preset {preset['run']} is not trainable")

    run_dir = pathlib.Path(runs_dir) / preset["run"]
    specs = training.write_specs(
        run_dir, row["split"], data_root, hparams=hp, geom_dimension=geom
    )
    training.write_metadata(run_dir, dataset=preset["dataset"], preset=True)
    written = json.loads(specs.read_text(encoding="utf-8"))
    log(
        f"{preset['run']}: d={written['CodeLength']} geom={geom} "
        f"dims={written['NetworkSpecs']['dims']} epochs={written['NumEpochs']} "
        f"sigma={written['CodeInitStdDev']:.4f} "
        f"split={pathlib.Path(written['TrainSplit']).name} "
        f"({row['n_instances']} shapes)"
    )
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write the two plate-with-hole training presets into runs/."
    )
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = parser.parse_args(argv)
    for preset in PRESETS:
        write_preset(preset, pathlib.Path(args.data_root), pathlib.Path(args.runs_dir))


if __name__ == "__main__":
    main()
