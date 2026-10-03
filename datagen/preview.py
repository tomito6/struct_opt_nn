"""Preview of a written dataset: the exact field of a few instances, samples on top.

Shared by the ``make_*`` scripts. Each panel shows the exact field (colour,
zero level in black) and, over it, a subsample of the rows actually written
to the ``.npz`` -- the check that the files hold what the field says. For
``dim=3`` it is the mid-plane ``z = 0`` and the samples within a thin slab
around it.
"""

from __future__ import annotations

import numpy as np

PREVIEW_INSTANCES = 6


def plot_preview(paths, dim, names, fields, titles, n_show=PREVIEW_INSTANCES):
    """Write ``preview.png`` into the dataset folder. Returns its path.

    Parameters
    ----------
    paths : dict
        :func:`datagen.dataset.dataset_paths` of the dataset, already written.
    dim : {2, 3}
    names : sequence of str
        Instance names, in split order.
    fields : sequence of callable
        ``phi(points)`` of each instance, ``points`` of shape ``(n, dim)`` in
        the normalized frame; same order as ``names``.
    titles : sequence of str
        One per instance, shown above its panel.
    n_show : int
        How many instances, spread evenly over the set.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.unique(np.linspace(0, len(names) - 1, min(n_show, len(names))).astype(int))
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.4))
    grid = np.linspace(-1.0, 1.0, 241)
    gx, gy = np.meshgrid(grid, grid)
    plane = np.column_stack([gx.ravel(), gy.ravel()])
    if dim == 3:
        plane = np.column_stack([plane, np.zeros(len(plane))])

    for ax, i in zip(axs.ravel(), idx):
        field = fields[i](plane).reshape(gx.shape)
        lim = float(np.abs(field).max())
        ax.imshow(
            field,
            extent=(-1, 1, -1, 1),
            origin="lower",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )
        ax.contour(gx, gy, field, levels=[0.0], colors="k", linewidths=1.2)

        with np.load(paths["class_dir"] / f"{names[i]}.npz") as npz:
            rows = np.vstack([npz["pos"], npz["neg"]])
        if dim == 3:
            rows = rows[np.abs(rows[:, 2]) < 0.02]
        rng = np.random.default_rng(i)
        rows = rows[rng.permutation(len(rows))[:1500]]
        ax.scatter(
            rows[:, 0],
            rows[:, 1],
            s=1.5,
            c=np.where(rows[:, -1] < 0, "k", "0.55"),
            linewidths=0,
        )
        ax.set_title(titles[i], fontsize=9)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
    for ax in axs.ravel()[len(idx) :]:
        ax.axis("off")

    where = "plane" if dim == 2 else "mid-plane z = 0"
    fig.suptitle(
        f"{paths['dataset_dir'].name}: exact field on the {where} "
        "(black line: surface) and stored samples (black: inside)"
    )
    fig.tight_layout()
    out = paths["dataset_dir"] / "preview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
