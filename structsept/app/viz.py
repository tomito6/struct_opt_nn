"""Drawing layer of the structsept app.

Last stage of the pipeline: takes plain numpy arrays produced by models.py /
datasets.py and paints them onto matplotlib axes, plus one pyvista window for
the surface mesh. No torch and no DeepSDFStruct here, so this module stays
importable and testable without the heavy stack.
"""

import math

import numpy as np

_CP_CMAP = "turbo"


def _xy_extent(bounds):
    """(xmin, xmax, ymin, ymax) from a (2,2) or (2,3) lo/hi array."""
    if bounds is None:
        return (0.0, 1.0, 0.0, 1.0)
    b = np.asarray(bounds, dtype=float)
    lo, hi = b[0], b[1]
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def _grid(field, bounds):
    x0, x1, y0, y1 = _xy_extent(bounds)
    x = np.linspace(x0, x1, field.shape[1])
    y = np.linspace(y0, y1, field.shape[0])
    return x, y


def slice_axes(fig, n):
    """Reset `fig` and lay out `n` panels on it. Returns the list of axes."""
    fig.clear()
    n = max(int(n), 1)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(n)]
    fig.set_layout_engine("constrained")
    return axes


def draw_sdf_slice(ax, field, bounds, title="f_theta"):
    """Signed-distance slice: seismic map plus the zero level set.

    Reuses the image already on ``ax`` when one of the same shape is there, so
    a slider drag only pushes new data instead of rebuilding the panel and its
    colorbar, which costs several times more than evaluating the SDF itself.
    """
    field = np.asarray(field, dtype=float)
    finite = field[np.isfinite(field)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    if not scale > 0:
        scale = 1.0
    extent = _xy_extent(bounds)

    reuse = ax.images and ax.images[0].get_array().shape == field.shape
    if reuse:
        im = ax.images[0]
        im.set_data(field)
        im.set_clim(-scale, scale)
        im.set_extent(extent)
    else:
        ax.clear()
        im = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="seismic",
            vmin=-scale,
            vmax=scale,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _clear_contour(ax)
    x, y = _grid(field, bounds)
    # the zero contour is the geometry itself, everything else is just context
    if finite.size and finite.min() < 0.0 < finite.max():
        ax._structsept_contour = ax.contour(
            x, y, field, levels=[0.0], colors="k", linewidths=1.2
        )
    ax.set_title(title)
    return im


def _clear_contour(ax):
    contour = getattr(ax, "_structsept_contour", None)
    if contour is not None:
        # a contour whose figure was cleared underneath it is already gone, and
        # matplotlib raises rather than no-op when asked to remove it again
        try:
            contour.remove()
        except Exception:
            pass
    ax._structsept_contour = None


def draw_latent_slices(axes, field, bounds):
    """One panel per latent component of a (res, res, d) field."""
    field = np.asarray(field, dtype=float)
    if field.ndim == 2:
        field = field[..., None]
    extent = _xy_extent(bounds)
    images = []
    for i, ax in enumerate(axes[: field.shape[-1]]):
        comp = field[..., i]
        ax.clear()
        lo, hi = float(np.nanmin(comp)), float(np.nanmax(comp))
        if hi - lo < 1e-9:
            lo, hi = lo - 0.05, hi + 0.05
        im = ax.imshow(
            comp,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(f"lambda_{i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        images.append(im)
    return images


def draw_latent_plane(ax, trained, control_points):
    """Trained latent cloud with the current control points on top (d == 2)."""
    ax.clear()
    trained = None if trained is None else np.asarray(trained, dtype=float)
    cps = None if control_points is None else np.asarray(control_points, dtype=float)
    dim = None
    for arr in (trained, cps):
        if arr is not None and arr.ndim == 2:
            dim = arr.shape[1]
            break
    if dim != 2:
        ax.text(
            0.5,
            0.5,
            "so faz sentido com d=2",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return False

    if trained is not None and len(trained):
        # the trained codes are a cloud with holes: the hull only hints at
        # where the decoder was ever supervised, it does not certify it
        ax.scatter(trained[:, 0], trained[:, 1], s=8, c="0.6", alpha=0.6, zorder=1)
        _draw_hull(ax, trained)
    if cps is not None and len(cps):
        ax.scatter(
            cps[:, 0],
            cps[:, 1],
            s=70,
            c=np.arange(len(cps)),
            cmap=_CP_CMAP,
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
        )
    ax.set_xlabel("lambda_1")
    ax.set_ylabel("lambda_2")
    ax.set_title("espaco latente")
    ax.grid(alpha=0.2)
    return True


def _draw_hull(ax, points):
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
    except Exception:
        return
    loop = np.append(hull.vertices, hull.vertices[0])
    ax.fill(points[loop, 0], points[loop, 1], color="0.6", alpha=0.12, zorder=0)
    ax.plot(points[loop, 0], points[loop, 1], color="0.5", lw=0.8, alpha=0.6, zorder=0)


def draw_phi_histogram(ax, phi):
    """Distribution of sampled distance values; inside is phi < 0."""
    ax.clear()
    phi = np.asarray(phi, dtype=float).ravel()
    phi = phi[np.isfinite(phi)]
    if not phi.size:
        ax.text(
            0.5, 0.5, "sem amostras", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    counts, edges, patches = ax.hist(phi, bins=80, color="#4c72b0")
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("phi")
    ax.set_ylabel("amostras")
    inside = float(np.mean(phi < 0.0))
    ax.set_title(f"phi  ({inside:.0%} dentro)")
    return counts, edges


def show_mesh(mesh, title="lattice"):
    """Open a pyvista window on a trimesh surface.

    Returns False when there is nothing to draw or VTK cannot show it. An empty
    mesh is refused on purpose: pyvista happily wraps it and would put up a
    blank window that blocks the interface until the user closes it.
    """
    if mesh is None or len(getattr(mesh, "faces", ())) == 0:
        return False
    try:
        import pyvista as pv

        try:
            poly = pv.wrap(mesh)
        except Exception:
            faces = np.asarray(mesh.faces, dtype=np.int64)
            padded = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces])
            poly = pv.PolyData(np.asarray(mesh.vertices, dtype=float), padded)
        plotter = pv.Plotter(title=title)
        plotter.add_mesh(poly, color="#b0c4de", smooth_shading=True, show_edges=False)
        plotter.add_axes()
        plotter.show()
        return True
    except Exception:
        return False
