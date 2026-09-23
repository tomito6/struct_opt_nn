"""Drawing layer of the structsept app.

Last stage of the pipeline: takes plain numpy arrays produced by models.py /
datasets.py and paints them onto matplotlib axes, plus one pyvista window for
the surface mesh. No torch and no DeepSDFStruct here, so this module stays
importable and testable without the heavy stack.

Two rendering decisions are load-bearing and easy to undo by accident:

* The SDF slice is drawn as **material / void**, not as a diverging map with a
  per-frame symmetric autoscale. The magnitude of ``phi`` away from the
  surface is an artefact of the SDF construction (a clamp plus the border cap)
  and the network is only accurate near the zero level set anyway; what the
  panel has to answer while a control point is dragged is *what shape is this
  now*, which is ``sign(phi)``. A per-frame autoscale also re-colours the
  image at the same time as it changes the geometry, which is the one thing an
  exploration tool must not do.
* The latent panels use a **fixed** normalization per component - the trained
  range - not the min/max of the frame. The starting design is a constant
  latent field whose numerical spread is ~1e-8; autoscaled, it renders that
  round-off as a full-contrast pattern.
"""

import math

import numpy as np

GAP_FRACTION = 0.12  # a gap counts as untrained past this share of the range
MESH_PREVIEW_CAP = 250_000  # triangles; above this the 3D preview is decimated
MATERIAL = "#334C66"
VOID = "#EEF1F4"
_LATENT_CMAP = "cividis"


def _xy_extent(bounds):
    """(xmin, xmax, ymin, ymax) from a (2,2) or (2,3) lo/hi array."""
    if bounds is None:
        return (0.0, 1.0, 0.0, 1.0)
    b = np.asarray(bounds, dtype=float)
    lo, hi = b[0], b[1]
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def _drop(artist):
    """Detach an artist we own, tolerating one ``ax.clear()`` already did.

    matplotlib's ``Axes.clear`` rebinds the children list and leaves the old
    artists with ``_remove_method = None``; calling ``remove()`` on one of
    those raises ``NotImplementedError``. We keep our own references to
    overlays across redraws, so exactly that collision happens whenever a
    panel is rebuilt (a resolution change, say) while an overlay is up.
    """
    if artist is None:
        return
    try:
        if getattr(artist, "axes", None) is None:
            return
        artist.remove()
    except Exception:
        pass


def _pixel_extent(extent, shape):
    """Grow a node-sampled extent by half a cell in each direction.

    ``models.eval_sdf_slice`` samples on ``linspace(lo, hi, res)``, so the
    first and last samples sit exactly on the domain faces - they are grid
    *nodes*. ``imshow`` maps the extent to pixel *edges*, so passing the domain
    box directly puts every sample half a pixel off and shrinks the picture by
    ``(res-1)/res``. Extending by half a cell puts sample i at its own pixel
    centre, which is where the control-point overlay expects it.
    """
    x0, x1, y0, y1 = extent
    ny, nx = int(shape[0]), int(shape[1])
    hx = 0.5 * (x1 - x0) / max(nx - 1, 1)
    hy = 0.5 * (y1 - y0) / max(ny - 1, 1)
    return (x0 - hx, x1 + hx, y0 - hy, y1 + hy)


def slice_axes(fig, n):
    """Reset `fig` and lay out `n` panels on it. Returns the list of axes."""
    fig.clear()
    n = max(int(n), 1)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(n)]
    fig.set_layout_engine("constrained")
    return axes


def single_axes(fig):
    """Reset `fig` down to one full-bleed panel."""
    fig.clear()
    fig.set_layout_engine("constrained")
    return fig.add_subplot(111)


# --------------------------------------------------------------------------- #
# signed distance
# --------------------------------------------------------------------------- #


def _material_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("material_void", [MATERIAL, VOID])


def _edge_halfwidth(field, bounds, ax, aa_px=1.2):
    """Half-width of the normalization window, in phi units.

    Chosen so the material/void transition is anti-aliased over roughly
    ``aa_px`` screen pixels and no more. It cannot be read off the pixel size
    alone: ``phi`` is the network's pseudo-distance in unit-cell coordinates
    and the transformation function rescales its gradient, so ``|grad phi|``
    in domain coordinates is measured rather than assumed (3.9 for
    AnalyticRoundCross, 0.85 for ChiAndCross on the same domain).
    """
    x0, x1, y0, y1 = _xy_extent(bounds)
    try:
        width_px = max(float(ax.get_window_extent().width), 1.0)
    except Exception:
        width_px = 400.0
    per_pixel = abs(x1 - x0) / width_px

    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 0.5 * aa_px * per_pixel
    band = np.abs(field) < max(3.0 * per_pixel, 1e-6)
    grad = 1.0
    if band.sum() > 4 and field.shape[0] > 1 and field.shape[1] > 1:
        dy = abs(y1 - y0) / max(field.shape[0] - 1, 1)
        dx = abs(x1 - x0) / max(field.shape[1] - 1, 1)
        gy, gx = np.gradient(field, dy, dx)
        local = np.hypot(gx, gy)[band]
        local = local[np.isfinite(local) & (local > 0)]
        if local.size:
            grad = float(np.median(local))
    return max(0.5 * aa_px * per_pixel * grad, 1e-9)


def draw_sdf_slice(ax, field, bounds, reuse=True):
    """Material/void image of a signed-distance slice.

    Reuses the image already on ``ax`` when one of the same shape is there, so
    a slider drag only pushes new data instead of rebuilding the panel, which
    costs several times more than evaluating the SDF itself. No title and no
    colorbar on purpose: glyph rasterisation dominates the frame budget, and
    both belong in ttk labels beside the panel.
    """
    from matplotlib.colors import Normalize

    field = np.asarray(field, dtype=float)
    extent = _pixel_extent(_xy_extent(bounds), field.shape)
    half = _edge_halfwidth(field, bounds, ax)
    norm = Normalize(-half, half)

    have = reuse and ax.images and ax.images[0].get_array().shape == field.shape
    if have:
        im = ax.images[0]
        im.set_data(field)
        im.set_norm(norm)
        im.set_extent(extent)
    else:
        ax.clear()
        im = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=_material_cmap(),
            norm=norm,
            interpolation="bilinear",
            interpolation_stage="data",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # a little air around the domain, so control-point markers sitting exactly
    # on the boundary are not cut in half by the axes. Limits come from the
    # domain, not from the half-pixel-extended image extent, so the overlay
    # and the image stay in the same coordinates.
    x0, x1, y0, y1 = _xy_extent(bounds)
    padx, pady = 0.035 * abs(x1 - x0), 0.035 * abs(y1 - y0)
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady)
    return im


def draw_control_points(ax, positions, values, lo, hi, selected=None):
    """Overlay the control points of one layer on a slice, in domain coordinates.

    Makes the B-spline's local support checkable rather than something to take
    on faith: the marker the user is dragging sits on the region that responds.
    """
    for artist in getattr(ax, "_structsept_cps", ()) or ():
        _drop(artist)
    ax._structsept_cps = []

    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    if positions.size == 0:
        return
    values = np.asarray(values, dtype=float).ravel()
    span = float(hi) - float(lo)
    t = np.full(len(positions), 0.5) if span <= 1e-12 else (values - lo) / span

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.clip(t, 0.0, 1.0),
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        s=46,
        edgecolors="white",
        linewidths=1.0,
        zorder=5,
    )
    ax._structsept_cps.append(scatter)
    if selected is not None and 0 <= selected < len(positions):
        ring = ax.scatter(
            positions[selected, 0],
            positions[selected, 1],
            s=150,
            facecolors="none",
            edgecolors="#2563EB",
            linewidths=2.0,
            zorder=6,
        )
        ax._structsept_cps.append(ring)


# --------------------------------------------------------------------------- #
# latent field
# --------------------------------------------------------------------------- #


def draw_latent_slices(axes, field, bounds, ranges, components=None):
    """One panel per selected latent component of a (res, res, d) field.

    ``ranges`` is the fixed (lo, hi) per component - the trained range - so the
    colour of a panel means the same thing from frame to frame and across
    components. Each panel states its own interval in the title because latent
    dimensions are not commensurable: forcing them onto one shared scale would
    be a different false equivalence, not a fix.
    """
    from matplotlib.colors import Normalize

    field = np.asarray(field, dtype=float)
    if field.ndim == 2:
        field = field[..., None]
    if components is None:
        components = list(range(field.shape[-1]))
    extent = _pixel_extent(_xy_extent(bounds), field.shape[:2])

    for ax, comp in zip(axes, components):
        if comp >= field.shape[-1]:
            continue
        data = field[..., comp]
        lo, hi = ranges[comp]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        norm = Normalize(lo, hi)
        if ax.images and ax.images[0].get_array().shape == data.shape:
            im = ax.images[0]
            im.set_data(data)
            im.set_norm(norm)
            im.set_extent(extent)
        else:
            ax.clear()
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=_LATENT_CMAP,
                norm=norm,
                interpolation="bilinear",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax.set_title(f"λ{_sub(comp + 1)}   [{lo:+.2f}, {hi:+.2f}]", fontsize=8)

        # A flat panel is the honest rendering of the uniform lattice, but on
        # its own it is indistinguishable from a broken colour scale - say so.
        _drop(getattr(ax, "_structsept_flat", None))
        ax._structsept_flat = None
        if float(np.nanmax(data) - np.nanmin(data)) < 1e-6:
            ax._structsept_flat = ax.text(
                0.5,
                0.5,
                f"constant   {float(np.nanmean(data)):+.3f}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )


def draw_latent_coverage(ax, trained, values, component, palette=None):
    """Where the trained codes of one latent component are, and where we are.

    This is the honest replacement for the latent scatter plot: same data, one
    dimension at a time, readable for any ``d``, and it shows the thing the
    scatter never did - the gaps. A design sitting in a gap is inside the
    min/max box and still unsupported.
    """
    warn = (palette or {}).get("warning_soft", "#FDF2E2")
    accent = (palette or {}).get("primary", "#2563EB")
    muted = (palette or {}).get("text_muted", "#5B6B7C")

    ax.clear()
    trained = np.asarray(trained, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    if trained.ndim != 2 or component >= trained.shape[1] or trained.shape[0] == 0:
        ax.text(
            0.5,
            0.5,
            "no trained codes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=muted,
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    codes = trained[:, component]
    lo, hi = float(np.min(codes)), float(np.max(codes))
    span = max(hi - lo, 1e-6)

    ax.axvspan(lo, hi, color=accent, alpha=0.07, lw=0)

    # the widest empty stretch between two trained codes is untrained territory
    order = np.unique(codes)
    if order.size > 1:
        gaps = np.diff(order)
        idx = int(np.argmax(gaps))
        if gaps[idx] > GAP_FRACTION * span:
            ax.axvspan(order[idx], order[idx + 1], color=warn, zorder=0, lw=0)

    if codes.size > 400:
        # thousands of codes: a rug turns into a solid bar, so show density
        ax.hist(codes, bins=80, color=muted, alpha=0.55, lw=0)
        ax.set_yticks([])
    else:
        ax.vlines(codes, 0.0, 1.0, color=muted, lw=0.9, alpha=0.7)
        ax.set_ylim(-0.6, 2.2)
        ax.set_yticks([])

    top = ax.get_ylim()[1]
    ax.plot(
        values,
        np.full(values.shape, top * 0.78),
        "v",
        color=accent,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.8,
        clip_on=False,
    )
    pad = 0.08 * span
    ax.set_xlim(
        min(lo, float(np.min(values))) - pad, max(hi, float(np.max(values))) + pad
    )
    # no axes title: the strip is short and the component is named in the
    # caption underneath, where it does not eat plot height
    ax.tick_params(axis="x", labelsize=7)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def _sub(n):
    """Unicode subscript digits, so panel titles read as maths without mathtext."""
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


# --------------------------------------------------------------------------- #
# planar decoders (Explore 2-D)
# --------------------------------------------------------------------------- #

_PARAM_CMAP = "viridis"
_ERROR_CMAP = "magma_r"
BOUNDARY_COLOR = "#F59E0B"


def _map_xy(codes, comps, colour=None):
    """Map coordinates of each code: two latent components, or for ``d = 1``
    the one component against the colour column (a lone axis has no y)."""
    codes = np.asarray(codes, dtype=float)
    x = codes[:, comps[0]]
    if codes.shape[1] == 1:
        y = np.zeros_like(x) if colour is None else np.asarray(colour, dtype=float)
    else:
        y = codes[:, comps[1]]
    return x, y


def draw_latent_map(fig, codes, comps, colour=None, colour_label=None, error=False):
    """One dot per trained code, coloured by what generated that shape.

    This is the latent scatter the Explore tab dropped, back for a reason: a
    scatter of bare codes says nothing, but coloured by the parameters that
    made each shape it answers the question the planar experiment asks -
    which of those parameters the latent space kept. A smooth colour gradient
    across the cloud is a parameter the codes encode; speckle is one they lost.

    Rebuilds the figure (a colorbar cannot be updated in place); the marker
    of the current latent vector goes on top with :func:`draw_latent_marker`,
    which is cheap enough to follow a drag.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    codes : array_like, shape (n, d)
    comps : (int, int)
        Latent components on the x and y axes; the second is ignored for d = 1.
    colour : array_like, shape (n,), optional
        Value per code; ``None`` draws plain dots.
    colour_label : str, optional
    error : bool
        Colour with the error map (bright = bad) instead of the parameter map.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig.clear()
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(111)
    codes = np.asarray(codes, dtype=float)
    if codes.ndim != 2 or not len(codes):
        ax.text(0.5, 0.5, "no trained codes", ha="center", va="center")
        ax.set_axis_off()
        return ax
    x, y = _map_xy(codes, comps, colour)
    if colour is None:
        ax.scatter(x, y, s=16, color="#5B6B7C", alpha=0.8, linewidths=0)
    else:
        mapped = ax.scatter(
            x,
            y,
            c=np.asarray(colour, dtype=float),
            cmap=_ERROR_CMAP if error else _PARAM_CMAP,
            s=20,
            edgecolors="white",
            linewidths=0.3,
        )
        bar = fig.colorbar(mapped, ax=ax, shrink=0.9, pad=0.02)
        bar.set_label(colour_label or "", fontsize=8)
        bar.ax.tick_params(labelsize=7)
    ax.set_xlabel(f"λ{_sub(comps[0] + 1)}", fontsize=9)
    if codes.shape[1] == 1:
        ax.set_ylabel(colour_label or "", fontsize=9)
    else:
        ax.set_ylabel(f"λ{_sub(comps[1] + 1)}", fontsize=9)
        # distances in the latent space are what "nearest code" means, so the
        # two axes share a scale
        ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(0.08)
    return ax


def draw_latent_marker(ax, x, y=None, nearest=None):
    """Mark the current latent vector, and ring the trained code nearest to it.

    ``y`` is ``None`` on a one-component map, where the current value is a
    vertical line rather than a point. Earlier markers are removed first, so
    this can run on every slider step.
    """
    for artist in getattr(ax, "_current_markers", []):
        _drop(artist)
    artists = []
    if nearest is not None:
        artists += ax.plot(
            [nearest[0]],
            [nearest[1]],
            "o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor=BOUNDARY_COLOR,
            markeredgewidth=1.8,
            zorder=5,
        )
    if y is None:
        artists.append(ax.axvline(x, color="#C0271B", lw=1.6, zorder=6))
    else:
        artists += ax.plot(
            [x], [y], "+", color="#C0271B", markersize=16, markeredgewidth=2.2, zorder=6
        )
    ax._current_markers = artists


def draw_parameter_panels(fig, codes, comps, columns, scores=None):
    """The trained codes once per generating parameter, side by side.

    Small multiples of :func:`draw_latent_map`: reading which parameter the
    codes keep should not depend on flipping a colour menu. ``scores`` puts a
    number on each panel (``models.neighbour_r2``), so the verdict does not
    rest on eyeballing colours alone.

    Parameters
    ----------
    columns : list of (str, array_like)
        ``(label, value per code)``; a label starting with "error" gets the
        error colour map.
    scores : list of float, optional
    """
    fig.clear()
    fig.set_layout_engine("constrained")
    codes = np.asarray(codes, dtype=float)
    if not columns or codes.ndim != 2 or not len(codes):
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "no parameter table next to this dataset\n" "(datagen writes params.csv)",
            ha="center",
            va="center",
            fontsize=8,
            color="#5B6B7C",
        )
        ax.set_axis_off()
        return []
    n = len(columns)
    ncols = 1 if n == 1 else 2
    nrows = math.ceil(n / ncols)
    axes = []
    for i, (label, values) in enumerate(columns):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        x, y = _map_xy(codes, comps, values)
        mapped = ax.scatter(
            x,
            y,
            c=np.asarray(values, dtype=float),
            cmap=_ERROR_CMAP if label.startswith("error") else _PARAM_CMAP,
            s=9,
            linewidths=0,
        )
        bar = fig.colorbar(mapped, ax=ax, shrink=0.85, pad=0.01)
        bar.ax.tick_params(labelsize=6)
        title = label
        if scores is not None and i < len(scores) and np.isfinite(scores[i]):
            title += f"   R² {scores[i]:.2f}"
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if codes.shape[1] > 1:
            ax.set_aspect("equal", adjustable="datalim")
        axes.append(ax)
    return axes


def draw_boundary_points(ax, points):
    """Overlay stored near-surface samples of a training shape on the field.

    Replaces the previous overlay; an empty array just removes it.
    """
    _drop(getattr(ax, "_boundary_overlay", None))
    ax._boundary_overlay = None
    points = np.asarray(points, dtype=float)
    if points.size:
        ax._boundary_overlay = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=1.2,
            color=BOUNDARY_COLOR,
            linewidths=0,
            zorder=4,
        )


# --------------------------------------------------------------------------- #
# dataset and training
# --------------------------------------------------------------------------- #


def draw_phi_histogram(ax, phi):
    """Distribution of sampled distance values; inside is phi < 0."""
    ax.clear()
    phi = np.asarray(phi, dtype=float).ravel()
    phi = phi[np.isfinite(phi)]
    if not phi.size:
        ax.text(
            0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    counts, edges, _ = ax.hist(phi, bins=80, color="#4c72b0")
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("phi")
    ax.set_ylabel("samples")
    inside = float(np.mean(phi < 0.0))
    ax.set_title(f"phi  ({inside:.0%} inside)")
    return counts, edges


def draw_loss_curve(ax, losses, epochs=None, palette=None):
    """Training loss against epoch, log scale.

    A 200-epoch run otherwise shows an unchanging log box from start to finish,
    with no way to tell a running job from a hung one.
    """
    accent = (palette or {}).get("primary", "#2563EB")
    muted = (palette or {}).get("text_muted", "#5B6B7C")

    ax.clear()
    losses = np.asarray(losses, dtype=float).ravel()
    losses = losses[np.isfinite(losses) & (losses > 0)]
    if losses.size == 0:
        ax.text(
            0.5,
            0.5,
            "waiting for the first checkpoint",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=muted,
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    x = (
        np.linspace(0.0, float(epochs), losses.size)
        if epochs
        else np.arange(losses.size)
    )
    ax.plot(x, losses, color=accent, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("epoch" if epochs else "batch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# 3D
# --------------------------------------------------------------------------- #


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


def draw_mesh_preview(ax, mesh, palette=None):
    """Static 3-D preview of a surface mesh on a matplotlib Axes3D.

    Not a replacement for the pyvista window - no rotation, no lighting worth
    the name - but it puts the result on screen without blocking the Tk loop,
    which ``show_mesh`` does by design.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    muted = (palette or {}).get("text_muted", "#5B6B7C")
    ax.clear()
    if mesh is None or len(getattr(mesh, "faces", ())) == 0:
        ax.text2D(
            0.5,
            0.5,
            "no surface",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=muted,
            fontsize=8,
        )
        ax.set_axis_off()
        return False

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    # Dropping every n-th triangle turns a closed surface into see-through
    # speckle, so decimate only when the draw would actually hurt. Measured
    # with matplotlib 3.11 on this machine: 82 000 triangles draw in 0.02 s,
    # so the cap is a guard against pathological meshes, not a normal path.
    if len(faces) > MESH_PREVIEW_CAP:
        faces = faces[:: int(np.ceil(len(faces) / MESH_PREVIEW_CAP))]

    # Flat-shade the triangles. Without it every face gets the same colour and
    # the preview is a silhouette with no depth at all.
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths > 0, lengths, 1.0)
    light = np.array([-0.45, -0.75, 0.5])
    light = light / np.linalg.norm(light)
    lambert = normals @ light
    # FlexiCubes does not promise a winding direction, so take the sign from
    # the mesh itself rather than assuming outward normals. abs() would work
    # for either winding but lights the surface from both sides at once and
    # throws away a third of the contrast.
    if float(np.mean(lambert)) < 0.0:
        lambert = -lambert
    shade = 0.28 + 0.72 * np.clip(lambert, 0.0, 1.0)
    base = np.array([0.58, 0.67, 0.78])
    colours = np.clip(base[None, :] * shade[:, None], 0.0, 1.0)

    collection = Poly3DCollection(tris, facecolors=colours, edgecolor="none")
    ax.add_collection3d(collection)

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = 0.5 * float(np.max(hi - lo)) or 0.5
    for setter, value in (
        (ax.set_xlim, centre[0]),
        (ax.set_ylim, centre[1]),
        (ax.set_zlim, centre[2]),
    ):
        setter(value - radius, value + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-58)
    return True
