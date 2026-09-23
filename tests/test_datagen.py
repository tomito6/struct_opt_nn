"""Tests of the datagen package: the design space, the exact field, the files.

The field tests compare against a brute-force distance to a densely sampled
boundary, so "exact" in the module docstrings is checked, not assumed. The
file tests go through the command line into ``tmp_path`` and read the result
back the way its consumers do: the GUI's dataset audit for 3-D, the trainer's
own loader and a two-epoch training run for 2-D.
"""

import json

import numpy as np
import pytest

from datagen import dataset, make_plate_hole
from datagen.plate_hole_params import PlateHoleSpace
from datagen.plate_hole_sdf import (
    PlateFrame,
    SamplingConfig,
    band_points,
    extrude,
    plate_hole_sdf,
    sample_instance,
)

# A hole off-centre and close to one edge, the case where plate edge and hole
# wall compete for the nearest-boundary role.
HOLE = (0.3, 0.62, 0.17)


def _boundary_2d(frame, x_c, y_c, r, n=40_000):
    """Dense points on the plate outline and on the hole circle."""
    a, b = frame.half_extents
    centre, radius = frame.hole(x_c, y_c, r)
    t = np.linspace(0.0, 1.0, n // 8, endpoint=False)
    edges = np.vstack(
        [
            np.column_stack([-a + 2 * a * t, np.full_like(t, -b)]),
            np.column_stack([-a + 2 * a * t, np.full_like(t, b)]),
            np.column_stack([np.full_like(t, -a), -b + 2 * b * t]),
            np.column_stack([np.full_like(t, a), -b + 2 * b * t]),
        ]
    )
    theta = np.linspace(0.0, 2 * np.pi, n // 2, endpoint=False)
    circle = centre + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    return np.vstack([edges, circle])


def _inside_2d(points, frame, x_c, y_c, r):
    a, b = frame.half_extents
    centre, radius = frame.hole(x_c, y_c, r)
    in_plate = (np.abs(points[:, 0]) <= a) & (np.abs(points[:, 1]) <= b)
    in_hole = np.linalg.norm(points - centre, axis=1) < radius
    return in_plate & ~in_hole


# ------------------------------------------------------------ parameter space


def test_design_space_reproduces_the_existing_set():
    """The default draw is the 134-shape set of outputs/plate_hole_params.csv
    (made on 17/09); two of its names are pinned so a changed seed shows."""
    params = PlateHoleSpace(margin=0.05).sample(128, method="sobol", seed=0)
    assert len(params) == 134  # 128 Sobol + the 6 distinct extremes
    assert params.all_valid()
    assert len(set(params.names)) == len(params)
    assert params.names[6] == "hole_x0p7664_y0p8278_r0p0889"
    assert params.names[-1] == "hole_x0p7703_y0p3470_r0p1715"


def test_unit_cube_round_trip():
    space = PlateHoleSpace(margin=0.05)
    u = np.random.default_rng(1).random((500, 3))
    back = space.to_unit(*space.from_unit(u))
    assert np.allclose(back, u, atol=1e-12)


def test_the_three_parameter_draw_varies_all_three():
    params = PlateHoleSpace(margin=0.05).sample(16)
    assert params.varied() == ["x_c", "y_c", "r"]


def test_radius_only_family_pins_the_centre():
    """One generating parameter: an even sweep of r at the plate centre, both
    ends included, and a unit description whose first two columns are flat."""
    space = PlateHoleSpace(margin=0.05)
    params = space.sample_radius(40)
    assert len(params) == 40
    assert params.all_valid()
    assert len(set(params.names)) == 40
    assert np.all(params.x_c == 0.5) and np.all(params.y_c == 0.5)
    assert params.r[0] == pytest.approx(space.r_min)
    assert params.r[-1] == pytest.approx(space.r_max_global)  # 0.45
    assert np.allclose(np.diff(params.r), params.r[1] - params.r[0])
    assert params.varied() == ["r"]
    assert np.allclose(params.unit[:, :2], 0.5)
    assert np.allclose(params.unit[:, 2], np.linspace(0.0, 1.0, 40))
    assert params.min_clearance() == pytest.approx(0.05)


def test_radius_only_drawn_methods_keep_the_ends():
    space = PlateHoleSpace(margin=0.05)
    params = space.sample_radius(8, method="sobol", seed=0)
    assert params.varied() == ["r"]
    assert params.r.min() == pytest.approx(space.r_min)
    assert params.r.max() == pytest.approx(space.r_max_global)
    assert 8 <= len(params) <= 10  # the draws plus the two ends, minus collisions
    assert len(set(params.names)) == len(params)

    # Off centre the range is shorter: 0.3 from the left edge minus the margin.
    off = space.sample_radius(5, centre=(0.3, 0.5))
    assert np.all(off.x_c == 0.3)
    assert off.r.max() == pytest.approx(0.25)
    with pytest.raises(ValueError):
        space.sample_radius(5, centre=(0.12, 0.5))  # only r_min fits there
    with pytest.raises(ValueError):
        space.sample_radius(5, method="halton")


# -------------------------------------------------------------------- the field


def test_frame_maps_a_non_square_plate():
    """Hard-coded numbers: every other field test takes the hole from
    frame.hole itself, so a swapped x/y there would pass them all."""
    frame = PlateFrame(length=2.0, width=1.0)  # scale 0.9
    centre, radius = frame.hole(1.5, 0.25, 0.1)
    assert np.allclose(centre, [0.45, -0.225])
    assert radius == pytest.approx(0.09)
    assert frame.half_extents == pytest.approx((0.9, 0.45))
    phi = plate_hole_sdf(2, frame, 1.5, 0.25, 0.1)
    assert phi(np.array([[0.45, -0.225]]))[0] > 0  # in the hole
    assert phi(np.array([[-0.45, -0.225]]))[0] < 0  # mirrored: material


def test_a_2d_frame_needs_no_thickness():
    # A plate measured in small units used to be refused over the unused
    # default thickness.
    frame = PlateFrame(length=0.02, width=0.02)
    assert frame.scale == pytest.approx(90.0)
    with pytest.raises(ValueError):
        frame.half_thickness
    with pytest.raises(ValueError):
        PlateFrame(thickness=2.0)  # h = 1.8 > 1 - pad


def test_2d_field_is_the_exact_distance():
    frame = PlateFrame()
    phi = plate_hole_sdf(2, frame, *HOLE)
    points = np.random.default_rng(0).uniform(-1, 1, (3000, 2))

    boundary = _boundary_2d(frame, *HOLE)
    dist = np.concatenate(
        [
            np.linalg.norm(chunk[:, None, :] - boundary[None, :, :], axis=-1).min(1)
            for chunk in np.array_split(points, 30)
        ]
    )
    sign = np.where(_inside_2d(points, frame, *HOLE), -1.0, 1.0)

    # The boundary spacing is ~2e-4, so the brute force is that accurate.
    assert np.abs(phi(points) - sign * dist).max() < 1e-3


def test_3d_field_is_the_extruded_2d_field():
    frame = PlateFrame(thickness=0.1)
    phi3 = plate_hole_sdf(3, frame, *HOLE)
    phi2 = plate_hole_sdf(2, frame, *HOLE)
    h = frame.half_thickness
    rng = np.random.default_rng(2)

    # Mid-plane, inside the material and far from every side wall: the nearest
    # surface is a face, at distance h.
    p = np.array([[-0.6, -0.6, 0.0]])
    assert phi2(p[:, :2])[0] < -h
    assert phi3(p)[0] == pytest.approx(-h)

    # Straight above the plate: distance to the top face.
    z = rng.uniform(h, 1.0, 200)
    above = np.column_stack([np.full(200, -0.6), np.full(200, -0.6), z])
    assert np.allclose(phi3(above), z - h)

    # Beyond a side wall at mid height: the 2-D distance itself.
    side = np.column_stack([rng.uniform(0.95, 1.0, 200), rng.uniform(-0.5, 0.5, 200)])
    assert np.allclose(phi3(np.column_stack([side, np.zeros(200)])), phi2(side))

    # A true distance has a unit gradient almost everywhere.
    pts = rng.uniform(-1, 1, (4000, 3))
    eps = 1e-5
    grad = np.stack(
        [(phi3(pts + eps * e) - phi3(pts - eps * e)) / (2 * eps) for e in np.eye(3)],
        axis=1,
    )
    assert np.median(np.abs(np.linalg.norm(grad, axis=1) - 1.0)) < 1e-6


def test_3d_field_is_the_exact_distance():
    """Brute force against a dense sampling of the whole plate surface."""
    from scipy.spatial import cKDTree

    frame = PlateFrame(thickness=0.1)
    a, b = frame.half_extents
    h = frame.half_thickness
    centre, radius = frame.hole(*HOLE)
    rng = np.random.default_rng(3)

    n = 400_000
    faces = rng.uniform([-a, -b], [a, b], (n, 2))
    faces = faces[np.linalg.norm(faces - centre, axis=1) >= radius]
    faces = np.column_stack([faces, rng.choice([-h, h], len(faces))])
    outline = _boundary_2d(frame, *HOLE, n=200_000)
    walls = np.column_stack([outline, rng.uniform(-h, h, len(outline))])
    surface = np.vstack([faces, walls])

    points = np.column_stack(
        [rng.uniform(-1, 1, (3000, 2)), rng.uniform(-0.3, 0.3, 3000)]
    )
    dist, _ = cKDTree(surface).query(points)
    in_plane = _inside_2d(points[:, :2], frame, *HOLE)
    sign = np.where(in_plane & (np.abs(points[:, 2]) <= h), -1.0, 1.0)

    phi = plate_hole_sdf(3, frame, *HOLE)(points)
    # The face sampling is ~2e-3 apart, which bounds how close the brute force
    # gets; the sign has no such slack.
    assert np.abs(phi - sign * dist).max() < 5e-3
    assert np.all(np.sign(phi) == sign)


def test_extrude_off_a_corner_is_euclidean():
    # 0.3 out in the plane and 0.4 above the face: the nearest point is the
    # edge, at distance 0.5.
    assert extrude(np.array([0.3]), np.array([0.5]), 0.1)[0] == pytest.approx(0.5)


# ------------------------------------------------------------------ the samples


@pytest.mark.parametrize("dim", [2, 3])
def test_band_puts_its_share_on_the_hole(dim):
    # A vanishing offset keeps every sample on the surface it was drawn from,
    # so "on the hole wall" is exact: nothing else lies on that circle.
    frame = PlateFrame(thickness=0.1)
    config = SamplingConfig(n_uniform=0, n_band=4000, hole_fraction=0.3, stds=[1e-12])
    points = band_points(np.random.default_rng(0), dim, frame, *HOLE, config)
    centre, radius = frame.hole(*HOLE)
    on_hole = np.abs(np.linalg.norm(points[:, :2] - centre, axis=1) - radius) < 1e-9
    assert on_hole.sum() == 1200
    # And every band sample sits on the zero level set.
    assert np.abs(plate_hole_sdf(dim, frame, *HOLE)(points)).max() < 1e-9


def test_samples_are_reproducible_and_float32():
    frame, config = PlateFrame(), SamplingConfig(n_uniform=300, n_band=300)
    one = sample_instance(np.random.default_rng([0, 5]), 2, frame, *HOLE, config)
    two = sample_instance(np.random.default_rng([0, 5]), 2, frame, *HOLE, config)
    assert one.dtype == np.float32 and one.shape == (600, 3)
    assert np.array_equal(one, two)


# -------------------------------------------------------------------- the files


def _make(tmp_path, dim, *extra):
    argv = [
        "--dim",
        str(dim),
        "--n",
        "4",
        "--no-extremes",
        "--n-uniform",
        "400",
        "--n-band",
        "400",
        "--data-root",
        str(tmp_path),
        *extra,
    ]
    return make_plate_hole.main(argv)


def test_2d_dataset_layout(tmp_path):
    paths = _make(tmp_path, 2)

    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    names = split["plate_hole_2d"]["plate"]
    assert len(names) == 4

    manifest = dataset.read_manifest(paths["dataset_dir"])
    assert manifest["geom_dimension"] == 2
    assert manifest["columns"] == ["x", "y", "phi"]
    assert manifest["parameters"]["shown_to_network"] is False

    # params.csv row i is split entry i is latent code i.
    rows = paths["params"].read_text(encoding="utf-8").splitlines()
    assert [r.split(",")[0] for r in rows[1:]] == names

    for name in names:
        with np.load(paths["class_dir"] / f"{name}.npz") as npz:
            pos, neg = npz["pos"], npz["neg"]
        assert pos.dtype == neg.dtype == np.float32
        assert pos.shape[1] == neg.shape[1] == 3
        assert (pos[:, -1] >= 0).all() and (neg[:, -1] < 0).all()
        assert len(pos) + len(neg) == 800


def test_existing_dataset_is_not_overwritten_silently(tmp_path):
    _make(tmp_path, 2)
    with pytest.raises(FileExistsError):
        _make(tmp_path, 2)

    # Another seed draws other holes, so other instance names: any .npz of the
    # first run left behind would show up next to the new split.
    paths = _make(tmp_path, 2, "--overwrite", "--seed", "3", "--n-uniform", "100")
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    on_disk = {p.stem for p in paths["class_dir"].glob("*.npz")}
    assert on_disk == set(split["plate_hole_2d"]["plate"])
    assert len(on_disk) == 4
    with np.load(paths["class_dir"] / f"{sorted(on_disk)[0]}.npz") as npz:
        assert len(npz["pos"]) + len(npz["neg"]) == 500


def test_overwrite_leaves_other_classes_alone(tmp_path):
    paths = _make(tmp_path, 2)
    other = paths["dataset_dir"] / "someone_else"
    other.mkdir()
    np.savez(other / "keep.npz", pos=np.zeros((1, 3)), neg=np.zeros((1, 3)))
    with pytest.raises(FileExistsError):
        _make(tmp_path, 2, "--overwrite")
    assert (other / "keep.npz").is_file()


def test_radius_only_dataset_says_so(tmp_path):
    """The manifest states what varied, params.csv carries the fixed centre,
    and every .npz holds the field of the hole its row describes."""
    from datagen.plate_hole_params import PlateHoleSpace as Space

    paths = _make(tmp_path, 2, "--radius-only")
    assert paths["dataset_dir"].name == "plate_hole_2d_r"

    manifest = dataset.read_manifest(paths["dataset_dir"])
    assert manifest["parameters"]["varied"] == ["r"]
    assert manifest["parameters"]["fixed_centre"] == [0.5, 0.5]
    assert manifest["parameter_sampling"]["family"] == "radius_only"
    assert manifest["parameter_sampling"]["method"] == "grid"
    assert manifest["parameter_sampling"]["n_requested"] == 4

    lines = paths["params"].read_text(encoding="utf-8").splitlines()
    rows = [line.split(",") for line in lines[1:]]
    names = json.loads(paths["split"].read_text(encoding="utf-8"))
    names = names["plate_hole_2d_r"]["plate"]
    assert [r[0] for r in rows] == names and len(names) == 4
    assert {r[1] for r in rows} == {"0.500000"} and {r[2] for r in rows} == {"0.500000"}
    radii = [float(r[3]) for r in rows]
    assert radii == sorted(radii)
    assert radii[0] == pytest.approx(0.07) and radii[-1] == pytest.approx(0.45)
    assert Space(margin=0.05).is_valid(0.5, 0.5, radii).all()

    frame = PlateFrame()
    for name, r in zip(names, radii):
        with np.load(paths["class_dir"] / f"{name}.npz") as npz:
            stored = np.vstack([npz["pos"], npz["neg"]])
        phi = plate_hole_sdf(2, frame, 0.5, 0.5, r)(stored[:, :2])
        assert np.abs(phi - stored[:, 2]).max() < 1e-5

    # The default family keeps saying so too.
    paths = _make(tmp_path, 2)
    manifest = dataset.read_manifest(paths["dataset_dir"])
    assert manifest["parameters"]["varied"] == ["x_c", "y_c", "r"]
    assert manifest["parameters"]["fixed_centre"] is None
    assert manifest["parameter_sampling"]["family"] == "centre_and_radius"
    assert manifest["parameter_sampling"]["method"] == "sobol"

    with pytest.raises(SystemExit):
        _make(tmp_path, 2, "--centre", "0.3", "0.3", "--name", "misuse")


def test_3d_dataset_passes_the_gui_audit(tmp_path):
    from structsept.app import datasets as gui_datasets

    paths = _make(tmp_path, 3)
    listed = gui_datasets.list_datasets(tmp_path)
    assert [(d["name"], d["n_instances"]) for d in listed] == [("plate_hole_3d", 4)]
    assert listed[0]["split"] == str(paths["split"])

    report = gui_datasets.validate_dataset(paths["dataset_dir"], log=lambda *_: None)
    assert report["problems"] == []


def test_2d_dataset_trains_a_2d_decoder(tmp_path):
    """The trainer accepts geom_dimension 2 and returns one 2-D code per shape,
    with the dimension read off the dataset the way the Train tab reads it."""
    import torch
    from DeepSDFStruct.deep_sdf.data import unpack_sdf_samples

    from structsept.app import datasets, training

    paths = _make(tmp_path / "data", 2)
    first = next(paths["class_dir"].glob("*.npz"))
    assert unpack_sdf_samples(str(first), 2, subsample=100).shape == (100, 3)

    geom = datasets.geom_dimension(paths["dataset_dir"])
    assert geom == 2
    run_dir = tmp_path / "runs" / "plate_2d"
    training.write_specs(
        run_dir,
        paths["split"],
        paths["data_root"],
        geom_dimension=geom,
        latent_dim=2,
        num_epochs=2,
        scenes_per_batch=4,
        samples_per_scene=200,
    )
    training.train(run_dir, paths["data_root"], log=lambda *_: None)

    codes = torch.load(run_dir / "LatentCodes" / "latest.pth", weights_only=False)
    assert tuple(codes["latent_codes"]["weight"].shape) == (4, 2)
