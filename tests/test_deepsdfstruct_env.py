"""Smoke test for the DeepSDFStruct submodule and this project's uv environment.

Run with either:
    uv run pytest tests/test_deepsdfstruct_env.py -v -s
    uv run python tests/test_deepsdfstruct_env.py

Everything here is offline: no pretrained weights are downloaded. The
pretrained-model check is opt-in via DEEPSDF_TEST_PRETRAINED=1.

NOTE: the submodule's own directory is also named DeepSDFStruct, so any script
run from the repo root puts that directory on sys.path, where it can shadow the
installed package as a namespace package and silently skip
DeepSDFStruct/__init__.py. pyproject.toml pins the editable install to
setuptools' path-based (compat) mode so the real package wins either way; the
two test_package_resolves_* checks guard that from both working directories.
"""

import os

import pytest
import torch

from DeepSDFStruct.SDF import SDFBase


class ParametricSphereTile(SDFBase):
    """Minimal lattice microtile: a sphere centred in the [-1, 1]^3 unit cell.

    Its radius is supplied by the lattice's parametrization, which is what
    LatticeSDFStruct requires of a microtile -- plain primitives have no
    ``_set_param`` and cannot be tiled.
    """

    def __init__(self):
        super().__init__()
        self.radius = torch.tensor(0.3)

    def _set_param(self, parameters):
        self.radius = parameters[:, 0:1]

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(queries, dim=1, keepdim=True) - self.radius

    def _get_domain_bounds(self) -> torch.Tensor:
        return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])


def test_package_resolves_to_submodule():
    """The import resolves to the real package, not a shadowing namespace dir."""
    from importlib.metadata import version

    import DeepSDFStruct

    assert DeepSDFStruct.__file__ is not None, (
        "DeepSDFStruct imported as a namespace package -- the submodule "
        "directory is shadowing the installed one and __init__.py never ran"
    )
    assert DeepSDFStruct.__file__.endswith("__init__.py")
    assert DeepSDFStruct.__author__ == "Michael Kofler"
    print(f"DeepSDFStruct {version('DeepSDFStruct')} at {DeepSDFStruct.__file__}")
    print(f"torch {torch.__version__}")


def test_package_resolves_with_repo_root_on_path():
    """Same guarantee for a script run from the repo root.

    There sys.path[0] is the root, so the submodule checkout directory -- also
    named DeepSDFStruct -- competes with the installed package. pytest never
    puts the repo root on sys.path, so the check above cannot see this case and
    needs its own interpreter with the root as cwd.

    This is the configuration that breaks if the editable install reverts to
    setuptools' import-hook finder: the finder is consulted only after the path
    finder has already claimed the checkout directory as a namespace package,
    which leaves submodule imports working while __init__.py never runs. The
    editable_mode=compat setting in pyproject.toml is what keeps it path-based.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "DeepSDFStruct").is_dir(), "expected the submodule checkout"

    # __author__ via getattr: a namespace package has neither attribute, and
    # a bare access would abort the probe before the assertions below can say
    # why.
    probe = (
        "import DeepSDFStruct as d; "
        "print(d.__file__); "
        "print(getattr(d, '__author__', '<missing>'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"probe failed:\n{result.stderr}"
    resolved, author = (line.strip() for line in result.stdout.strip().splitlines())
    assert resolved != "None", (
        "DeepSDFStruct resolved to a namespace package when run from the repo "
        "root -- the submodule directory is shadowing the installed package"
    )
    assert resolved.endswith("__init__.py"), resolved
    assert author == "Michael Kofler"
    print(f"repo-root import resolves to {resolved}")


def test_sphere_sdf_analytic_values():
    """SphereSDF returns exact signed distances at known points."""
    from DeepSDFStruct.sdf_primitives import SphereSDF

    sphere = SphereSDF(center=[0.0, 0.0, 0.0], radius=0.5)
    queries = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # center      -> -0.5
            [0.5, 0.0, 0.0],  # on surface  ->  0.0
            [1.0, 0.0, 0.0],  # outside     -> +0.5
            [0.0, 2.0, 0.0],  # far outside -> +1.5
        ]
    )
    values = sphere(queries)

    assert values.shape == (4, 1), f"expected (4, 1), got {tuple(values.shape)}"
    expected = torch.tensor([[-0.5], [0.0], [0.5], [1.5]])
    torch.testing.assert_close(values, expected, atol=1e-6, rtol=0)


def test_boolean_operations():
    """Union takes the min distance; difference carves the second shape out."""
    from DeepSDFStruct.SDF import DifferenceSDF, UnionSDF
    from DeepSDFStruct.sdf_primitives import SphereSDF

    left = SphereSDF(center=[-0.5, 0.0, 0.0], radius=0.5)
    right = SphereSDF(center=[0.5, 0.0, 0.0], radius=0.5)

    # Point sits inside `left` only, so the union must report it as inside.
    probe = torch.tensor([[-0.5, 0.0, 0.0]])
    union = UnionSDF(left, right)
    assert union(probe).item() == pytest.approx(-0.5, abs=1e-6)

    # Same point, but now carved away by a sphere that contains it.
    carved = DifferenceSDF(left, SphereSDF(center=[-0.5, 0.0, 0.0], radius=0.25))
    assert carved(probe).item() > 0.0, "carved-out point should read as outside"


def test_sdf_is_differentiable():
    """Gradients flow back to SDF parameters -- the framework's core promise."""
    from DeepSDFStruct.sdf_primitives import SphereSDF

    sphere = SphereSDF(center=[0.0, 0.0, 0.0], radius=0.5)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    sphere(queries).sum().backward()

    assert sphere.r.grad is not None, "no gradient reached the radius parameter"
    # d/dr (|q - c| - r) = -1 per query point.
    assert sphere.r.grad.item() == pytest.approx(-2.0, abs=1e-6)


def test_surface_mesh_extraction():
    """FlexiCubes meshing produces a surface that lies on the sphere."""
    from DeepSDFStruct.mesh import create_3D_mesh
    from DeepSDFStruct.sdf_primitives import SphereSDF

    radius = 0.5
    sphere = SphereSDF(center=[0.0, 0.0, 0.0], radius=radius)
    mesh, derivative = create_3D_mesh(sphere, 24, mesh_type="surface")

    assert derivative is None, "no derivative expected when differentiate=False"
    assert mesh.vertices.shape[0] > 0, "meshing produced no vertices"
    assert mesh.faces.shape[1] == 3, "surface mesh should be triangles"

    # Every extracted vertex should sit near the analytic sphere surface.
    distances = torch.linalg.norm(mesh.vertices, dim=1)
    max_error = (distances - radius).abs().max().item()
    print(f"mesh: {mesh.vertices.shape[0]} verts, max radial error {max_error:.4f}")
    assert max_error < 0.05, f"vertices stray from the sphere: {max_error}"


def test_lattice_tiling_and_parametrization():
    """Tiling maps each cell onto the unit cell and feeds the microtile its parameter."""
    from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    from DeepSDFStruct.parametrization import Constant

    radius = 0.3
    lattice = LatticeSDFStruct(
        tiling=(2, 2, 1),
        microtile=ParametricSphereTile(),
        parametrization=Constant([radius]),
    )

    # Centres of two different tiles both map to the unit-cell origin, so each
    # must report exactly -radius. That proves both the periodic transform and
    # the parametrization handoff.
    centres = torch.tensor([[0.25, 0.25, 0.5], [0.75, 0.75, 0.5]])
    values = lattice(centres)

    assert values.shape == (2, 1)
    torch.testing.assert_close(values, torch.full((2, 1), -radius), atol=1e-6, rtol=0)

    # A point far from any sphere centre must read as outside.
    corner = lattice(torch.tensor([[0.0, 0.0, 0.0]]))
    assert corner.item() > 0.0
    assert torch.isfinite(corner).all()


@pytest.mark.skipif(
    os.environ.get("DEEPSDF_TEST_PRETRAINED") != "1",
    reason="downloads weights from HuggingFace; set DEEPSDF_TEST_PRETRAINED=1 to run",
)
def test_pretrained_model_roundtrip():
    """A pretrained DeepSDF decoder evaluates for a given latent vector."""
    from DeepSDFStruct.SDF import SDFfromDeepSDF
    from DeepSDFStruct.pretrained_models import PretrainedModels, get_model

    sdf = SDFfromDeepSDF(get_model(PretrainedModels.AnalyticRoundCross))
    sdf.set_latent_vec(torch.tensor([0.3]))

    values = sdf(torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
    assert values.shape == (2, 1)
    assert torch.isfinite(values).all()


if __name__ == "__main__":
    checks = [
        test_package_resolves_to_submodule,
        test_package_resolves_with_repo_root_on_path,
        test_sphere_sdf_analytic_values,
        test_boolean_operations,
        test_sdf_is_differentiable,
        test_surface_mesh_extraction,
        test_lattice_tiling_and_parametrization,
    ]
    failures = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - smoke test reports, never raises
            failures += 1
            print(f"[FAIL] {check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"[ OK ] {check.__name__}")

    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    raise SystemExit(1 if failures else 0)
