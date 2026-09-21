"""Dataset stage of the structsept app.

Turns a folder of meshes into an SdfSamples dataset that ``train_deep_sdf`` can
read, and audits a dataset that already exists on disk. This is step 1 of the
offline pipeline: sample points around each geometry and store the signed
distances. Everything downstream inherits the quality of what is written here.
"""

import json
import pathlib
import time

import numpy as np
import trimesh

from DeepSDFStruct.sampling import SDFSampler

SDF_SAMPLES_DIR = "SdfSamples"
SPLITS_DIR = "splits"

# Surface samples are pushed off the surface along the normal with a Gaussian of
# std up to 0.05, so a few sigma past the normalized [-1, 1] box is expected.
XYZ_TOL = 0.2

# A geometry normalized into [-1, 1]^3 touches the box, so no sample point can be
# much farther from it than the box half-diagonal. More than this means the
# geometry was never scaled down.
PHI_MAX_EXPECTED = 2.5

MIN_INSIDE_FRACTION = 0.02
MAX_INSIDE_FRACTION = 0.98
PHI_SAMPLE_CAP = 200_000


def list_datasets(data_root):
    """List the datasets stored under ``<data_root>/SdfSamples``.

    Parameters
    ----------
    data_root : str or pathlib.Path
        Root that holds ``SdfSamples/`` and ``splits/``.

    Returns
    -------
    list of dict
        One dict per dataset with keys ``name``, ``path``, ``n_instances``,
        ``classes`` and ``split`` (path to the split json, or None).
    """
    data_root = pathlib.Path(data_root)
    samples_root = data_root / SDF_SAMPLES_DIR
    if not samples_root.is_dir():
        return []

    datasets = []
    for dataset_dir in sorted(p for p in samples_root.iterdir() if p.is_dir()):
        classes = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
        n_instances = sum(len(list((dataset_dir / c).glob("*.npz"))) for c in classes)
        split_path = data_root / SPLITS_DIR / f"{dataset_dir.name}.json"
        datasets.append(
            {
                "name": dataset_dir.name,
                "path": str(dataset_dir),
                "n_instances": n_instances,
                "classes": classes,
                "split": str(split_path) if split_path.is_file() else None,
            }
        )
    return datasets


def load_meshes(folder, ext="stl", log=print):
    """Load every mesh of a given extension from a folder.

    Files that fail to load are skipped with a warning. Meshes that are not
    watertight are loaded but flagged: ``SDFfromMesh`` takes the inside/outside
    sign from a winding number, which is meaningless on an open surface.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder to scan, not recursive.
    ext : str, default "stl"
        File extension, with or without the leading dot.
    log : callable, default print
        Receives short Portuguese progress lines.

    Returns
    -------
    list of trimesh.Trimesh
        Loaded meshes, sorted by file name.
    """
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        log(f"Pasta nao encontrada: {folder}")
        return []

    meshes = []
    not_watertight = []
    for path in sorted(folder.glob("*." + ext.lstrip(".").lower())):
        try:
            mesh = trimesh.load_mesh(path)
        except Exception as exc:
            log(f"Ignorado {path.name}: nao deu para ler ({exc})")
            continue

        if isinstance(mesh, trimesh.Scene):
            parts = list(mesh.geometry.values())
            if not parts:
                log(f"Ignorado {path.name}: cena vazia")
                continue
            mesh = trimesh.util.concatenate(parts)

        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            log(f"Ignorado {path.name}: nao e uma malha de triangulos")
            continue

        mesh.metadata["file_name"] = path.name
        if not mesh.is_watertight:
            not_watertight.append(path.name)
        meshes.append(mesh)

    log(f"{len(meshes)} malha(s) carregada(s) de {folder}")
    if not_watertight:
        log(
            f"Aviso: {len(not_watertight)} malha(s) nao sao watertight "
            f"({', '.join(not_watertight)}). O sinal do SDF fica sem sentido nelas."
        )
    return meshes


def make_dataset(
    meshes,
    data_root,
    dataset_name,
    class_name,
    n_samples=50000,
    add_surface_samples=True,
    log=print,
):
    """Sample a list of meshes into an SdfSamples dataset plus its split file.

    Writes ``<data_root>/SdfSamples/<dataset_name>/<class_name>/*.npz`` and
    ``<data_root>/splits/<dataset_name>.json``. The ``SdfSamples`` level is not
    cosmetic: the sampler writes under ``outdir`` while training looks under
    ``<data_source>/SdfSamples``, so ``outdir`` has to be that subfolder.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
        Geometries to sample. Each is normalized into [-1, 1]^3 by the sampler.
    data_root : str or pathlib.Path
        Becomes ``DataSource`` in specs.json.
    dataset_name, class_name : str
        Dataset and class folder names.
    n_samples : int, default 50000
        Uniform samples per geometry; surface sampling adds ``n_samples // 2``
        more per noise level.
    add_surface_samples : bool, default True
        Concentrate extra samples near the zero level set.
    log : callable, default print
        Receives short Portuguese progress lines.

    Returns
    -------
    dict
        ``dataset_dir``, ``split_path`` and ``n_instances``.
    """
    data_root = pathlib.Path(data_root)
    outdir = data_root / SDF_SAMPLES_DIR
    splitdir = data_root / SPLITS_DIR
    dataset_dir = outdir / dataset_name
    class_dir = dataset_dir / class_name
    split_path = splitdir / f"{dataset_name}.json"

    if not meshes:
        raise ValueError("Nenhuma geometria para amostrar.")

    outdir.mkdir(parents=True, exist_ok=True)
    splitdir.mkdir(parents=True, exist_ok=True)

    log(f"Dataset '{dataset_name}', classe '{class_name}' -> {class_dir}")
    log(f"{len(meshes)} geometria(s), {int(n_samples)} amostras uniformes cada")

    open_meshes = [
        m for m in meshes if isinstance(m, trimesh.Trimesh) and not m.is_watertight
    ]
    if open_meshes:
        log(
            f"Aviso: {len(open_meshes)} malha(s) nao sao watertight; "
            "dentro/fora vai sair errado nelas."
        )

    # The sampler skips any .npz that already exists, so overwrite_existing on its
    # own would leave stale files in place.
    stale = sorted(class_dir.glob("*.npz")) + sorted(class_dir.glob("*.stl"))
    if stale:
        log(f"Removendo {len(stale)} arquivo(s) antigo(s) de {class_dir.name}")
        for path in stale:
            path.unlink()

    sampler = SDFSampler(
        outdir=str(outdir),
        splitdir=str(splitdir),
        dataset_name=dataset_name,
        overwrite_existing=True,
    )
    # SDFfromMesh normalizes in place, which would silently rescale the caller's
    # meshes.
    sampler.add_class([_copy_geometry(m) for m in meshes], class_name)

    log("Amostrando... (a parte demorada)")
    started = time.perf_counter()
    sampler.process_geometries(
        n_samples=int(n_samples),
        add_surface_samples=add_surface_samples,
        also_save_mesh=True,
        scale=True,
        n_workers=0,
    )
    log(f"Amostragem concluida em {time.perf_counter() - started:.1f} s")

    sampler.write_json(split_path.name)
    log(f"Split escrito em {split_path}")

    n_instances = len(list(class_dir.glob("*.npz")))
    log(f"Pronto: {n_instances} instancia(s) em {dataset_dir}")
    return {
        "dataset_dir": str(dataset_dir),
        "split_path": str(split_path),
        "n_instances": n_instances,
    }


def validate_dataset(dataset_dir, log=print):
    """Audit a dataset on disk for the defects that quietly ruin a training run.

    Checks every ``.npz`` for missing inside samples, NaNs, points outside the
    unit cube, an unnormalized phi range and a badly skewed inside/outside ratio.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        A ``<data_root>/SdfSamples/<dataset_name>`` folder.
    log : callable, default print
        Receives short Portuguese progress lines.

    Returns
    -------
    dict
        ``instances`` (one dict per .npz), ``problems`` (Portuguese strings) and
        ``phi_all`` (subsampled phi values, for a histogram).
    """
    dataset_dir = pathlib.Path(dataset_dir)
    instances = []
    problems = []
    phi_chunks = []

    files = sorted(dataset_dir.glob("**/*.npz"))
    if not files:
        problems.append(f"Nenhum arquivo .npz encontrado em {dataset_dir}.")
        log(problems[-1])
        return {"instances": [], "problems": problems, "phi_all": np.zeros(0)}

    log(f"Validando {len(files)} instancia(s) em {dataset_dir}")
    per_file_cap = max(1, PHI_SAMPLE_CAP // len(files))

    for path in files:
        name = path.stem
        try:
            with np.load(path) as npz:
                pos = _read_block(npz, "pos")
                neg = _read_block(npz, "neg")
        except Exception as exc:
            problems.append(f"{name}: arquivo ilegivel ({exc}).")
            log(problems[-1])
            continue

        if pos is None or neg is None:
            missing = "pos" if pos is None else "neg"
            problems.append(f"{name}: falta o array '{missing}' no .npz.")
            log(problems[-1])
            empty = np.zeros((0, 4))
            pos = empty if pos is None else pos
            neg = empty if neg is None else neg

        rows = np.vstack([pos, neg])
        n_pos, n_neg = len(pos), len(neg)
        n_total = n_pos + n_neg
        phi = rows[:, 3]
        n_nan = int(np.isnan(rows).any(axis=1).sum())
        finite = rows[np.isfinite(rows).all(axis=1)]
        inside_fraction = n_neg / n_total if n_total else 0.0

        if len(finite):
            xyz_min = finite[:, :3].min(axis=0)
            xyz_max = finite[:, :3].max(axis=0)
            phi_min = float(finite[:, 3].min())
            phi_max = float(finite[:, 3].max())
        else:
            xyz_min = xyz_max = np.full(3, np.nan)
            phi_min = phi_max = float("nan")

        instances.append(
            {
                "name": name,
                "class": path.parent.name,
                "path": str(path),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_nan": n_nan,
                "phi_min": phi_min,
                "phi_max": phi_max,
                "xyz_min": xyz_min,
                "xyz_max": xyz_max,
                "inside_fraction": inside_fraction,
            }
        )

        if n_total == 0:
            problems.append(f"{name}: arquivo vazio, nenhuma amostra.")
        elif n_neg == 0:
            problems.append(
                f"{name}: nenhuma amostra interna (neg vazio). A geometria nunca "
                "fechou volume - malha aberta ou sinal invertido."
            )
        elif n_pos == 0:
            problems.append(
                f"{name}: nenhuma amostra externa (pos vazio). "
                "Sinal provavelmente invertido."
            )
        elif not MIN_INSIDE_FRACTION <= inside_fraction <= MAX_INSIDE_FRACTION:
            problems.append(
                f"{name}: razao dentro/fora muito desequilibrada "
                f"({100 * inside_fraction:.1f}% dentro)."
            )
        if n_nan:
            problems.append(f"{name}: {n_nan} linha(s) com NaN.")
        if len(finite):
            worst_xyz = float(np.abs(finite[:, :3]).max())
            if worst_xyz > 1.0 + XYZ_TOL:
                problems.append(
                    f"{name}: pontos fora do cubo unitario (|x| ate {worst_xyz:.2f}). "
                    "O decoder so conhece [-1, 1]^3."
                )
            worst_phi = max(abs(phi_min), abs(phi_max))
            if worst_phi > PHI_MAX_EXPECTED:
                problems.append(
                    f"{name}: phi chega a {worst_phi:.2f}, "
                    "a geometria parece nao normalizada."
                )
            elif worst_phi < 1e-6:
                problems.append(f"{name}: phi e praticamente zero em todo lugar.")

        log(
            f"  {name}: {n_pos} fora / {n_neg} dentro, "
            f"phi [{phi_min:.3f}, {phi_max:.3f}], NaN {n_nan}"
        )

        finite_phi = phi[np.isfinite(phi)]
        if len(finite_phi):
            take = min(per_file_cap, len(finite_phi))
            idx = np.random.choice(len(finite_phi), size=take, replace=False)
            phi_chunks.append(finite_phi[idx])

    if problems:
        log(f"{len(problems)} problema(s) encontrado(s).")
    else:
        log("Nenhum problema encontrado.")

    phi_all = np.concatenate(phi_chunks) if phi_chunks else np.zeros(0)
    return {"instances": instances, "problems": problems, "phi_all": phi_all}


def read_split(split_path):
    """Read a split json as ``{dataset: {class: [instance_names]}}``."""
    with open(split_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _copy_geometry(geom):
    return geom.copy() if isinstance(geom, trimesh.Trimesh) else geom


def _read_block(npz, key):
    """Read a pos/neg block as (N, 4), tolerating the legacy '<key>.npy' name."""
    for candidate in (key, key + ".npy"):
        if candidate in npz:
            return np.asarray(npz[candidate], dtype=np.float64).reshape(-1, 4)
    return None
