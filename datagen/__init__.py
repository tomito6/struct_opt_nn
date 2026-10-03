"""Where the SDF training data comes from.

The rest of the repo starts from the assumption that signed-distance samples
already exist: the GUI trains on whatever sits in ``data/``, the optimization
loop loads a trained decoder. This package is the step before that -- it draws
the shapes, samples their signed distance and writes a training set. It knows
nothing about networks.

The boundary is a folder, not an import::

    datagen/  --writes-->  data/  <--reads--  structsept.app  (Train tab)

``datagen`` imports nothing from ``structsept``, ``experiments`` or
``DeepSDFStruct``, and nothing imports ``datagen`` except its tests. The file
layout both sides agree on is documented in :mod:`datagen.dataset`.

Modules
-------
``plate_hole_params``
    Which ``(x_c, y_c, r)`` triples are a plate with a hole and not a notch,
    and a Sobol/LHS/grid draw of them.
``plate_hole_sdf``
    The exact signed distance of one such plate, in 2-D or 3-D, in the
    ``[-1, 1]^d`` frame the decoder works in, and the samples of it.
``plate_tri_params``, ``plate_tri_sdf``
    The same two steps for the square plate with four triangular holes,
    whose one parameter is the shared triangle height ``h``. The frame,
    the extrusion and the sampling recipe are the hole modules', imported.
``dataset``
    The on-disk contract: ``.npz`` files, split, parameter table, manifest.
``preview``
    The preview figure of a written dataset, shared by the ``make_*`` scripts.
``make_plate_hole``, ``make_plate_tri``
    The command lines that tie them together::

        uv run python -m datagen.make_plate_hole --dim 2
        uv run python -m datagen.make_plate_tri --dim 2
"""
