"""Reusable code for the Struct_Sept lattice optimization work.

This package holds everything that more than one thing uses. The rule that
keeps the repo legible is one-directional::

    experiments/  --imports-->  structsept/  --imports-->  DeepSDFStruct/

Nothing in ``structsept`` may import from ``experiments``, and nothing outside
``structsept`` may be imported by two different places -- the moment a second
caller appears, the code moves in here.

Modules
-------
``plate_with_hole``
    The recurring test geometry: a lattice plate with a circular hole, plus
    ``ScaledSpaceSDF``, the bridge between the parametric unit cube the
    lattice lives on and the physical metres the hole is described in.
``plate_hole_params``
    The admissible design space of that geometry -- which ``(x_c, y_c, r)``
    triples are a plate with a hole and not a notch. Pure geometry of the
    parameters; imports nothing from DeepSDFStruct.
``pointcloud_sdf``
    A signed distance field built from a raw, unoriented point cloud
    (winding number for the sign, KD-tree for the distance), so a scan can
    feed the reconstruction path that normally needs a watertight mesh.
``fem``
    Tetrahedral meshing and stiffness assembly on top of torch-fem.
``app``
    The Tkinter application: sample a dataset, train a decoder, then explore
    ``f_theta(lambda(x), x)`` by dragging the latent spline control points.
"""
