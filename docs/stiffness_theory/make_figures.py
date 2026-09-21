"""Theory figures for the PDF: the transformation function T(x), an SDF in 1D,
and the spectra of k_e and K measured on the solid plate."""

import sys, pathlib, json
import numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

out = pathlib.Path(sys.argv[1])

# --- 1. transformation function, eq. (18) of the paper, for t_x = 4 --------
t = 4
x = np.linspace(0, 1, 2001)
T = 4 * np.abs(t * x / 2 - np.floor((t * x + 1) / 2)) - 1
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.plot(x, T, color="k", lw=1.8)
for k in range(1, t):
    ax.axvline(k / t, color="0.6", ls="--", lw=0.8)
ax.set_xlabel(r"$x$  (coordenada paramétrica da placa)")
ax.set_ylabel(r"$T(x)$  (entrada da rede)")
ax.set_yticks([-1, 0, 1])
ax.set_xticks([k / t for k in range(t + 1)])
ax.set_title(
    r"$T(x) = 4\,|\,t_x x/2 - \lfloor t_x x/2 + 1/2 \rfloor\,| - 1$,  $t_x = 4$",
    fontsize=10,
)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out / "transformation.png", dpi=200)

# --- 2. a 1D signed distance function --------------------------------------
xs = np.linspace(-1.5, 1.5, 601)
a, b = -0.6, 0.4  # a "solid" segment [a, b]
phi = np.maximum(a - xs, xs - b)  # SDF of a segment: negative inside
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.plot(xs, phi, color="k", lw=1.8, label=r"$\varphi(x)$")
ax.axhspan(-2, 0, color="0.85", zorder=0)
ax.fill_between(
    xs,
    phi,
    0,
    where=phi < 0,
    color="steelblue",
    alpha=0.35,
    label=r"$\Omega$: $\varphi<0$ (material)",
)
ax.axhline(0, color="0.4", lw=0.8)
for p in (a, b):
    ax.plot([p], [0], "o", color="crimson", ms=6)
ax.text(a, 0.12, r"$\Gamma$", ha="center", color="crimson")
ax.text(b, 0.12, r"$\Gamma$", ha="center", color="crimson")
ax.set_xlabel("x")
ax.set_ylim(-0.9, 1.2)
ax.legend(loc="upper center", fontsize=9, ncol=2)
ax.set_title(
    r"Distância com sinal de um segmento: $|\varphi'| = 1$, zero na superfície",
    fontsize=10,
)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out / "sdf_1d.png", dpi=200)

# --- 3. spectra measured on the solid plate --------------------------------
from structsept.plate_with_hole import plate_with_hole
from structsept.fem import (
    tetrahedral_mesh,
    build_solid,
    assemble_stiffness,
    default_dtype,
)

plate, defo = plate_with_hole(solid=True, device="cpu")
V, Tt = tetrahedral_mesh(plate, defo, resolution=10)
with default_dtype(torch.float64):
    solid = build_solid(V, Tt)
    k, K = assemble_stiffness(solid)
    we = torch.linalg.eigvalsh(k[0]).numpy()
    wK = torch.linalg.eigvalsh(K.to_dense()).numpy()
numbers = dict(
    ke_eig=we.tolist(),
    K_eig_smallest=wK[:10].tolist(),
    K_eig_max=float(wK[-1]),
    n_nodes=int(solid.n_nod),
    n_elem=int(solid.n_elem),
    n_dofs=int(solid.n_dofs),
    nnz=int(K._nnz()),
)
json.dump(numbers, open(out / "numbers_solid.json", "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
idx = np.arange(1, 13)
axs[0].semilogy(idx, np.abs(we), "o", color="k")
axs[0].axvspan(0.5, 6.5, color="crimson", alpha=0.12)
axs[0].set_title(r"$|\lambda_i|$ de uma $\mathbf{k}_e$ (12 × 12)", fontsize=10)
axs[0].set_xlabel("i")
axs[0].set_xticks(idx)
axs[1].semilogy(np.arange(1, 11), np.abs(wK[:10]), "o", color="k")
axs[1].axvspan(0.5, 6.5, color="crimson", alpha=0.12)
axs[1].set_title(
    r"10 menores $|\lambda_i|$ de $\mathbf{K}$ (%d × %d)" % (K.shape[0], K.shape[1]),
    fontsize=10,
)
axs[1].set_xlabel("i")
axs[1].set_xticks(np.arange(1, 11))
for ax in axs:
    ax.grid(alpha=0.2, which="both")
    ax.set_ylabel("N/m")
fig.tight_layout()
fig.savefig(out / "spectra.png", dpi=200)
print(json.dumps(numbers, indent=1))
