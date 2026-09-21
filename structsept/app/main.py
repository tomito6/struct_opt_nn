"""GUI shell of the structsept app.

Tkinter front end over the three pipeline stages: sample a folder of meshes into
an SdfSamples dataset, train a DeepSDF decoder on it, then explore
f_theta(lambda(x), x) by dragging the latent B-spline control points. Sampling,
training and meshing run on worker threads; the Tk loop only draws.
"""

from __future__ import annotations

import contextlib
import queue
import signal
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np

# The trainer draws its loss curve with pyplot from inside the worker thread; on
# an interactive backend that builds Tk widgets off the main loop and kills the
# whole app. Our own canvases are FigureCanvasTkAgg instances built by hand, so
# they do not care what the pyplot backend is.
matplotlib.use("Agg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from structsept.app import datasets, models, training, viz  # noqa: E402

# Datasets and training runs are large, regenerable and gitignored, so they
# live at the repo root rather than inside the package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

POLL_MS = 100
DEBOUNCE_MS = 120
SETTLE_MS = 400
SLICE_RES = 64
MAX_LATENT_PANELS = 3
MANY_SLIDERS = 128

# A 2D latent space only interpolates sensibly if the shapes fill it; the paper
# trained 120 unit cells for d = 2.
MIN_SHAPES_2D = 40


# --------------------------------------------------------------------------- #
# small widget helpers
# --------------------------------------------------------------------------- #


def _log_box(parent, height=12):
    """Read-only scrolling text widget."""
    frame = ttk.Frame(parent)
    text = tk.Text(frame, height=height, wrap="word", state="disabled")
    bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    return frame, text


def _append(text, line):
    text.configure(state="normal")
    text.insert("end", str(line) + "\n")
    text.see("end")
    text.configure(state="disabled")


def _scrollable(parent, width=400):
    """Canvas-backed frame that scrolls vertically. Returns the inner frame."""
    canvas = tk.Canvas(parent, highlightthickness=0, width=width)
    bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=bar.set)

    inner.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))

    def wheel(event):
        canvas.yview_scroll(int(-event.delta / 120), "units")

    # bind_all only while the pointer is over the list, otherwise the wheel is
    # hijacked for the whole window
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", wheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    canvas.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    return inner


def _figure_canvas(parent, figsize):
    fig = Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return fig, canvas


def _spinbox(parent, var, low, high, step=1, width=7):
    return ttk.Spinbox(
        parent,
        textvariable=var,
        from_=low,
        to=high,
        increment=step,
        width=width,
    )


def _read_int(var, fallback):
    """Spinboxes accept typed text, so a value can be empty or garbage."""
    try:
        return int(var.get())
    except (tk.TclError, ValueError):
        var.set(fallback)
        return fallback


# --------------------------------------------------------------------------- #
# worker threads
# --------------------------------------------------------------------------- #


def _run_worker(st, button, log_widget, work):
    """Run ``work(log)`` off the Tk loop, funnelling output through the queue."""
    if st["busy"]:
        messagebox.showinfo("Ocupado", "Espere a tarefa atual terminar.")
        return
    st["busy"] = True
    if button is not None:
        button.configure(state="disabled")

    def log(line):
        st["queue"].put(("log", (log_widget, line)))

    def target():
        try:
            work(log)
        except Exception as exc:
            log(f"ERRO: {exc}")
            st["queue"].put(
                ("call", lambda exc=exc: messagebox.showerror("Erro", str(exc)))
            )
        finally:
            st["queue"].put(("call", lambda: _worker_done(st, button)))

    threading.Thread(target=target, daemon=True).start()


@contextlib.contextmanager
def _signals_off():
    """Neutralize signal.signal while a worker runs.

    The DeepSDFStruct trainer installs a SIGINT handler so a terminal run can be
    stopped with Ctrl-C; off the main thread that raises instead.
    """
    original = signal.signal
    signal.signal = lambda *args, **kwargs: None
    try:
        yield
    finally:
        signal.signal = original


def _worker_done(st, button):
    st["busy"] = False
    if button is not None:
        button.configure(state="normal")


def _poll(st):
    """Drain the worker queue on the Tk loop."""
    while True:
        try:
            kind, payload = st["queue"].get_nowait()
        except queue.Empty:
            break
        if kind == "log":
            _append(*payload)
        else:
            payload()
    st["root"].after(POLL_MS, lambda: _poll(st))


# --------------------------------------------------------------------------- #
# shared registries
# --------------------------------------------------------------------------- #


def _refresh_datasets(st):
    rows = datasets.list_datasets(DATA_ROOT)
    st["datasets"] = {f"{r['name']} ({r['n_instances']} formas)": r for r in rows}
    values = list(st["datasets"])
    for combo in (st["combo_data"], st["combo_train"]):
        combo.configure(values=values)
        if values and combo.get() not in st["datasets"]:
            combo.set(values[0])


def _refresh_models(st):
    entries = models.list_models(RUNS_DIR)
    labels = {}
    for entry in entries:
        d = "?" if entry.latent_dim is None else entry.latent_dim
        tag = "treinado aqui" if entry.source == "run" else "pronto"
        labels[f"{entry.name} - d={d}, {entry.n_latents} formas ({tag})"] = entry
    st["models"] = labels
    st["combo_model"].configure(values=list(labels))
    if labels and st["combo_model"].get() not in labels:
        st["combo_model"].set(next(iter(labels)))


# --------------------------------------------------------------------------- #
# aba 1 - SDF maker
# --------------------------------------------------------------------------- #


def _build_tab_data(st, parent):
    st["folder"] = tk.StringVar()
    st["ext"] = tk.StringVar(value="stl")
    st["ds_name"] = tk.StringVar(value="meu_dataset")
    st["class_name"] = tk.StringVar(value="celulas")
    st["n_samples"] = tk.IntVar(value=50000)
    st["surface_samples"] = tk.BooleanVar(value=True)

    form = ttk.LabelFrame(parent, text="Gerar dataset", padding=8)
    form.pack(fill="x")
    form.columnconfigure(1, weight=1)

    ttk.Label(form, text="Pasta com malhas:").grid(row=0, column=0, sticky="w")
    ttk.Entry(form, textvariable=st["folder"]).grid(
        row=0, column=1, sticky="ew", padx=4
    )
    ttk.Button(form, text="Procurar...", command=lambda: _pick_folder(st)).grid(
        row=0, column=2
    )

    line = ttk.Frame(form)
    line.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
    ttk.Label(line, text="Extensao:").pack(side="left")
    ttk.Entry(line, textvariable=st["ext"], width=6).pack(side="left", padx=(4, 12))
    ttk.Label(line, text="Dataset:").pack(side="left")
    ttk.Entry(line, textvariable=st["ds_name"], width=18).pack(
        side="left", padx=(4, 12)
    )
    ttk.Label(line, text="Classe:").pack(side="left")
    ttk.Entry(line, textvariable=st["class_name"], width=18).pack(
        side="left", padx=(4, 12)
    )
    ttk.Label(line, text="Amostras/geometria:").pack(side="left")
    _spinbox(line, st["n_samples"], 1000, 500000, 1000, width=9).pack(
        side="left", padx=(4, 12)
    )
    ttk.Checkbutton(
        line, text="amostras na superficie", variable=st["surface_samples"]
    ).pack(side="left")

    st["btn_generate"] = ttk.Button(
        form, text="Gerar dataset", command=lambda: _gerar_dataset(st)
    )
    st["btn_generate"].grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

    check = ttk.LabelFrame(parent, text="Conferir dataset", padding=8)
    check.pack(fill="x", pady=(8, 0))
    ttk.Label(check, text="Dataset:").pack(side="left")
    st["combo_data"] = ttk.Combobox(check, state="readonly", width=40)
    st["combo_data"].pack(side="left", padx=4)
    ttk.Button(check, text="Atualizar", command=lambda: _refresh_datasets(st)).pack(
        side="left", padx=4
    )
    st["btn_validate"] = ttk.Button(check, text="Validar", command=lambda: _validar(st))
    st["btn_validate"].pack(side="left", padx=4)

    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True, pady=(8, 0))
    frame, st["log_data"] = _log_box(body)
    frame.pack(side="left", fill="both", expand=True)
    right = ttk.Frame(body, width=460)
    right.pack(side="right", fill="both")
    st["fig_hist"], st["canvas_hist"] = _figure_canvas(right, (4.6, 3.2))


def _pick_folder(st):
    folder = filedialog.askdirectory(title="Pasta com as malhas")
    if folder:
        st["folder"].set(folder)


def _gerar_dataset(st):
    folder = st["folder"].get().strip()
    if not folder or not Path(folder).is_dir():
        messagebox.showerror("Pasta invalida", "Escolha uma pasta que exista.")
        return
    name = st["ds_name"].get().strip()
    class_name = st["class_name"].get().strip()
    if not name or not class_name:
        messagebox.showerror("Faltam nomes", "Preencha o dataset e a classe.")
        return

    ext = st["ext"].get().strip() or "stl"
    meshes = datasets.load_meshes(folder, ext, log=lambda t: _append(st["log_data"], t))
    if not meshes:
        messagebox.showerror("Sem malhas", f"Nenhum arquivo .{ext} legivel na pasta.")
        return

    open_meshes = [m for m in meshes if not m.is_watertight]
    if open_meshes:
        _append(
            st["log_data"],
            f"ATENCAO: {len(open_meshes)} de {len(meshes)} malhas nao sao watertight.",
        )
        ok = messagebox.askyesno(
            "Malhas abertas",
            f"{len(open_meshes)} de {len(meshes)} malhas nao sao watertight.\n\n"
            "O sinal do SDF vem de um winding number: numa malha aberta o "
            "dentro/fora sai errado e o dataset fica quebrado sem avisar.\n\n"
            "Gerar mesmo assim?",
        )
        if not ok:
            _append(st["log_data"], "Cancelado. Conserte as malhas primeiro.")
            return

    n_samples = _read_int(st["n_samples"], 50000)
    surface = bool(st["surface_samples"].get())

    def work(log):
        info = datasets.make_dataset(
            meshes,
            DATA_ROOT,
            name,
            class_name,
            n_samples=n_samples,
            add_surface_samples=surface,
            log=log,
        )
        log(f"Split: {info['split_path']}")
        st["queue"].put(("call", lambda: _refresh_datasets(st)))

    _run_worker(st, st["btn_generate"], st["log_data"], work)


def _validar(st):
    row = st["datasets"].get(st["combo_data"].get())
    if row is None:
        messagebox.showerror("Sem dataset", "Escolha um dataset na lista.")
        return

    def work(log):
        report = datasets.validate_dataset(row["path"], log=log)
        st["queue"].put(("call", lambda: _show_report(st, report)))

    _run_worker(st, st["btn_validate"], st["log_data"], work)


def _show_report(st, report):
    ax = viz.slice_axes(st["fig_hist"], 1)[0]
    viz.draw_phi_histogram(ax, report["phi_all"])
    st["canvas_hist"].draw_idle()
    problems = report["problems"]
    if problems:
        messagebox.showwarning(
            "Dataset com problemas",
            "\n".join(problems[:10])
            + ("\n..." if len(problems) > 10 else "")
            + "\n\nTreinar assim gera um decoder sem sentido.",
        )


# --------------------------------------------------------------------------- #
# aba 2 - Treinar
# --------------------------------------------------------------------------- #


def _build_tab_train(st, parent):
    st["latent_dim"] = tk.IntVar(value=2)
    st["n_layers"] = tk.IntVar(value=6)
    st["width"] = tk.IntVar(value=128)
    st["epochs"] = tk.IntVar(value=200)
    st["run_name"] = tk.StringVar(value="")

    form = ttk.LabelFrame(parent, text="Treinar decoder", padding=8)
    form.pack(fill="x")

    top = ttk.Frame(form)
    top.pack(fill="x")
    ttk.Label(top, text="Dataset:").pack(side="left")
    st["combo_train"] = ttk.Combobox(top, state="readonly", width=40)
    st["combo_train"].pack(side="left", padx=4)
    ttk.Button(top, text="Atualizar", command=lambda: _refresh_datasets(st)).pack(
        side="left"
    )

    line = ttk.Frame(form)
    line.pack(fill="x", pady=(6, 0))
    for label, var, low, high, step in (
        ("Dim. latente d:", st["latent_dim"], 1, 64, 1),
        ("Camadas:", st["n_layers"], 2, 12, 1),
        ("Largura:", st["width"], 16, 512, 16),
        ("Epocas:", st["epochs"], 1, 5000, 10),
    ):
        ttk.Label(line, text=label).pack(side="left")
        _spinbox(line, var, low, high, step).pack(side="left", padx=(4, 12))

    line2 = ttk.Frame(form)
    line2.pack(fill="x", pady=(6, 0))
    ttk.Label(line2, text="Nome da run:").pack(side="left")
    ttk.Entry(line2, textvariable=st["run_name"], width=28).pack(side="left", padx=4)
    ttk.Button(line2, text="Preset rapido", command=lambda: _preset_rapido(st)).pack(
        side="left", padx=4
    )
    st["btn_train"] = ttk.Button(line2, text="Treinar", command=lambda: _treinar(st))
    st["btn_train"].pack(side="left", padx=4)

    frame, st["log_train"] = _log_box(parent, height=22)
    frame.pack(fill="both", expand=True, pady=(8, 0))


def _preset_rapido(st):
    st["epochs"].set(30)
    _append(
        st["log_train"],
        "Preset rapido: 30 epocas, o suficiente para fechar o ciclo em ~1 min de CPU.",
    )


def _treinar(st):
    row = st["datasets"].get(st["combo_train"].get())
    if row is None:
        messagebox.showerror("Sem dataset", "Escolha um dataset na lista.")
        return
    if not row["split"]:
        messagebox.showerror(
            "Sem split", f"O dataset '{row['name']}' nao tem arquivo de split."
        )
        return

    latent_dim = _read_int(st["latent_dim"], 2)
    n_layers = _read_int(st["n_layers"], 6)
    width = _read_int(st["width"], 128)
    epochs = _read_int(st["epochs"], 200)

    name = st["run_name"].get().strip()
    if not name:
        name = f"{row['name']}_d{latent_dim}_{datetime.now():%Y%m%d_%H%M}"
        st["run_name"].set(name)
    run_dir = RUNS_DIR / name
    if (run_dir / "specs.json").is_file():
        if not messagebox.askyesno("Run existente", f"Sobrescrever '{name}'?"):
            return

    if latent_dim >= 2 and row["n_instances"] < MIN_SHAPES_2D:
        _append(
            st["log_train"],
            f"AVISO: d={latent_dim} com so {row['n_instances']} formas. Um espaco "
            f"latente 2D precisa de uma grade de formas (~{MIN_SHAPES_2D}+, o paper "
            "usou 120); com poucas, interpolar passa por buracos nao treinados.",
        )

    def work(log):
        specs = training.write_specs(
            run_dir,
            latent_dim,
            row["split"],
            DATA_ROOT,
            n_layers=n_layers,
            width=width,
            num_epochs=epochs,
        )
        log(f"specs: {specs}")
        log(f"Treinando '{name}' em CPU, {epochs} epocas...")
        with _signals_off():
            info = training.train(run_dir, DATA_ROOT, log=log)
        training.write_metadata(run_dir, dataset=row["name"], epochs=info["epochs"])
        log(f"Treino terminou em {info['seconds']:.0f} s -> {run_dir}")
        st["queue"].put(("call", lambda: _refresh_models(st)))

    _run_worker(st, st["btn_train"], st["log_train"], work)


# --------------------------------------------------------------------------- #
# aba 3 - Usar
# --------------------------------------------------------------------------- #


def _build_tab_use(st, parent):
    st["n_ctrl"] = [tk.IntVar(value=2) for _ in range(3)]
    st["tiling"] = [tk.IntVar(value=2) for _ in range(3)]
    st["n_base"] = tk.IntVar(value=8)
    st["status"] = tk.StringVar(value="Nenhum modelo carregado.")

    top = ttk.LabelFrame(parent, text="Modelo", padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="Decoder:").pack(side="left")
    st["combo_model"] = ttk.Combobox(top, state="readonly", width=46)
    st["combo_model"].pack(side="left", padx=4)
    ttk.Button(top, text="Atualizar", command=lambda: _refresh_models(st)).pack(
        side="left"
    )
    ttk.Label(top, text="   Controle (nx,ny,nz):").pack(side="left")
    for var in st["n_ctrl"]:
        _spinbox(top, var, 2, 6, width=4).pack(side="left", padx=1)
    ttk.Label(top, text="   Tiling:").pack(side="left")
    for var in st["tiling"]:
        _spinbox(top, var, 1, 8, width=4).pack(side="left", padx=1)
    st["btn_load"] = ttk.Button(
        top, text="Carregar", command=lambda: _carregar_modelo(st)
    )
    st["btn_load"].pack(side="left", padx=8)

    middle = ttk.Frame(parent)
    middle.pack(fill="both", expand=True, pady=(8, 0))

    left = ttk.LabelFrame(middle, text="Pontos de controle do latente", padding=4)
    left.pack(side="left", fill="both")
    st["slider_frame"] = _scrollable(left, width=420)

    right = ttk.Frame(middle)
    right.pack(side="right", fill="both", expand=True)
    st["fig_plane"], st["canvas_plane"] = _figure_canvas(right, (4.0, 3.4))

    bottom = ttk.Frame(parent)
    bottom.pack(fill="both", expand=True, pady=(8, 0))
    phi_side = ttk.Frame(bottom)
    phi_side.pack(side="left", fill="both", expand=True)
    st["fig_phi"], st["canvas_phi"] = _figure_canvas(phi_side, (4.4, 3.0))
    fields_side = ttk.Frame(bottom)
    fields_side.pack(side="right", fill="both", expand=True)
    st["fig_fields"], st["canvas_fields"] = _figure_canvas(fields_side, (6.6, 3.0))

    controls = ttk.Frame(parent)
    controls.pack(fill="x", pady=(4, 0))
    ttk.Label(controls, text="Altura z:").pack(side="left")
    st["z_scale"] = ttk.Scale(
        controls, from_=0.0, to=1.0, value=0.5, command=lambda v: _schedule_redraw(st)
    )
    st["z_scale"].pack(side="left", fill="x", expand=True, padx=6)
    ttk.Label(controls, text="N_base:").pack(side="left")
    _spinbox(controls, st["n_base"], 4, 40, width=5).pack(side="left", padx=4)
    st["btn_mesh"] = ttk.Button(
        controls, text="Malha 3D", command=lambda: _mostrar_malha(st)
    )
    st["btn_mesh"].pack(side="left", padx=4)

    ttk.Label(parent, textvariable=st["status"]).pack(fill="x", pady=(4, 0))
    frame, st["log_use"] = _log_box(parent, height=6)
    frame.pack(fill="x")


def _carregar_modelo(st):
    entry = st["models"].get(st["combo_model"].get())
    if entry is None:
        messagebox.showerror("Sem modelo", "Escolha um decoder na lista.")
        return
    n_ctrl = tuple(_read_int(v, 2) for v in st["n_ctrl"])
    tiling = tuple(_read_int(v, 2) for v in st["tiling"])

    def work(log):
        log(f"Carregando {entry.name}...")
        model = models.load_model(entry)
        trained = models.trained_latents(model)
        control_points = models.default_control_points(model, n_ctrl)
        sdf, param, lattice = models.build_lattice(
            model, tiling, n_ctrl, control_points
        )
        bounds = np.asarray(lattice._get_domain_bounds().detach().cpu(), dtype=float)
        # first evaluation builds the lazy kernels; do it here, not on the Tk loop
        models.eval_sdf_slice(sdf, float(bounds[:, 2].mean()), res=SLICE_RES)
        log(
            f"d={trained.shape[1]}, {len(trained)} latentes treinados, "
            f"{len(control_points)} pontos de controle, tiling {list(tiling)}"
        )
        st["queue"].put(
            (
                "call",
                lambda: _instalar_modelo(
                    st, entry, trained, control_points, sdf, param, bounds
                ),
            )
        )

    _run_worker(st, st["btn_load"], st["log_use"], work)


def _instalar_modelo(st, entry, trained, control_points, sdf, param, bounds):
    st["entry"] = entry
    st["trained"] = trained
    st["cps"] = np.asarray(control_points, dtype=float)
    st["sdf"] = sdf
    st["param"] = param
    st["bounds"] = bounds
    st["ax_phi"] = None

    lo, hi = trained.min(axis=0), trained.max(axis=0)
    margin = 0.1 * np.maximum(hi - lo, 1e-3)
    st["trained_range"] = (lo, hi)
    st["slider_range"] = (lo - margin, hi + margin)

    n_sliders = st["cps"].size
    if n_sliders > MANY_SLIDERS:
        _append(
            st["log_use"],
            f"{n_sliders} sliders ({st['cps'].shape[0]} pontos x d="
            f"{st['cps'].shape[1]}); a lista fica pesada.",
        )
    d = st["cps"].shape[1]
    if d > MAX_LATENT_PANELS:
        _append(
            st["log_use"],
            f"d={d}: mostrando so as {MAX_LATENT_PANELS} primeiras componentes de "
            "lambda nos paineis.",
        )

    _build_sliders(st)
    st["z_scale"].configure(from_=float(bounds[0, 2]), to=float(bounds[1, 2]))
    st["z_scale"].set(float(bounds[:, 2].mean()))
    _redraw(st)
    _draw_context(st)


def _slider_row(parent, text, value, lo, hi, command):
    row = ttk.Frame(parent)
    row.pack(fill="x", padx=2, pady=1)
    ttk.Label(row, text=text, width=14).pack(side="left")
    label = ttk.Label(row, text=f"{value:+.3f}", width=7)
    label.pack(side="right")
    scale = ttk.Scale(
        row, from_=lo, to=hi, value=value, command=lambda v: command(v, label)
    )
    scale.pack(side="left", fill="x", expand=True, padx=4)
    return scale, label


def _build_sliders(st):
    frame = st["slider_frame"]
    for child in frame.winfo_children():
        child.destroy()

    cps = st["cps"]
    lo, hi = st["slider_range"]
    st["cp_widgets"] = {}

    # A constant latent field is the uniform lattice: every cell identical. That
    # is the starting point, so it gets its own controls instead of making the
    # user drag all n_cp sliders of a component to the same spot.
    glob = ttk.LabelFrame(frame, text="Lambda global (peca inteira)", padding=4)
    glob.pack(fill="x", padx=2, pady=(2, 8))
    st["global_widgets"] = {}
    for j in range(cps.shape[1]):
        st["global_widgets"][j] = _slider_row(
            glob,
            f"lambda_{j + 1}",
            float(cps[:, j].mean()),
            float(lo[j]),
            float(hi[j]),
            lambda v, lb, j=j: _on_global_slider(st, j, v, lb),
        )

    per = ttk.LabelFrame(frame, text="Por canto (grada o lattice)", padding=4)
    per.pack(fill="x", padx=2, pady=2)
    for i in range(cps.shape[0]):
        for j in range(cps.shape[1]):
            st["cp_widgets"][(i, j)] = _slider_row(
                per,
                f"cp{i} lambda_{j + 1}",
                float(cps[i, j]),
                float(lo[j]),
                float(hi[j]),
                lambda v, lb, i=i, j=j: _on_slider(st, i, j, v, lb),
            )


def _on_slider(st, i, j, raw, label):
    value = float(raw)
    st["cps"][i, j] = value
    label.configure(text=f"{value:+.3f}")
    _schedule_redraw(st)


def _on_global_slider(st, j, raw, label):
    value = float(raw)
    st["cps"][:, j] = value
    label.configure(text=f"{value:+.3f}")
    for (i, jj), (scale, lb) in st["cp_widgets"].items():
        if jj == j:
            scale.set(value)
            lb.configure(text=f"{value:+.3f}")
    _schedule_redraw(st)


def _reschedule(st, key, delay, fn):
    if st.get(key) is not None:
        st["root"].after_cancel(st[key])
    st[key] = st["root"].after(delay, fn)


def _schedule_redraw(st):
    """Coalesce slider events: one geometry redraw per DEBOUNCE_MS.

    Rebuilding a matplotlib panel costs an order of magnitude more than
    evaluating the SDF, so only f_theta - the geometry the slider is supposed to
    move - is redrawn while dragging. lambda(x) and the latent plane are context
    and wait for the drag to stop.
    """
    _reschedule(st, "job_phi", DEBOUNCE_MS, lambda: _redraw(st))
    _reschedule(st, "job_context", SETTLE_MS, lambda: _draw_context(st))


def _redraw(st):
    st["job_phi"] = None
    if "sdf" not in st:
        return

    models.set_control_points(st["param"], st["cps"])
    z = float(st["z_scale"].get())
    phi = models.eval_sdf_slice(st["sdf"], z, res=SLICE_RES, bounds=st["bounds"])

    # keep the axes across frames so draw_sdf_slice can update the image in
    # place; anything that clears the figure detaches it, so check, do not trust
    if st.get("ax_phi") not in st["fig_phi"].axes:
        st["ax_phi"] = viz.slice_axes(st["fig_phi"], 1)[0]
    viz.draw_sdf_slice(st["ax_phi"], phi, st["bounds"], title=f"f_theta  (z = {z:.2f})")
    st["canvas_phi"].draw_idle()

    _update_status(st)


def _draw_context(st):
    st["job_context"] = None
    if "param" not in st:
        return
    z = float(st["z_scale"].get())
    latent = models.eval_latent_slice(
        st["param"], z, res=SLICE_RES, bounds=st["bounds"]
    )

    n_panels = min(latent.shape[-1], MAX_LATENT_PANELS)
    axes = viz.slice_axes(st["fig_fields"], n_panels)
    viz.draw_latent_slices(axes, latent, st["bounds"])
    st["canvas_fields"].draw_idle()

    ax = viz.slice_axes(st["fig_plane"], 1)[0]
    viz.draw_latent_plane(ax, st["trained"], st["cps"])
    st["canvas_plane"].draw_idle()


def _update_status(st):
    lo, hi = st["trained_range"]
    outside = int(np.sum((st["cps"] < lo) | (st["cps"] > hi)))
    text = (
        f"{st['entry'].name}: {st['cps'].shape[0]} pontos de controle, "
        f"d={st['cps'].shape[1]}"
    )
    if outside:
        text += (
            f"   |   {outside} valor(es) fora da faixa treinada: o decoder nao "
            "aprendeu nada la, a geometria pode virar lixo."
        )
    st["status"].set(text)


def _mostrar_malha(st):
    if "sdf" not in st:
        messagebox.showerror("Sem modelo", "Carregue um decoder primeiro.")
        return
    models.set_control_points(st["param"], st["cps"])
    sdf = st["sdf"]
    n_base = _read_int(st["n_base"], 8)

    def work(log):
        log(f"Extraindo a superficie com N_base={n_base}...")
        mesh = models.surface_mesh(sdf, n_base)
        log(f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        st["queue"].put(("call", lambda: _abrir_janela_3d(st, mesh)))

    _run_worker(st, st["btn_mesh"], st["log_use"], work)


def _abrir_janela_3d(st, mesh):
    if mesh is None or len(mesh.faces) == 0:
        _append(
            st["log_use"],
            "A superficie saiu vazia: f_theta nao cruza zero neste dominio. "
            "Ou o decoder ainda esta cru (poucas epocas), ou os latentes atuais "
            "estao fora da faixa treinada. Veja o corte de f_theta: se ele for "
            "todo da mesma cor, nao ha geometria para extrair.",
        )
        return
    _append(st["log_use"], "Abrindo a janela 3D; ela trava a interface ate fechar.")
    # flush the warning before VTK takes over the event loop
    st["root"].update_idletasks()
    if not viz.show_mesh(mesh, title="lattice"):
        _append(st["log_use"], "Nao deu para abrir a janela 3D (VTK indisponivel).")


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #


def build_app():
    """Build the Tk root with the three tabs wired up, without starting the loop."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.title("Oficina DeepSDF - dataset, treino e f_theta(lambda(x), x)")
    root.geometry("1360x900")

    st = {"root": root, "queue": queue.Queue(), "busy": False}
    root.app_state = st

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)
    for title, builder in (
        ("SDF maker", _build_tab_data),
        ("Treinar", _build_tab_train),
        ("Usar", _build_tab_use),
    ):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text=title)
        builder(st, tab)

    _refresh_datasets(st)
    _refresh_models(st)
    _append(st["log_data"], f"Datasets em {DATA_ROOT}")
    _append(st["log_train"], f"Runs em {RUNS_DIR}")
    _append(st["log_use"], "Escolha um decoder e clique em Carregar.")

    root.after(POLL_MS, lambda: _poll(st))
    return root


def main():
    build_app().mainloop()


if __name__ == "__main__":
    main()
