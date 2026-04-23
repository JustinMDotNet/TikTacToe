# AI801 Project -- Group 7

**6×6 Tic-Tac-Toe with Random Pre-allocation**

Members: Daniel Cortes Correales, Puyun Guo, Justin Moran.

## Repo layout

```
.
├── settings.py           # shared configuration (paths, board size, weights, seeds)
├── tictactoe66.py        # game class + heuristic + players (built on AIMA games4e)
├── experiments.py        # shared experiment helpers (pool, run_or_load, wilson_ci, ...)
├── viz.py                # plotting helpers used by the notebooks
├── aima/                 # vendored AIMA modules (games4e, search4e, agents4e, utils4e)
├── cache/                # auto-populated by notebooks (results, figures, games)
└── notebooks/
    ├── 01_game_design.ipynb       # Task 1 / Milestone 1
    ├── 02_ai_agents.ipynb         # Task 2 / Milestone 2
    ├── 03_optimization.ipynb      # Task 3 / Milestone 3
    ├── 04_tuning.ipynb            # Task 3 follow-up: deeper/wider tuning
    └── 05_final_analysis.ipynb    # Task 4 / Milestone 4
```

## Quick start

This repo ships a `pyproject.toml` + `uv.lock`. The one-liner is:

```powershell
uv sync
uv run jupyter lab notebooks
```

Or with pip:

```powershell
pip install jupyter numpy pandas matplotlib scipy pyarrow ipykernel
pip install numba   # optional, speeds up notebook 03/04 sweeps 5-10x
jupyter lab notebooks
```

Run notebooks in order; later notebooks read pickled artifacts from `cache/`
that earlier notebooks create. The `cache/results/` and `cache/figures/`
directories are committed so teammates can open every notebook and see the
original figures without re-running a single experiment. Delete `cache/`
(or set `FORCE_RECOMPUTE = True` in `settings.py`) to recompute from
scratch.

## Configuration

All knobs (board size, win length, depth sweep, heuristic weights, seed,
games-per-config) live in `settings.py`. Notebooks import from there so a
single edit propagates everywhere.

## Performance note: numba

`tictactoe66.Heuristic` dispatches its inner scoring loop through a
numba-jitted kernel. If `numba` is not importable the module falls back
to a pure-Python path that produces identical results but runs roughly
**5–10× slower**, which makes the notebook 03 sweeps (especially expC)
significantly slower to recompute from scratch. Cached pickles in
`cache/results/` are unaffected. Install numba (`pip install numba`)
before deleting `cache/` if you intend to rerun the experiments.

## AIMA dependency

`aima/` contains a pinned copy of the relevant AIMA-Python modules. They are
added to `sys.path` automatically by `settings.ensure_aima_on_path()`,
which `tictactoe66.py` calls at import time.
