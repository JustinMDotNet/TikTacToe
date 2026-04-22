# 6×6 Tic-Tac-Toe with Random Pre-allocation

AI801 — Group 7
Daniel Cortes Correales · Puyun Guo · Justin Moran

A study of alpha-beta search with a tunable heuristic on a 6×6 board where
**k = 4** in a row wins. Each game starts with a small number of randomly
pre-placed markers so the opening varies between runs, which is what we
sweep over in the experiments.

## Repo layout

```
.
├── settings.py           # all knobs: board size, depth, weights, seeds, N games
├── tictactoe66.py        # game class, heuristic, players, multiprocessing worker
├── aima/                 # vendored AIMA modules (games4e, search4e, agents4e, utils4e)
├── cache/
│   ├── results/          # pickled experiment data (committed; reviewers can skip NB3)
│   ├── figures/          # final report figures (committed)
│   └── games/            # game replays (regenerated on demand, not committed)
└── notebooks/
    ├── 01_game_design.ipynb     # board, rules, random pre-allocation, human player
    ├── 02_ai_agents.ipynb       # baselines: random, alpha-beta with cutoff
    ├── 03_optimization.ipynb    # parallel sweeps: depth, pre-allocation, weights
    └── 04_final_analysis.ipynb  # head-to-head + summary plots
```

## Setup

Python 3.13+ recommended.

```powershell
# with uv (preferred — uses the pinned uv.lock)
uv sync
uv run jupyter lab notebooks

# or with pip
pip install jupyter numpy pandas matplotlib pyarrow>=24
jupyter lab notebooks
```

## How to read the notebooks

The notebooks are committed **with their cell outputs**, so you can read
through them top-to-bottom without running anything. The pickled experiment
results in `cache/results/` are also committed, so notebook 4 (final
analysis) loads in seconds.

If you want to re-run from scratch:

```powershell
Remove-Item cache\results\*.pkl, cache\figures\*.png
jupyter nbconvert --to notebook --execute --inplace notebooks\03_optimization.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks\04_final_analysis.ipynb
```

Each NB3 experiment is wrapped in `run_or_load(name, fn)` which loads from
`cache/results/<name>.pkl` if present. To force a full recompute set
`FORCE_RECOMPUTE = True` in `settings.py` (one switch, applies to all three
experiments).

Notebook 3 is the heavy one — it runs hundreds of games in a process pool
across all CPU cores. On a 16-core / 32-thread machine it takes roughly
10–15 minutes; budget more on smaller machines. Notebooks 1, 2 and 4 each
finish in well under a minute.

## Configuration

Every experimental knob lives in `settings.py`:

- `BOARD_SIZE = 6`, `WIN_LENGTH = 4`
- `DEFAULT_DEPTH = 3` (alpha-beta cutoff used in head-to-head)
- `TUNING_DEPTH  = 3` (depth used during weight tuning — kept equal so the
  weights we pick actually transfer to the deployment depth)
- `N_GAMES_PER_CONFIG = 80` (per cell of every sweep, side-balanced)
- `FORCE_RECOMPUTE = False` (set `True` to make notebook 3 ignore the cached
  pkls and rerun every sweep from scratch)
- `DEFAULT_HEURISTIC_WEIGHTS` — the weights `w_two`, `w_three`,
  `w_block_three`, `w_center`, `w_win`, plus `w_open_three` /
  `w_block_open_three` (currently 0; see notebook 3 for why)

Change a value in `settings.py`, restart the kernel, and every notebook
picks it up.

## Parallelism

Notebooks 3 and 4 use a single persistent `ProcessPoolExecutor` (created in
the first cell, shut down via `atexit`). The pool size defaults to
`os.cpu_count()`, so it scales with whatever machine you run on — no fixed
thread count to edit. Workers re-import the game on first use; subsequent
jobs are essentially free.

The order in which finished games come back is non-deterministic, but each
game has a deterministic per-game seed (`base_seed * 1_000_003 + i`), so
the results are reproducible regardless of completion order.

## AIMA dependency

`aima/` contains a pinned copy of the relevant AIMA-Python modules
(`games4e`, `search4e`, `agents4e`, `utils4e`). They are added to
`sys.path` automatically by `settings.ensure_aima_on_path()`, which
`tictactoe66.py` calls at import time. No separate install is needed.
