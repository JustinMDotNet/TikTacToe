"""Shared settings for the 6x6 Tic-Tac-Toe project (Group 7).

All notebooks import from this module to ensure consistent configuration,
paths, and reproducibility across the experimental runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
AIMA_DIR: Path = PROJECT_ROOT / "aima"
CACHE_DIR: Path = PROJECT_ROOT / "cache"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
RESULTS_DIR: Path = CACHE_DIR / "results"
GAMES_DIR: Path = CACHE_DIR / "games"
FIGURES_DIR: Path = CACHE_DIR / "figures"

for _d in (CACHE_DIR, RESULTS_DIR, GAMES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def ensure_aima_on_path() -> None:
    """Prepend the vendored AIMA directory to ``sys.path``.

    The vendored AIMA modules (``games4e``, ``search4e``, ``utils4e``,
    ``agents4e``) reference each other with bare imports, so the directory
    itself must be on ``sys.path`` (not the project root).
    """
    aima_str = str(AIMA_DIR)
    if aima_str not in sys.path:
        sys.path.insert(0, aima_str)
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ---------------------------------------------------------------------------
# Game configuration
# ---------------------------------------------------------------------------
BOARD_SIZE: int = 6           # h = v = 6
WIN_LENGTH: int = 4           # k-in-a-row to win
PREALLOC_CHOICES: tuple[int, ...] = (1, 2, 3)  # markers per player at start

# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------
DEFAULT_DEPTH: int = 3        # alpha-beta cutoff depth (per ply)
DEPTH_SWEEP: tuple[int, ...] = (1, 2, 3)
TUNING_DEPTH: int = 3         # match deployment depth so tuned weights transfer

# ---------------------------------------------------------------------------
# Heuristic weights (tuned in notebook 03)
# Keys map to "k consecutive markers in an open/half-open window of WIN_LENGTH"
# ---------------------------------------------------------------------------
DEFAULT_HEURISTIC_WEIGHTS: dict[str, float] = {
    "w_two":          1.0,    # 2 markers in a length-4 window with no opponent
    "w_three":       10.0,    # 3 markers in a length-4 window with no opponent
    "w_block_two":    1.0,    # opponent's 2-in-window we are blocking against
    "w_block_three":  8.0,    # opponent's 3-in-window threat
    # Open-three / block-open-three feature was implemented and tested
    # but did not improve play at depth 3 (the search already sees these
    # tactically). Weights left at 0 so the feature is documented but
    # disabled. See notebook 03 / project report.
    "w_open_three":       0.0,
    "w_block_open_three": 0.0,
    "w_center":   0.5,        # bonus per own marker in central 2x2
    "w_win":  10000.0,        # terminal win value used by the heuristic
}

# ---------------------------------------------------------------------------
# Reproducibility / experiment defaults
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_GAMES_PER_CONFIG: int = 80  # used in notebook 03/04 sweeps

# When True, notebook 03 ignores cache/results/*.pkl and recomputes every
# experiment from scratch. Leave False unless you've changed the heuristic,
# the search depth, or N_GAMES_PER_CONFIG.
FORCE_RECOMPUTE: bool = False

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def cache_path(name: str, subdir: str = "results") -> Path:
    """Return a path inside the cache directory, creating subdirs as needed."""
    sub = CACHE_DIR / subdir
    sub.mkdir(parents=True, exist_ok=True)
    return sub / name


__all__ = [
    "PROJECT_ROOT", "AIMA_DIR", "CACHE_DIR", "RESULTS_DIR",
    "GAMES_DIR", "FIGURES_DIR", "NOTEBOOKS_DIR",
    "ensure_aima_on_path",
    "BOARD_SIZE", "WIN_LENGTH", "PREALLOC_CHOICES",
    "DEFAULT_DEPTH", "DEPTH_SWEEP", "TUNING_DEPTH", "DEFAULT_HEURISTIC_WEIGHTS",
    "RANDOM_SEED", "N_GAMES_PER_CONFIG",
    "cache_path",
]
