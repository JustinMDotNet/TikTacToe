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
TUNING_DEPTH: int = 2         # tune cheap at d=2 (~10x faster than d=3);
                              # the validation cell in notebook 03 re-plays
                              # the top candidates at DEFAULT_DEPTH and falls
                              # back to defaults if none transfers, so
                              # depth-mismatch risk is bounded.

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
N_TUNING_GAMES: int = 80      # per-candidate games in Exp C. expC is a
                              # cheap *screen* at TUNING_DEPTH=2 -- the real
                              # statistical work happens in the validation
                              # cell, which re-plays the top-K at
                              # DEFAULT_DEPTH with fresh seeds. Wilson
                              # interval at N=80 is +/-0.11, fine for
                              # ranking the top few of 38 candidates.
N_RANDOM_CANDIDATES: int = 20 # extra random-search candidates added
                              # alongside the 18-cell weight grid

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
    "RANDOM_SEED", "N_GAMES_PER_CONFIG", "N_TUNING_GAMES",
    "N_RANDOM_CANDIDATES",
    "cache_path",
]
