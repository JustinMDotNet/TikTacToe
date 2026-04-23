"""Shared experiment helpers for the optimization notebooks.

Notebooks 03, 04, and 05 all need the same boilerplate: a process pool
warmed up against the numba JIT, a pickle-backed memoization wrapper for
expensive runs, a unified job-tuple shape that ``_play_one_worker`` can
consume, and a Wilson confidence-interval helper. Before this module
existed every notebook carried its own copy of those helpers, which had
already drifted (for example nb03 used a 6-tuple job and nb04 used a
7-tuple). Centralizing them here keeps every notebook on the same
implementation and makes it possible to fix bugs in one place.

All helpers are written to be import-safe: they make no global state,
take the pool/worker-count explicitly, and never touch ``cache/`` outside
of ``run_or_load``.
"""

from __future__ import annotations

import atexit
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from typing import Callable, Iterable

import pandas as pd

import settings
import tictactoe66


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------
def make_pool(max_workers: int | None = None) -> tuple[ProcessPoolExecutor, int]:
    """Allocate a ProcessPoolExecutor and register atexit shutdown.

    Returns ``(pool, n_workers)``. Notebooks should call this once at
    startup (Windows ``spawn`` makes per-call pools prohibitively slow).
    """
    n = max_workers or os.cpu_count() or 1
    pool = ProcessPoolExecutor(max_workers=n)
    atexit.register(pool.shutdown, wait=False, cancel_futures=True)
    return pool, n


def prewarm_pool(pool: ProcessPoolExecutor, n_workers: int) -> float:
    """Run a handful of trivial games so each worker JIT-compiles upfront.

    Returns the wall-clock seconds spent warming. Without this, the first
    real batch stalls on numba compilation in every worker simultaneously.
    """
    jobs = [((1, None), (1, None), 1, i, True, i) for i in range(n_workers * 2)]
    t0 = time.perf_counter()
    list(pool.map(tictactoe66._play_one_worker, jobs, chunksize=1))
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Cache-on-disk
# ---------------------------------------------------------------------------
def run_or_load(name: str, fn: Callable, force: bool | None = None):
    """Memoize an experiment to ``cache/results/<name>.pkl``.

    ``force`` defaults to ``settings.FORCE_RECOMPUTE`` so the recompute
    switch lives in one place. Pass ``force=True`` at the call site to
    override per-experiment.
    """
    if force is None:
        force = settings.FORCE_RECOMPUTE
    p = settings.cache_path(name + ".pkl")
    if p.exists() and not force:
        print(f"[cache] loading {p.name}")
        return pickle.loads(p.read_bytes())
    print(f"[run]   computing {p.name} ...")
    t0 = time.perf_counter()
    out = fn()
    print(f"        done in {time.perf_counter() - t0:.1f}s")
    p.write_bytes(pickle.dumps(out))
    return out


# ---------------------------------------------------------------------------
# Player specs and job construction
# ---------------------------------------------------------------------------
def make_spec(player_kind: str = "ab",
              depth: int | None = None,
              weights: dict | None = None):
    """Encode a player as something ``_play_one_worker`` can rebuild.

    ``"random"`` is the random-legal baseline; an ``("ab", depth, weights)``
    spec is encoded as the bare ``(depth, weights)`` tuple the worker
    expects. Centralizing this avoids the depth-vs-weights argument-order
    mistakes that crept into the per-notebook copies.
    """
    if player_kind == "random":
        return "random"
    if player_kind != "ab":
        raise ValueError(f"unknown player_kind: {player_kind!r}")
    return (depth, weights)


def make_balanced_jobs(spec_a, spec_b, n_prealloc, n_games, seed, tag):
    """Build ``n_games`` worker jobs with sides alternated each game.

    The tag is carried in slot 5 of the tuple; ``run_games`` swaps it for a
    unique ``game_idx`` before dispatching to the worker, then re-attaches
    the tag onto each result. Sides alternate so first-mover bias cancels
    within each cell, and the per-game seed is derived deterministically
    from ``seed`` so re-runs are reproducible regardless of worker
    completion order.
    """
    jobs = []
    for i in range(n_games):
        a_is_X = (i % 2 == 0)
        per_seed = seed * 1_000_003 + i
        jobs.append((spec_a, spec_b, n_prealloc, per_seed, a_is_X, tag))
    return jobs


def run_games(jobs, pool: ProcessPoolExecutor,
              max_workers: int | None = None) -> list[dict]:
    """Submit a flat list of jobs to ``pool`` and return the results.

    Submitting a single big batch (rather than one batch per configuration)
    keeps every core busy: short games no longer have to wait for the
    longest game in their batch before the next batch can start. The chunk
    size is tuned for Windows ``spawn`` overhead.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    tags = [j[5] for j in jobs]
    worker_jobs = [j[:5] + (i,) for i, j in enumerate(jobs)]
    chunk = max(1, min(8, len(worker_jobs) // (4 * max_workers) or 1))
    results = list(pool.map(tictactoe66._play_one_worker,
                            worker_jobs, chunksize=chunk))
    for r, tag in zip(results, tags):
        r["tag"] = tag
    return results


def games_to_df(results, label_a: str = "A", label_b: str = "B") -> pd.DataFrame:
    """Turn worker results into a DataFrame with human-friendly winner labels."""
    df = pd.DataFrame(results)
    df["winner_label"] = df["winner_label"].map(
        {"A": label_a, "B": label_b, "draw": "draw"})
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% (default) confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


# ---------------------------------------------------------------------------
# Convenience wrapper for paired head-to-head matches
# ---------------------------------------------------------------------------
def head_to_head(pool: ProcessPoolExecutor,
                 weights_new: dict, weights_old: dict,
                 n_games: int, n_prealloc: int = 2,
                 depth: int | None = None, seed: int = 999) -> pd.DataFrame:
    """Paired head-to-head: each layout is played twice with sides swapped.

    Pairing cancels first-mover advantage and layout variance within each
    pair, so any non-trivial residual signal in ``winner_label`` is
    attributable to the weight difference rather than to side bias. With
    ``weights_new == weights_old`` the design is structurally tied (every
    pair must split or both-draw), which makes this a useful null check.
    """
    if depth is None:
        depth = settings.DEFAULT_DEPTH
    s_new = (depth, weights_new)
    s_old = (depth, weights_old)
    jobs: list[tuple] = []
    n_pairs = max(1, n_games // 2)
    idx = 0
    for pair in range(n_pairs):
        per_seed = seed * 1_000_003 + pair
        for a_is_X in (True, False):
            jobs.append((s_new, s_old, n_prealloc, per_seed, a_is_X, idx))
            idx += 1
    return pd.DataFrame(pool.map(tictactoe66._play_one_worker,
                                 jobs, chunksize=1))


__all__ = [
    "make_pool", "prewarm_pool",
    "run_or_load",
    "make_spec", "make_balanced_jobs", "run_games", "games_to_df",
    "wilson_ci",
    "head_to_head",
]
