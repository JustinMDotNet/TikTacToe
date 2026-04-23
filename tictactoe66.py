"""6x6 Tic-Tac-Toe with random pre-allocation, built on AIMA games4e.

This module subclasses :class:`games4e.TicTacToe` so we inherit the
``actions`` / ``result`` / ``terminal_test`` / ``utility`` / ``k_in_row``
machinery and only override what changes for the modified variant:

* board size 6x6, win length k=4 (handled by passing h=v=6, k=4 to base ctor)
* a stochastic initial state: each player gets ``n_prealloc`` markers placed
  at random, uniformly, in distinct empty cells before X moves first.
* a richer heuristic suitable for cutoff-based alpha-beta search.

The module is intentionally self-contained: notebooks just do
``from ttt66 import TicTacToe66, make_initial_state, heuristic_eval`` etc.
"""

from __future__ import annotations

import random
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Iterable

import settings
settings.ensure_aima_on_path()

from games4e import TicTacToe, GameState, alpha_beta_cutoff_search  # noqa: E402

# ---------------------------------------------------------------------------
# Numba kernel for the inner heuristic loop.
# Profiling shows the heuristic is called millions of times during Exp C, and
# almost all of its runtime is in two pure-Python loops over the line list.
# A jitted kernel that operates on a flat int8 board removes the dict lookups
# and lets the compiler vectorize the inner counting, which buys ~5-10x in
# our measurements.  We fall back to the pure-Python path if numba is not
# available so the module still imports cleanly in a stripped environment.
# ---------------------------------------------------------------------------
import numpy as _np

try:
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover -- exercised only when numba is missing
    _HAVE_NUMBA = False
    def _njit(*a, **kw):
        def deco(fn):
            return fn
        return deco


@_njit(cache=True)
def _score_board_kernel(board_flat, lines_idx, open5_idx, center_idx,
                        me_sign,
                        w_two, w_three, w_block_two, w_block_three,
                        w_open, w_block_open, w_center):
    """Numba-jitted scoring loop.  See ``Heuristic.__call__`` for semantics.

    All weights and ``me_sign`` are scalars; the index arrays are precomputed
    once per game instance.  ``open5_idx`` may be empty (shape (0, 5)) when
    the open-three weights are both zero -- the loop below handles that.
    """
    score = 0.0
    n_lines = lines_idx.shape[0]
    line_len = lines_idx.shape[1]
    for li in range(n_lines):
        mine = 0
        theirs = 0
        for j in range(line_len):
            v = board_flat[lines_idx[li, j]]
            if v == me_sign:
                mine += 1
            elif v == -me_sign and v != 0:
                theirs += 1
        if theirs == 0 and mine > 0:
            if mine == 2:
                score += w_two
            elif mine == 3:
                score += w_three
        elif mine == 0 and theirs > 0:
            if theirs == 2:
                score -= w_block_two
            elif theirs == 3:
                score -= w_block_three

    n_open = open5_idx.shape[0]
    if n_open > 0 and (w_open != 0.0 or w_block_open != 0.0):
        ow = open5_idx.shape[1]
        for wi in range(n_open):
            left = board_flat[open5_idx[wi, 0]]
            right = board_flat[open5_idx[wi, ow - 1]]
            if left != 0 or right != 0:
                continue
            mine_inner = 0
            theirs_inner = 0
            for j in range(1, ow - 1):
                v = board_flat[open5_idx[wi, j]]
                if v == me_sign:
                    mine_inner += 1
                elif v == -me_sign and v != 0:
                    theirs_inner += 1
            inner_len = ow - 2
            if mine_inner == inner_len:
                score += w_open
            elif theirs_inner == inner_len:
                score -= w_block_open

    for ci in range(center_idx.shape[0]):
        v = board_flat[center_idx[ci]]
        if v == me_sign:
            score += w_center
        elif v == -me_sign and v != 0:
            score -= w_center

    return score


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------
class TicTacToe66(TicTacToe):
    """6x6 Tic-Tac-Toe, win condition = 4 in a row.

    Parameters
    ----------
    h, v : int
        Board height and width. Defaults to ``settings.BOARD_SIZE``.
    k : int
        Number in a row required to win. Defaults to ``settings.WIN_LENGTH``.
    """

    def __init__(self,
                 h: int = settings.BOARD_SIZE,
                 v: int = settings.BOARD_SIZE,
                 k: int = settings.WIN_LENGTH):
        super().__init__(h=h, v=v, k=k)
        self._idx_cache = None

    # -- numba index arrays (computed once per game instance) --------------
    def _build_idx_arrays(self):
        """Return ``(lines_idx, open5_idx, center_idx)`` as numpy int32
        arrays of flat board indices, suitable for the numba kernel."""
        if self._idx_cache is not None:
            return self._idx_cache
        v = self.v
        def flat(rc):
            r, c = rc
            return (r - 1) * v + (c - 1)
        lines_idx = _np.array(
            [[flat(c) for c in line] for line in self.lines()],
            dtype=_np.int32)
        opens = self.open_three_windows()
        if opens:
            open5_idx = _np.array(
                [[flat(c) for c in win] for win in opens], dtype=_np.int32)
        else:
            open5_idx = _np.zeros((0, self.k + 1), dtype=_np.int32)
        cx_lo = self.h // 2
        cy_lo = self.v // 2
        center_cells = [(cx_lo, cy_lo), (cx_lo, cy_lo + 1),
                        (cx_lo + 1, cy_lo), (cx_lo + 1, cy_lo + 1)]
        center_idx = _np.array([flat(c) for c in center_cells],
                               dtype=_np.int32)
        self._idx_cache = (lines_idx, open5_idx, center_idx)
        return self._idx_cache

    # -- pretty display ----------------------------------------------------
    def display(self, state: GameState) -> None:
        board = state.board
        col_hdr = "    " + " ".join(f"{y:>2}" for y in range(1, self.v + 1))
        print(col_hdr)
        print("   +" + "---" * self.v + "+")
        for x in range(1, self.h + 1):
            row = " ".join(f"{board.get((x, y), '.'):>2}"
                           for y in range(1, self.v + 1))
            print(f"{x:>2} | {row} |")
        print("   +" + "---" * self.v + "+")

    # -- helper used by the heuristic --------------------------------------
    def lines(self) -> list[list[tuple[int, int]]]:
        """Return all length-k windows on the board (rows, cols, diagonals).

        Cached per-game-instance because it depends only on (h, v, k).
        """
        if getattr(self, "_lines_cache", None) is not None:
            return self._lines_cache
        h, v, k = self.h, self.v, self.k
        lines: list[list[tuple[int, int]]] = []
        # rows
        for x in range(1, h + 1):
            for y in range(1, v - k + 2):
                lines.append([(x, y + i) for i in range(k)])
        # cols
        for y in range(1, v + 1):
            for x in range(1, h - k + 2):
                lines.append([(x + i, y) for i in range(k)])
        # diag down-right
        for x in range(1, h - k + 2):
            for y in range(1, v - k + 2):
                lines.append([(x + i, y + i) for i in range(k)])
        # diag up-right
        for x in range(k, h + 1):
            for y in range(1, v - k + 2):
                lines.append([(x - i, y + i) for i in range(k)])
        self._lines_cache = lines
        return lines

    # -- open-3 detection --------------------------------------------------
    def open_three_windows(self) -> list[list[tuple[int, int]]]:
        """Return all length-(k+1)=5 windows in which positions 1..k-1 form
        a contiguous run of (k-1)=3 cells flanked by positions 0 and k.

        At evaluation time, such a window is an "open k-1" for player P iff
        the inner three cells are all P and *both* flanking cells are empty
        (i.e. ``_ X X X _``).  An open three has two distinct winning
        completions, so the opponent cannot block it with a single move --
        the standard distinction between a "live" and "dead" three in
        connect-style games.

        Cached because it depends only on (h, v, k).
        """
        if getattr(self, "_open3_cache", None) is not None:
            return self._open3_cache
        h, v, k = self.h, self.v, self.k
        L = k + 1  # length-5 for k=4
        wins: list[list[tuple[int, int]]] = []
        # rows
        for x in range(1, h + 1):
            for y in range(1, v - L + 2):
                wins.append([(x, y + i) for i in range(L)])
        # cols
        for y in range(1, v + 1):
            for x in range(1, h - L + 2):
                wins.append([(x + i, y) for i in range(L)])
        # diag down-right
        for x in range(1, h - L + 2):
            for y in range(1, v - L + 2):
                wins.append([(x + i, y + i) for i in range(L)])
        # diag up-right
        for x in range(L, h + 1):
            for y in range(1, v - L + 2):
                wins.append([(x - i, y + i) for i in range(L)])
        self._open3_cache = wins
        return wins

    # -- move ordering -----------------------------------------------------
    def actions(self, state: GameState) -> list[tuple[int, int]]:
        """Order legal moves by Chebyshev distance to the nearest occupied
        cell.  Concentrating the search on cells adjacent to existing play
        prunes far more branches in alpha-beta than the default arbitrary
        ordering, especially at the deeper cutoffs.  Falls back to the raw
        move list when the board is empty (no occupied cells to anchor to).
        """
        moves = state.moves
        board = state.board
        if not board:
            return list(moves)
        occupied = list(board.keys())

        def proximity(m: tuple[int, int]) -> int:
            mx, my = m
            return min(max(abs(mx - ox), abs(my - oy)) for ox, oy in occupied)

        return sorted(moves, key=proximity)


# ---------------------------------------------------------------------------
# Random initial state
# ---------------------------------------------------------------------------
def make_initial_state(game: TicTacToe66,
                       n_prealloc: int,
                       rng: random.Random | None = None) -> GameState:
    """Return a fresh ``GameState`` with ``n_prealloc`` X's and O's placed
    at random in distinct cells. X is to move next.

    Parameters
    ----------
    game : TicTacToe66
        The game instance whose dimensions we use.
    n_prealloc : int
        Number of pre-placed markers per player (1, 2, or 3 per spec).
    rng : random.Random, optional
        Source of randomness. A fresh ``random.Random()`` is used if omitted
        (use a seeded RNG for reproducible experiments).
    """
    if rng is None:
        rng = random.Random()
    if n_prealloc < 0 or 2 * n_prealloc > game.h * game.v:
        raise ValueError(f"invalid n_prealloc={n_prealloc} for {game.h}x{game.v} board")

    all_cells = [(x, y) for x in range(1, game.h + 1)
                        for y in range(1, game.v + 1)]
    chosen = rng.sample(all_cells, 2 * n_prealloc)
    board: dict[tuple[int, int], str] = {}
    for cell in chosen[:n_prealloc]:
        board[cell] = 'X'
    for cell in chosen[n_prealloc:]:
        board[cell] = 'O'

    moves = [c for c in all_cells if c not in board]

    # Compute utility from the pre-placed board (in case k-in-a-row already
    # exists by chance).  We re-use TicTacToe.compute_utility on the last
    # placed marker for each player, but a safer approach is to scan all
    # lines.  A pre-allocation that already satisfies the win condition is
    # extremely unlikely for n_prealloc <= 3 but we still guard it.
    utility = _scan_utility(game, board)

    return GameState(to_move='X', utility=utility, board=board, moves=moves)


def _scan_utility(game: TicTacToe66, board: dict) -> int:
    """Return +1 if X already has k-in-a-row, -1 for O, else 0."""
    for line in game.lines():
        vals = [board.get(c) for c in line]
        if all(v == 'X' for v in vals):
            return +1
        if all(v == 'O' for v in vals):
            return -1
    return 0


# ---------------------------------------------------------------------------
# Heuristic evaluation
# ---------------------------------------------------------------------------
@dataclass
class Heuristic:
    """Configurable evaluation function for use with alpha-beta cutoff search.

    The score is computed from MAX (= 'X')'s perspective and then negated for
    'O' inside ``__call__`` so it can be passed directly as ``eval_fn``.
    """

    weights: dict = None
    game: TicTacToe66 = None
    perspective: str = 'X'  # which player is MAX for this evaluation

    def __post_init__(self):
        if self.weights is None:
            self.weights = dict(settings.DEFAULT_HEURISTIC_WEIGHTS)
        if self.game is None:
            raise ValueError("Heuristic needs a game instance")

    # ------------------------------------------------------------------
    def __call__(self, state: GameState) -> float:
        """Score ``state`` from ``self.perspective``'s point of view.

        Returns a positive value when the position favors the player named
        by ``self.perspective`` ('X' or 'O') and a negative value otherwise.
        ``alpha_beta_cutoff_search`` expects evaluations from the side to
        move's perspective, so callers using ``make_alpha_beta_player``
        construct one ``Heuristic`` per move with ``perspective=state.to_move``
        (the wrapping is handled by ``make_alpha_beta_player`` itself).
        """
        # Terminal short-circuit: use the cached utility * a big weight.
        if state.utility != 0:
            sign = +1 if self.perspective == 'X' else -1
            return sign * state.utility * self.weights["w_win"]

        # Convert the dict-board to a flat int8 array (1 = X, -1 = O, 0 = empty)
        # and dispatch to the jitted kernel.  The conversion cost (36 dict
        # lookups) is negligible next to the savings inside the kernel.
        v = self.game.v
        board_flat = _np.zeros(self.game.h * self.game.v, dtype=_np.int8)
        for (r, c), marker in state.board.items():
            board_flat[(r - 1) * v + (c - 1)] = 1 if marker == 'X' else -1

        lines_idx, open5_idx, center_idx = self.game._build_idx_arrays()
        me_sign = 1 if self.perspective == 'X' else -1

        w = self.weights
        return _score_board_kernel(
            board_flat, lines_idx, open5_idx, center_idx, me_sign,
            float(w["w_two"]), float(w["w_three"]),
            float(w["w_block_two"]), float(w["w_block_three"]),
            float(w.get("w_open_three", 0.0)),
            float(w.get("w_block_open_three", 0.0)),
            float(w["w_center"]),
        )


# ---------------------------------------------------------------------------
# Players (functions taking (game, state) and returning a move)
# ---------------------------------------------------------------------------
def make_alpha_beta_player(depth: int = settings.DEFAULT_DEPTH,
                           weights: dict | None = None,
                           name: str | None = None) -> Callable:
    """Build a player function using alpha-beta cutoff search with our
    heuristic.  ``name`` is attached to the returned callable for logging."""
    def player(game: TicTacToe66, state: GameState):
        h = Heuristic(weights=weights, game=game, perspective=state.to_move)
        return alpha_beta_cutoff_search(
            state, game, d=depth,
            cutoff_test=lambda s, d: d > depth or game.terminal_test(s),
            eval_fn=h,
        )
    player.__name__ = name or f"alphabeta_d{depth}"
    return player


def make_random_player(rng: random.Random | None = None,
                       name: str = 'random_legal_player'):
    """Build a random-move player bound to a specific RNG so seeded matches
    are fully reproducible across worker processes."""
    local_rng = rng if rng is not None else random
    def player(game: TicTacToe66, state: GameState):
        return local_rng.choice(state.moves)
    player.__name__ = name
    return player


def random_legal_player(game: TicTacToe66, state: GameState):
    """Uniformly random player using the module-global RNG.  Kept for
    convenience in single-game demos; parallel runs should use
    ``make_random_player(rng)`` to stay reproducible per seed."""
    return random.choice(state.moves)


# ---------------------------------------------------------------------------
# Game-runner with random initial state, returns metadata for logging
# ---------------------------------------------------------------------------
def play_one_game(game: TicTacToe66,
                  x_player: Callable,
                  o_player: Callable,
                  n_prealloc: int,
                  rng: random.Random | None = None,
                  display: bool = False) -> dict:
    """Play one game with a random initial state.  Returns a dict with the
    outcome and number of plies, suitable for logging into a DataFrame."""
    state = make_initial_state(game, n_prealloc, rng=rng)
    plies = 0
    players = (x_player, o_player)
    if display:
        game.display(state)
    # If pre-alloc accidentally finished the game, return immediately.
    if game.terminal_test(state):
        return {"winner": _winner_str(state.utility),
                "plies": 0,
                "n_prealloc": n_prealloc}
    while True:
        for p in players:
            move = p(game, state)
            state = game.result(state, move)
            plies += 1
            if display:
                print(f"\n{p.__name__} -> {move}")
                game.display(state)
            if game.terminal_test(state):
                return {"winner": _winner_str(state.utility),
                        "plies": plies,
                        "n_prealloc": n_prealloc}


def _winner_str(utility: int) -> str:
    return {1: "X", -1: "O", 0: "draw"}.get(utility, "draw")


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
# Notebook 03 fans out games across all CPU cores via ProcessPoolExecutor.
# On Windows the executor uses spawn(), so the callable plus its arguments
# have to be importable and picklable -- which rules out the closures we get
# from make_alpha_beta_player().  The trick is to ship lightweight specs
# instead, and let each worker rebuild its players on arrival.
def _play_one_worker(args: tuple) -> dict:
    """Play one game in a worker process.

    ``args`` is ``(spec_a, spec_b, n_prealloc, seed, a_is_X, game_idx)``.
    Each spec is either the string ``"random"`` (baseline opponent) or a
    ``(depth, weights_or_None)`` pair.  Sides are decided by ``a_is_X`` so
    the caller can alternate them across games to cancel first-mover bias.
    """
    import time as _time
    spec_a, spec_b, n_prealloc, seed, a_is_X, game_idx = args
    game = TicTacToe66()

    rng = random.Random(seed)

    def _build(spec):
        if spec == "random":
            return make_random_player(rng=rng)
        depth, weights = spec
        return make_alpha_beta_player(depth=depth, weights=weights)

    player_a = _build(spec_a)
    player_b = _build(spec_b)
    x_player, o_player = (player_a, player_b) if a_is_X else (player_b, player_a)
    t0 = _time.perf_counter()
    result = play_one_game(game, x_player, o_player, n_prealloc=n_prealloc, rng=rng)
    result["seconds"] = _time.perf_counter() - t0
    result["game_idx"] = game_idx
    result["a_is_X"] = a_is_X

    # Re-label the winner in terms of A/B rather than X/O so the caller
    # doesn't have to know which side A played in this particular game.
    if result["winner"] == "draw":
        result["winner_label"] = "draw"
    elif (result["winner"] == "X") == a_is_X:
        result["winner_label"] = "A"
    else:
        result["winner_label"] = "B"
    return result


__all__ = [
    "TicTacToe66", "make_initial_state", "Heuristic",
    "make_alpha_beta_player", "random_legal_player", "make_random_player",
    "play_one_game", "_play_one_worker",
]
