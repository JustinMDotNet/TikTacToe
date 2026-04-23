"""Plotting helpers for the 6x6 Tic-Tac-Toe notebooks.

Centralizes the board renderer, the heuristic heatmap, and a handful of
small chart helpers so each notebook can produce report-ready figures
without copy-pasting matplotlib boilerplate.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import settings


X_COLOR = "#2E86AB"
O_COLOR = "#E63946"
GRID_COLOR = "#444444"
BG_COLOR = "#FAFAFA"


def draw_board(ax, board, title=None, h=None, v=None, highlight=None):
    """Render a tic-tac-toe board on a matplotlib axis.

    ``board`` is a dict mapping ``(row, col)`` (1-indexed, matching the
    AIMA convention) to ``'X'`` or ``'O'``.  ``highlight``, if given, is
    an iterable of cells to outline in gold (used to mark a winning line).
    """
    if h is None:
        h = settings.BOARD_SIZE
    if v is None:
        v = settings.BOARD_SIZE
    ax.set_xlim(0.5, v + 0.5)
    ax.set_ylim(0.5, h + 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_facecolor(BG_COLOR)

    for i in range(1, v + 2):
        ax.plot([i - 0.5, i - 0.5], [0.5, h + 0.5],
                color=GRID_COLOR, lw=0.8)
    for i in range(1, h + 2):
        ax.plot([0.5, v + 0.5], [i - 0.5, i - 0.5],
                color=GRID_COLOR, lw=0.8)

    for (r, c), p in board.items():
        color = X_COLOR if p == "X" else O_COLOR
        ax.text(c, r, p, ha="center", va="center",
                fontsize=20, fontweight="bold", color=color)

    if highlight:
        for (r, c) in highlight:
            ax.add_patch(mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor="gold", lw=3))

    ax.set_xticks(range(1, v + 1))
    ax.set_yticks(range(1, h + 1))
    ax.tick_params(length=0, labelsize=8)
    if title:
        ax.set_title(title, fontsize=10)


def draw_board_strip(items, titles=None, suptitle=None,
                     figsize=None, ncols=None):
    """Render a row (or grid) of boards.

    Each item may be a ``GameState`` or a raw board dict.  When ``ncols`` is
    set the boards wrap onto multiple rows, otherwise everything is laid
    out on a single horizontal strip.
    """
    n = len(items)
    if ncols is None:
        ncols = n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (3 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    titles = titles or [None] * n
    for ax, item, title in zip(axes, items, titles):
        board = item.board if hasattr(item, "board") else item
        draw_board(ax, board, title=title)
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def draw_heuristic_heatmap(ax, game, state, heuristic, cmap="RdYlGn"):
    """Color each empty cell by the heuristic score that results from
    playing there, evaluated from the side-to-move's perspective.
    """
    h, v = game.h, game.v
    grid = np.full((h, v), np.nan)
    for move in state.moves:
        nxt = game.result(state, move)
        # ``heuristic`` is built for ``state.to_move``; after the move it
        # is the opponent's turn, so we negate to keep the score from the
        # original mover's point of view.
        grid[move[0] - 1, move[1] - 1] = -heuristic(nxt)

    im = ax.imshow(grid, cmap=cmap, origin="upper",
                   extent=(0.5, v + 0.5, h + 0.5, 0.5))

    for (r, c), p in state.board.items():
        ax.text(c, r, p, ha="center", va="center",
                fontsize=16, fontweight="bold", color="white")

    for move in state.moves:
        s = grid[move[0] - 1, move[1] - 1]
        ax.text(move[1], move[0], f"{s:.0f}",
                ha="center", va="center", fontsize=7, color="black")

    ax.set_xticks(range(1, v + 1))
    ax.set_yticks(range(1, h + 1))
    ax.tick_params(length=0, labelsize=8)
    ax.set_aspect("equal")
    return im


def draw_cell_coverage(ax, game):
    """Heatmap of how many length-k windows each cell appears in."""
    h, v = game.h, game.v
    counts = np.zeros((h, v), dtype=int)
    for line in game.lines():
        for (r, c) in line:
            counts[r - 1, c - 1] += 1
    im = ax.imshow(counts, cmap="viridis", origin="upper",
                   extent=(0.5, v + 0.5, h + 0.5, 0.5))
    vmax = counts.max()
    for r in range(h):
        for c in range(v):
            ax.text(c + 1, r + 1, str(counts[r, c]),
                    ha="center", va="center", fontsize=10,
                    color="white" if counts[r, c] < vmax * 0.6 else "black")
    ax.set_xticks(range(1, v + 1))
    ax.set_yticks(range(1, h + 1))
    ax.tick_params(length=0, labelsize=8)
    ax.set_aspect("equal")
    return im


def play_one_game_recorded(game, x_player, o_player, n_prealloc, rng):
    """Play one game and record every intermediate state.

    Returns ``(initial_state, [(move, state_after, player_name), ...], result)``.
    Useful for rendering a game-progression strip without re-running search.
    """
    from tictactoe66 import _winner_str, make_initial_state

    state = make_initial_state(game, n_prealloc, rng=rng)
    initial = state
    history: list[tuple] = []
    plies = 0
    if game.terminal_test(state):
        return initial, history, {"winner": _winner_str(state.utility),
                                  "plies": 0,
                                  "n_prealloc": n_prealloc}
    players = (x_player, o_player)
    while True:
        for p in players:
            move = p(game, state)
            state = game.result(state, move)
            plies += 1
            history.append((move, state, p.__name__))
            if game.terminal_test(state):
                return initial, history, {
                    "winner": _winner_str(state.utility),
                    "plies": plies,
                    "n_prealloc": n_prealloc,
                }


def find_winning_line(game, board):
    """Return the cells of the first length-k line owned by a single player,
    or ``None`` if no such line exists."""
    for line in game.lines():
        vals = [board.get(c) for c in line]
        if vals[0] is not None and all(v == vals[0] for v in vals):
            return line
    return None


__all__ = [
    "X_COLOR", "O_COLOR",
    "draw_board", "draw_board_strip",
    "draw_heuristic_heatmap", "draw_cell_coverage",
    "play_one_game_recorded", "find_winning_line",
]
