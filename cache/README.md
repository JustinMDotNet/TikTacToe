# Cache directory

This directory persists data across notebook runs:

- `results/` — pickled / CSV experiment results (depth sweep, weight sweep, …)
- `games/`   — recorded game trajectories
- `figures/` — saved matplotlib figures used in the final report

Anything in this directory is safe to delete; notebooks will regenerate it.
