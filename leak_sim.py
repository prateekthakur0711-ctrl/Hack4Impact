"""
Backward-compatible CLI wrapper for the simulation engine.

Core simulation logic lives in `algorithms/sim.py` for reuse by `app.py` and `main_demo.py`.
"""

from algorithms.sim import *  # noqa: F403
from algorithms.sim import main


if __name__ == "__main__":
    main()

