"""
Backward-compatible CLI wrapper for the localization engine.

Core DSP logic lives in `algorithms/locator.py` for reuse by `app.py` and `main_demo.py`.
"""

from algorithms.locator import *  # noqa: F403
from algorithms.locator import main


if __name__ == "__main__":
    main()

