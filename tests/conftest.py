import threading

import pytest

from harmonyagent.models.global_stats import GLOBAL_MODEL_STATS

# Global lock for tests that modify global state - this works across threads
_global_stats_lock = threading.Lock()


@pytest.fixture
def reset_global_stats():
    """Reset global model stats and ensure exclusive access for tests that need it.

    This fixture should be used by any test that depends on global model stats
    to ensure thread safety and test isolation.
    """
    with _global_stats_lock:
        # Reset at start
        GLOBAL_MODEL_STATS._cost = 0.0  # noqa: protected-access
        GLOBAL_MODEL_STATS._n_calls = 0  # noqa: protected-access
        yield
        # Reset at end to clean up
        GLOBAL_MODEL_STATS._cost = 0.0  # noqa: protected-access
        GLOBAL_MODEL_STATS._n_calls = 0  # noqa: protected-access
