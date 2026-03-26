# Proves RANDOM_SEED produces deterministic output for stdlib random and NumPy.
# should pass before generating any simulation data.
import random
import numpy as np
from api.app.config import RANDOM_SEED

# Enough values to prove sequence-level determinism, not just first-value luck.
SEQUENCE_LENGTH = 100

class TestStdlibRandomDeterminism:
    """Prove Python's random module is deterministic when seeded."""
    def test_same_seed_same_sequence(self):
        random.seed(RANDOM_SEED)
        first_run = [random.random() for _ in range(SEQUENCE_LENGTH)]

        random.seed(RANDOM_SEED)
        second_run = [random.random() for _ in range(SEQUENCE_LENGTH)]

        assert first_run == second_run, (
            "stdlib random produced different sequences with the same seed."
        )

    def test_different_seed_different_sequence(self):
        random.seed(RANDOM_SEED)
        first_run = [random.random() for _ in range(SEQUENCE_LENGTH)]

        random.seed(RANDOM_SEED + 1)
        different_run = [random.random() for _ in range(SEQUENCE_LENGTH)]

        assert first_run != different_run, (
            "Two different seeds produced identical sequences."
        )


class TestNumpyRandomDeterminism:
    """Prove NumPy's default_rng is deterministic when seeded."""
    def test_same_seed_same_sequence(self):
        rng_first = np.random.default_rng(RANDOM_SEED)
        first_run = [rng_first.random() for _ in range(SEQUENCE_LENGTH)]
        rng_second = np.random.default_rng(RANDOM_SEED)
        second_run = [rng_second.random() for _ in range(SEQUENCE_LENGTH)]
        assert first_run == second_run, (
            "NumPy default_rng produced different sequences with the same seed."
        )

    def test_different_seed_different_sequence(self):
        rng_a = np.random.default_rng(RANDOM_SEED)
        first_run = [rng_a.random() for _ in range(SEQUENCE_LENGTH)]

        rng_b = np.random.default_rng(RANDOM_SEED + 1)
        different_run = [rng_b.random() for _ in range(SEQUENCE_LENGTH)]

        assert first_run != different_run, (
            "Two different seeds produced identical NumPy sequences."
        )

class TestRandomSeedConfig:
    """Prove RANDOM_SEED from config.py is valid and usable."""
    def test_seed_is_integer(self):
        # Simulators do arithmetic with the seed (sub_seed = RANDOM_SEED + session_index)
        assert isinstance(RANDOM_SEED, int), (
            f"RANDOM_SEED should be int, got {type(RANDOM_SEED).__name__}: {RANDOM_SEED}"
        )

    def test_seed_is_not_none(self):
        assert RANDOM_SEED is not None, "RANDOM_SEED is None — check .env and config.py"