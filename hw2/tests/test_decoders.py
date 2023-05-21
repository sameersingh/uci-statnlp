import sys; sys.path.append("../code")
import math, random
import numpy as np


from decoders import (
    top_k_sampling,
    nucleus_sampling,
    constrained_decoding,
    constrained_decoding_no_repetition,
)

# Let us define a class to
class ModelTest:
    """Model used for testing.
    Contains next ID probabilities over 7 decoding steps over a vocab of 4 IDs
    This model is conditionally independent, meaning that no matter what
    the previously decoded ID was, the following probabilities is fixed.
    We use a ID of 0 as the end-of-sentence ID.
    """
    EOS_TOKEN_ID = 0

    def __init__(self):
        self._model = np.array([
            [0.1, 0.2, 0.3, 0.4], # timestep 0
            [0.2, 0.3, 0.4, 0.1], # timestep 1
            [0.1, 0.3, 0.4, 0.2], # timestep 2
            [0.4, 0.2, 0.3, 0.1], # timestep 3
            [0.1, 0.4, 0.2, 0.3], # timestep 4
            [0.1, 0.4, 0.2, 0.3], # timestep 5
            [0.1, 0.2, 0.3, 0.4], # timestep 6
        ])
        self.is_ngram = False

    def cond_logprob_dist(self, context: list):
        time_step = len(context)
        return np.log(np.array(self._model[time_step,:]))

    def word2id(self, a):
        return a

def test_temperature_top_k():
    print('\nTesting Temperature Top k...\n-----------------')

    # set seed for deterministic running/testing
    random.seed(42, version=1)

    # Call top_k sampling
    candidate = top_k_sampling(
        model= ModelTest(),
        # Get the top 3 k's at each time step
        top_k=3,
        # Temperature scaling of 0.05 (basically greedy decoding)
        temperature=0.05,
        # Only decode up to 6 IDs
        max_length=6,
    )

    # Check the generated candidate against gold candidate
    gold_candidate = {'decoded_ids': [3, 2, 2, 0], 'log_prob': -3.66516292749662}

    print(f"Your candidate. Decoded IDs: {candidate.decoded_ids} Score: {candidate.log_prob}")
    print(f"Gold candidate. Decoded IDs: {gold_candidate['decoded_ids']} Score: {gold_candidate['log_prob']}")
    assert candidate.decoded_ids == gold_candidate['decoded_ids']
    assert math.isclose(candidate.log_prob, gold_candidate['log_prob'], abs_tol=1e-3)


def test_nucleus_sampling():
    print('\nTesting Nucleus Sampling...\n-----------------')

    # set seed for deterministic running/testing
    random.seed(2)

    # Call beam search to get top `beam_size` candidates
    candidate = nucleus_sampling(
        model= ModelTest(),
        # Filter for the smallest # of IDs where the accumulated prob is >= 0.7
        top_p=0.7,
        # Only decode up to 6 IDs
        max_length=6,
    )

    # Check the generated candidate against gold candidate
    gold_candidate = {'decoded_ids': [2, 1, 2, 0], 'log_prob': -4.240527072400182}

    print(f"Your candidate. Decoded IDs: {candidate.decoded_ids} Score: {candidate.log_prob}")
    print(f"Gold candidate. Decoded IDs: {gold_candidate['decoded_ids']} Score: {gold_candidate['log_prob']}")
    assert candidate.decoded_ids == gold_candidate['decoded_ids']
    assert math.isclose(candidate.log_prob, gold_candidate['log_prob'], abs_tol=1e-3)


def test_constrained_decoding():
    print('\nTesting Constrained Decoder...\n-----------------')

    random.seed(2)
    # Call beam search to get top `beam_size` candidates
    candidate = constrained_decoding(
        model=ModelTest(),
        constraints_list=[0, 3],
        max_length=6,
    )

    # Check the generated candidates against gold candidates
    gold_candidate = {'decoded_ids': [1, 1, 2, 2, 2, 2], 'log_prob': -8.152550077828328}

    print(f"Your candidate. Decoded IDs: {candidate.decoded_ids} Score: {candidate.log_prob}")
    print(f"Gold candidate. Decoded IDs: {gold_candidate['decoded_ids']} Score: {gold_candidate['log_prob']}")
    assert candidate.decoded_ids == gold_candidate['decoded_ids']
    assert math.isclose(candidate.log_prob, gold_candidate['log_prob'], abs_tol=1e-3)


def test_constrained_decoding_no_repetition():

    print('\nTesting Constrained Decoder with no repetition...\n-----------------')

    random.seed(42)
    # Call beam search to get top `beam_size` candidates
    candidate = constrained_decoding_no_repetition(
        model=ModelTest(),
        max_length=6,
    )

    # Check the generated candidates against gold candidates
    gold_candidate = {'decoded_ids': [2, 1, 3, 0], 'log_prob': -4.933674252960127}

    print(f"Your candidate. Decoded IDs: {candidate.decoded_ids} Score: {candidate.log_prob}")
    print(f"Gold candidate. Decoded IDs: {gold_candidate['decoded_ids']} Score: {gold_candidate['log_prob']}")
    assert candidate.decoded_ids == gold_candidate['decoded_ids']
    assert math.isclose(candidate.log_prob, gold_candidate['log_prob'], abs_tol=1e-3)


if __name__ == "__main__":
    # ----------------------------------------------------------
    # You can execute this script in one of two ways:
    #
    # 1. You use Python command: python -m test_decoders
    # The file should execute with no errors. If an assertion
    # error is detected then, you may have a bug in your
    # implementation.
    #
    # 2. You use pytest and type down in "pytest" in the terminal
    # This will tell you how many tests you failed and how many
    # you passed, as well as provide you some details on which
    # line failed and why.
    # ----------------------------------------------------------
    # Both approaches work fairly well, I'd say the advantage of
    # number 2 is that you don't have to list all the test methods
    # in the main (you are less prone to forget a test).
    # Pytest will automatically execute every method in the files
    # whose name starts with "test_" for method names starting with
    # "test_".
    # ----------------------------------------------------------
    test_temperature_top_k()
    test_nucleus_sampling()
    test_constrained_decoding()
    test_constrained_decoding_no_repetition()