import math
import random

from decoders import top_k_sampling, beam_search_decoding, nucleus_sampling
from models import FixedModel


def test_temperature_random_sampling(model):
    print('\nTesting Temperature Random Sampling...\n-----------------')

    # set seed for deterministic running/testing
    random.seed(2, version=1)

    # Get the top 2 k's at each time step
    top_k = 3
    # Only decode up to 6 IDs
    max_length = 6
    # If we have generated the ID 0 (end-of-sentence ID), stop generating
    eos_id = 0
    # Temperature scaling of 0.05 (basically greedy decoding)
    temperature=0.05

    # Call top_k sampling
    candidate = top_k_sampling(
        model=model,
        top_k=top_k,
        temperature=temperature,
        max_length=max_length,
        eos_id=eos_id,
    )

    # Check the generated candidate against gold candidate
    gold_candidate = {'decoded_ids': [3, 2, 2, 0], 'score': -3.66516292749662}

    print(f"Your candidate. Decoded IDs: {candidate.decoded_ids} Score: {candidate.score}")
    print(f"Gold candidate. Decoded IDs: {gold_candidate['decoded_ids']} Score: {gold_candidate['score']}")
    assert candidate.decoded_ids == gold_candidate['decoded_ids']
    assert math.isclose(candidate.score, gold_candidate['score'], abs_tol=1e-3)


def test_nucleus_sampling(model):
    print('\nTesting Nucleus Sampling...\n-----------------')

    # set seed for deterministic running/testing
    random.seed(2, version=1)

    # Filter for the smallest # of IDs where the accumulated prob is >= 0.7
    top_p = 0.7
    # Only decode up to 6 IDs
    max_length = 6
    # If we have generated the ID 0 (end-of-sentence ID), stop generating
    eos_id = 0

    # Call beam search to get top `beam_size` candidates
    candidate = nucleus_sampling(
        model=model,
        top_p=top_p,
        max_length=max_length,
        eos_id=eos_id,
    )

    # Check the generated candidate against gold candidate
    gold_candidate = {'decoded_ids': [2, 1, 2, 0], 'score': -4.240527072400182}

    print(f"Your candidate. Decoded IDs: {candidate.decoded_ids} Score: {candidate.score}")
    print(f"Gold candidate. Decoded IDs: {gold_candidate['decoded_ids']} Score: {gold_candidate['score']}")
    assert candidate.decoded_ids == gold_candidate['decoded_ids']
    assert math.isclose(candidate.score, gold_candidate['score'], abs_tol=1e-3)


def test_beam_search_decoder(model):
    print('\nTesting Beam Search Decoder...\n-----------------')

    # Keep around 3 candidates at each time point
    beam_size=3
    # Only decode up to 6 IDs
    max_length = 6
    # If we have generated the ID 0 (end-of-sentence ID), stop generating
    eos_id = 0

    # Call beam search to get top `beam_size` candidates
    candidates = beam_search_decoding(
        model=model,
        beam_size=beam_size,
        max_length=max_length,
        eos_id=eos_id,
    )

    # Check the generated candidates against gold candidates
    gold_candidates = [{'decoded_ids': [3, 2, 2, 0], 'score': -3.66516292749662},
                       {'decoded_ids': [3, 2, 1, 0], 'score': -3.952844999948401},
                       {'decoded_ids': [3, 2, 2, 2, 1, 1], 'score': -5.785426463696711}]
    assert len(candidates) == len(gold_candidates)

    for i, (cand, gold_cand) in enumerate(zip(candidates, gold_candidates)):
        print(f"Testing candidate {i}...")
        print(f"Your candidate. Decoded IDs: {cand.decoded_ids} Score: {cand.score}")
        print(f"Gold candidate. Decoded IDs: {gold_cand['decoded_ids']} Score: {gold_cand['score']}\n")
        assert cand.decoded_ids == gold_cand['decoded_ids']
        assert math.isclose(cand.score, gold_cand['score'], abs_tol=1e-3)


def main():
    # Model used for testing.
    # Contains next ID probabilities over 7 decoding steps over a vocab of 4 IDs
    # This model is conditionally independent, meaning that no matter what
    # the previously decoded ID was, the following probabilities is fixed.
    #
    # We use a ID of 0 as the end-of-sentence ID.
    model = FixedModel([[0.1, 0.2, 0.3, 0.4],
                        [0.2, 0.3, 0.4, 0.1],
                        [0.1, 0.3, 0.4, 0.2],
                        [0.4, 0.2, 0.3, 0.1],
                        [0.1, 0.4, 0.2, 0.3],
                        [0.1, 0.4, 0.2, 0.3],
                        [0.1, 0.2, 0.3, 0.4]])

    test_temperature_random_sampling(model)
    test_nucleus_sampling(model)
    test_beam_search_decoder(model)


if __name__ == "__main__":
    main()

