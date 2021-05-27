import math
import random

from decoders import beam_search_decoding, nucleus_sampling
from models import FixedModel


def test_nucleus_sampling():
    print('\nTesting Nucleus Sampling...\n-----------------')

    # set seed for deterministic running/testing
    random.seed(2, version=1)

    # Next ID probabilities over 7 decoding steps over a vocab of 4 IDs (tokens)
    model = FixedModel([[0.1, 0.2, 0.3, 0.4],
                        [0.2, 0.3, 0.4, 0.1],
                        [0.1, 0.3, 0.4, 0.2],
                        [0.4, 0.2, 0.3, 0.1],
                        [0.1, 0.4, 0.2, 0.4],
                        [0.1, 0.4, 0.2, 0.3],
                        [0.1, 0.2, 0.3, 0.4]])

    # Filter for the smallest number of top IDs where the probability is >= 0.5
    top_p = 0.5
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


def test_beam_search_decoder():
    print('\nTesting Beam Search Decoder...\n-----------------')

    # Next ID probabilities over 7 decoding steps over a vocab of 4 IDs (tokens)
    model = FixedModel([[0.1, 0.2, 0.3, 0.4],
                        [0.2, 0.3, 0.4, 0.1],
                        [0.1, 0.3, 0.4, 0.2],
                        [0.4, 0.2, 0.3, 0.1],
                        [0.1, 0.4, 0.2, 0.4],
                        [0.1, 0.4, 0.2, 0.3],
                        [0.1, 0.2, 0.3, 0.4]])

    # Keep around 3 candidates at each time point
    beam_size=3
    # Only decode up to 6 IDs
    max_length = 6
    # If a candidate on the beam has generated 0 (EOS ID) then the beam is done
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


if __name__ == "__main__":
    test_nucleus_sampling()
    test_beam_search_decoder()

