import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores, as an LxL array
    - Start transition scores, as an Lx1 array
    - End transition scores, as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the array of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    y = []
    for i in xrange(N):
        # stupid sequence
        y.append(i % L)
    # score set to 0
    return (0.0, y)
