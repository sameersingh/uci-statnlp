#!/bin/python

def run_viterbi_test():
    """A simple tester for Viterbi algorithm.

    This function generates a bunch of random emission and transition scores,
    and computes the best sequence by performing a brute force search over all
    possible sequences and scoring them. It then runs Viterbi to see what is the
    score and sequence returned by it.

    Compares both the best sequence and its score to make sure Viterbi is correct.
    """
    from viterbi import run_viterbi
    from numpy import random
    import numpy as np
    from itertools import product

    maxN = 7
    maxL = 4
    num_tests = 1000
    random.seed(0)
    tolerance = 1e-5

    emission_var = 1.0
    trans_var = 1.0

    passed_y = 0
    passed_s = 0

    for t in xrange(num_tests):
        N = random.randint(1, maxN+1)
        L = random.randint(2, maxL+1)
        # print N,L

        # Generate the scores
        emission_scores = random.normal(0.0, emission_var, (N,L))
        trans_scores = random.normal(0.0, trans_var, (L,L))
        start_scores = random.normal(0.0, trans_var, L)
        end_scores = random.normal(0.0, trans_var, L)

        (viterbi_s,viterbi_y) = run_viterbi(emission_scores, trans_scores, start_scores, end_scores)
        # print "Viterbi", viterbi_s, viterbi_y
        best_y = []
        best_s = -np.inf
        for y in product(range(L), repeat=N):
            score = 0.0
            score += start_scores[y[0]]
            for i in xrange(N-1):
                score += trans_scores[y[i], y[i+1]]
                score += emission_scores[i,y[i]]
            score += emission_scores[N-1,y[N-1]]
            score += end_scores[y[N-1]]
            if score > best_s:
                best_s = score
                best_y = list(y)
        # print "Brute", best_s, best_y
        match_y = True
        for i in xrange(len(best_y)):
            if viterbi_y[i] != best_y[i]:
                match_y = False
        if match_y: passed_y += 1
        if abs(viterbi_s-best_s) < tolerance:
            passed_s += 1

    print "Passed(y)", passed_y*100.0/num_tests
    print "Passed(s)", passed_s*100.0/num_tests
    assert passed_y == num_tests
    assert passed_s == num_tests

if __name__ == "__main__":
    run_viterbi_test()
