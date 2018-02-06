#!/bin/python

from decoder import *

class MonotonicDecoder(StackDecoder):
    def __init__(self, model, beam_w):
        self.beam_w = beam_w
        self.model = model

class MonotonicLMDecoder(MonotonicDecoder):
    def __init__(self, model, beam_w):
        self.beam_w = beam_w
        self.model = model

    def lm_score(self, words):
        return 0.0

class NonMonotonicLMDecoder(MonotonicLMDecoder):
    def __init__(self, model, beam_w):
        self.beam_w = beam_w
        self.model = model

    def CheckOverlap(self, q, p):
        """Ensure q's bit vector is 0 for p."""
        return True

    def CheckDistLimit(self, r, p):
        """Ensure q doesn't have p to be too far."""
        return r == (p.s - 1)
