#!/bin/python

from __future__ import print_function

from lm import LangModel
import random
from math import log
import numpy as np

class Sampler:

    def __init__(self, lm, temp = 1.0):
        """Sampler for a given language model.

        Supports the use of temperature, i.e. how peaky we want to treat the
        distribution as. Temperature of 1 means no change, temperature <1 means
        less randomness (samples high probability words even more), and temp>1
        means more randomness (samples low prob words more than otherwise). See
        simulated annealing for what this means.
        """
        self.lm = lm
        self.rnd = random.Random()
        self.temp = temp

    def sample_sentence(self, prefix = [], max_length = 100):
        """Sample a random sentence (list of words) from the language model.

        Samples words till either EOS symbol is sampled or max_length is reached.
        Does not make any assumptions about the length of the context.
        """
        i = 0
        sent = prefix
        word = self.sample_next(sent, False)
        while i <= max_length and word != "END_OF_SENTENCE":
            sent.append(word)
            word = self.sample_next(sent)
            i += 1
        return sent

    def sample_next(self, prev, incl_eos = True):
        """Samples a single word from context.

        Can be useful to debug the model, for example if you have a bigram model,
        and know the probability of X-Y should be really high, you can run
        sample_next([Y]) to see how often X get generated.

        incl_eos determines whether the space of words should include EOS or not.
        """
        wps = []
        tot = -np.inf # this is the log (total mass)
        for w in self.lm.vocab():
            if not incl_eos and w == "END_OF_SENTENCE":
                continue
            lp = self.lm.cond_logprob(w, prev, 0)
            wps.append([w, lp/self.temp])
            tot = np.logaddexp2(lp/self.temp, tot)
        p = self.rnd.random()
        word = self.rnd.choice(wps)[0]
        s = -np.inf # running mass
        for w,lp in wps:
            s = np.logaddexp2(s, lp)
            if p < pow(2, s-tot):
                word = w
                break
        return word

if __name__ == "__main__":
    from lm import Unigram
    unigram = Unigram()
    corpus = [
        [ "sam", "i", "am" ]
    ]
    unigram.fit_corpus(corpus)
    print(unigram.model)
    sampler = Sampler(unigram)
    for i in range(10):
        print(i, ":", " ".join(str(x) for x in sampler.sample_sentence([])))

