#!/bin/python

# Ported from http://www.nltk.org/_modules/nltk/translate/bleu_score.html
# by Zhengli Zhao, UCI

# This code has the following license

# Copyright (C) 2001-2017 NLTK Project

# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import math
from fractions import Fraction
import warnings
from collections import Counter
from itertools import chain

def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):

    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):

    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n-1))
    return sequence

def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                  smoothing_function=None, auto_reweigh=False):

    return corpus_bleu([references], [hypothesis],
                        weights, smoothing_function, auto_reweigh)

def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None, auto_reweigh=False):
    # Before proceeding to compute BLEU, perform sanity checks.
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), "The number of hypotheses and their reference(s) should be the same"

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = ( 1 / hyp_lengths ,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i])
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method1 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function=SmoothingFunction().method1
        # smoothing_function = SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(p_n, references=references,
                             hypothesis=hypothesis, hyp_len=hyp_len)
    s = (w * math.log(p_i) for i, (w, p_i) in enumerate(zip(weights, p_n)))
    return bp * math.exp(math.fsum(s))

def modified_precision(references, hypothesis, n):
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    # Extract a union of references' counts.
    ## max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {ngram: min(count, max_counts[ngram])
                      for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator)

def closest_ref_length(references, hyp_len):

    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
                          (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len

def brevity_penalty(closest_ref_len, hyp_len):

    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

class SmoothingFunction:
    """
    This is an implementation of the smoothing techniques
    for segment-level BLEU scores that was presented in
    Boxing Chen and Collin Cherry (2014) A Systematic Comparison of
    Smoothing Techniques for Sentence-Level BLEU. In WMT14.
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """
    def __init__(self, epsilon=0.1, alpha=5, k=5):

        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method0(self, p_n, *args, **kwargs):
        """ No smoothing. """
        p_n_new = []
        for i, p_i in enumerate(p_n):
            if p_i.numerator != 0:
                p_n_new.append(p_i)
            else:
                _msg = str("\nCorpus/Sentence contains 0 counts of {}-gram overlaps.\n"
                           "BLEU scores might be undesirable; "
                           "use SmoothingFunction().").format(i+1)
                warnings.warn(_msg)
                # If this order of n-gram returns 0 counts, the higher order
                # n-gram would also return 0, thus breaking the loop here.
                break
        return p_n_new

    def method1(self, p_n, *args, **kwargs):
        """
        Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
        """
        return [(p_i.numerator + self.epsilon)/ p_i.denominator
                if p_i.numerator == 0 else p_i for p_i in p_n]

    def method2(self, p_n, *args, **kwargs):
        """
        Smoothing method 2: Add 1 to both numerator and denominator from
        Chin-Yew Lin and Franz Josef Och (2004) Automatic evaluation of
        machine translation quality using longest common subsequence and
        skip-bigram statistics. In ACL04.
        """
        return [Fraction(p_i.numerator + 1, p_i.denominator + 1) for p_i in p_n]

    def method3(self, p_n, *args, **kwargs):
        """
        Smoothing method 3: NIST geometric sequence smoothing
        The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
        precision score whose matching n-gram count is null.
        k is 1 for the first 'n' value for which the n-gram match count is null/
        For example, if the text contains:
         - one 2-gram match
         - and (consequently) two 1-gram matches
        the n-gram count for each individual precision score would be:
         - n=1  =>  prec_count = 2     (two unigrams)
         - n=2  =>  prec_count = 1     (one bigram)
         - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
         - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
        """
        incvnt = 1 # From the mteval-v13a.pl, it's referred to as k.
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0:
                p_n[i] = 1 / (2**incvnt * p_i.denominator)
                incvnt+=1
        return p_n

    def method4(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 4:
        Shorter translations may have inflated precision values due to having
        smaller denominators; therefore, we give them proportionally
        smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
        suggests dividing by 1/ln(len(T)), where T is the length of the translation.
        """
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0 and hyp_len != 0:
                incvnt = i+1 * self.k / math.log(hyp_len) # Note that this K is different from the K from NIST.
                p_n[i] = 1 / incvnt
        return p_n

    def method5(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 5:
        The matched counts for similar values of n should be similar. To
        calculate the n-gram matched count, it averages the n-1, n and n+1 gram
        matched counts.
        """
        m = {}
        # Requires an precision value for an addition ngram order.
        p_n_plus1 = p_n + [modified_precision(references, hypothesis, 5)]
        m[-1] = p_n[0] + 1
        for i, p_i in enumerate(p_n):
            p_n[i] = (m[i-1] + p_i + p_n_plus1[i+1]) / 3
            m[i] = p_n[i]
        return p_n

    def method6(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 6:
        Interpolates the maximum likelihood estimate of the precision *p_n* with
        a prior estimate *pi0*. The prior is estimated by assuming that the ratio
        between pn and pn-1 will be the same as that between pn-1 and pn-2; from
        Gao and He (2013) Training MRF-Based Phrase Translation Models using
        Gradient Ascent. In NAACL.
        """
        # This smoothing only works when p_1 and p_2 is non-zero.
        # Raise an error with an appropriate message when the input is too short
        # to use this smoothing technique.
        assert p_n[2], "This smoothing method requires non-zero precision for bigrams."
        for i, p_i in enumerate(p_n):
            if i in [0,1]: # Skips the first 2 orders of ngrams.
                continue
            else:
                pi0 = 0 if p_n[i-2] == 0 else p_n[i-1]**2 / p_n[i-2]
                # No. of ngrams in translation that matches the reference.
                m = p_i.numerator
                # No. of ngrams in translation.
                l = sum(1 for _ in ngrams(hypothesis, i+1))
                # Calculates the interpolated precision.
                p_n[i] = (m + self.alpha * pi0) / (l + self.alpha)
        return p_n

    def method7(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 7:
        Interpolates the maximum likelihood estimate of the precision *p_n* with
        a prior estimate *pi0*. The prior is estimated by assuming that the ratio
        between pn and pn-1 will be the same as that between pn-1 and pn-2.
        """
        p_n = self.method4(p_n, references, hypothesis, hyp_len)
        p_n = self.method5(p_n, references, hypothesis, hyp_len)
        return p_n


if __name__ == "__main__":

    hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', \
    'ensures', 'that', 'the', 'military', 'always', \
    'obeys', 'the', 'commands', 'of', 'the', 'party']

    hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops', \
    'forever', 'hearing', 'the', 'activity', 'guidebook', \
    'that', 'party', 'direct']

    reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', \
    'ensures', 'that', 'the', 'military', 'will', 'forever', \
    'heed', 'Party', 'commands']

    reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', \
    'guarantees', 'the', 'military', 'forces', 'always', \
    'being', 'under', 'the', 'command', 'of', 'the', 'Party']

    reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', \
    'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']

    print sentence_bleu([reference1, reference2, reference3], hypothesis1, \
        smoothing_function=SmoothingFunction().method1)
    # 0.5045...

    print sentence_bleu([reference1, reference2, reference3], hypothesis2, \
        smoothing_function=SmoothingFunction().method1)
    # 0.1255...
