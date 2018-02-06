from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import sys

if sys.version < '3':
    from codecs import getwriter
    stderr = getwriter('utf-8')(sys.stderr)
    stdout = getwriter('utf-8')(sys.stdout)
else:
    stderr = sys.stderr
    stdout = sys.stdout

class LangModel:
    def logprob_sentence(self, sentence):
        p = 0.0
        sentence = ["<s>", "<s>"] + sentence
        for i in xrange(2,len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('</s>', sentence)
        return p

    # return the log of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass

class SimpleLangModel(LangModel):
    def __init__(self, lbackoff=-10.0):
        self.lbackoff = lbackoff

    def cond_logprob(self, word, previous): return self.lbackoff

class KneserNeyLm(LangModel):

    def __init__(self, filename):
        self.map = dict() # ngram -> (prob,backoff)
        self.encoder = lambda x : x
        self.debug = True
        self.read_file(filename)
        self.unknownWordLogProb = -100.0

    def cond_logprob(self, word, prev):
        history = tuple(prev)
        lookup = history + tuple([word])
        if len(lookup) > self.order:
            lookup = lookup[-self.order:]
        try:
            return self.map[lookup][0]
        except KeyError:  # not found, back off
            if len(prev)==0:
                return self.unknownWordLogProb
            else:
                try:
                    backoffweight = self.map[history][1]
                except KeyError:
                    backoffweight = 0  # backoff weight will be 0 if not found
                return backoffweight + self.cond_logprob(word, history[1:])

    def read_file(self, filename):
        # Copied from https://github.com/proycon/pynlpl/blob/master/lm/lm.py
        import gzip
        with gzip.open(filename, 'rt') as f:
            order = None
            for line in f:
                line = line.strip().decode('utf-8')
                if line == '\\data\\':
                    order = 0
                elif line == '\\end\\':
                    break
                elif line.startswith('\\') and line.endswith(':'):
                    for i in range(1, 10):
                        if line == '\\{}-grams:'.format(i):
                            order = i
                            break
                    else:
                        raise ValueError("Order of n-gram is not supported!")
                elif line:
                    if order == 0:  # still in \data\ section
                        if line.startswith('ngram'): pass
                    elif order > 0:
                        fields = line.split('\t')
                        logprob = float(fields[0])
                        ngram = self.encoder(tuple(fields[1].split()))
                        if len(fields) > 2:
                            backoffprob = float(fields[2])
                            if self.debug and len(self.map)%1000000 == 0:
                                msg = "Adding to LM ({}): {}\t{}\t{}"
                                print(msg.format(len(self.map),ngram, logprob, backoffprob), file=stderr)
                        else:
                            backoffprob = 0.0
                            if self.debug and len(self.map)%1000000 == 0:
                                msg = "Adding to LM ({}): {}\t{}"
                                print(msg.format(len(self.map),ngram, logprob), file=stderr)
                        self.map[ngram] = (logprob, backoffprob)
                    elif self.debug:
                        print("Unable to parse ARPA LM line: " + line, file=stderr)
        self.order = order

if __name__ == "__main__":
    lm = KneserNeyLm("resources/data/filtered_lm.gz")
