#!/bin/python
import numpy as np

class Decoder:
    def decode(self, source_sent):
        """Returns a sequence of translations for the input sentence."""
        pass

class State:
    """Represents a partial translation in a stack decoder.

    n is the number of foreign tokens (in total)
    e1,e2 are the last two tokens of the translation so far
    r is the token index of the source sentence that the last phrase covers
    b is the bit vector representing the foreign tokens covered by the state
    score is the score of the partial translation
    """
    def __init__(self, n, e1="<s>", e2="<s>", b = None, r=0, score=0):
        self.n = n
        self.e1 = e1
        self.e2 = e2
        self.bkey = 0 # int representation of b
        self.len = 0 # number of 1s in b
        if b is not None:
            self.b = b
            self.first_empty = n # index of the first untranslated token
            for i in xrange(n):
                if self.b[i]:
                    self.bkey += 2**i
                    self.len += 1
                elif i < self.first_empty:
                    self.first_empty = i
        else:
            self.b = np.zeros(n,dtype=np.bool)
            self.first_empty = 0
        self.r = r
        self.score = score
        self.bp = None # backpointer to which partial translation this came from
        self.key = (self.r, self.bkey, self.e1, self.e2) # key for equality

    def eq(self, q):
        return self.r==q.r and self.e1==q.e1 and self.e2==q.e2 and (self.bkey==q.bkey) #.all()

    def __str__(self):
        return "bp:{}".format(self.bp)

class StateList:
    """A Simple implementation of a stack.

    Since the most common operation for multi-stack decoder is add and find,
    this stack implementation is mostly based on the dictionary. When the
    top-K are needed, it sorts the values and picks the top ones.
    """
    def __init__(self):
        self.map = dict()

    def add(self, q):
        assert q.key not in self.map
        self.map[q.key] = q

    def remove(self, q):
        del self.map[q.key]

    def find(self, qp):
        if qp.key in self.map:
            return [qp]
        else:
            return []

    def states(self): return self.map.values()

    def topK(self, k): return sorted(self.states(), key = lambda q: q.score, reverse = True)[:k]

class StackDecoder(Decoder):
    """Basic stack-based decoder that implements the monotonic decoder by default."""

    def CheckDistLimit(self, r, p):
        """Ensure p comes after r (for monotonic)."""
        return r == p.s-1

    def index_phrases(self, n, P):
        """Indexes phrases s.t. index[r] are phrases that can appear after r."""
        index = []
        for r in xrange(n+1):
            index.append([])
            for p in P:
                if self.CheckDistLimit(r,p):
                    index[r].append(p)
        return index

    def CheckOverlap(self, q, p):
        """Check whether state q and p overlap, not needed for monotonic."""
        return True

    def Compatible(self, q, P, dist_index):
        """Return the set of valid phrases for state q.
        A valid phrase is one that is not too far (CheckDistLimit) and does not
        overlap with the translations in the state (CheckOverlap).
        Corresponds to ph(q) in Collins notes.
        """
        phrases = []
        for p in dist_index[q.r]:
            if self.CheckOverlap(q,p):
                phrases.append(p)
        return phrases

    def Beam(self, Q):
        """Return the top elements of Q"""
        return Q.topK(self.beam_w)

    def lm_score(self, words):
        """Compute the language model score of the words, conditioning on the
        first two words."""
        return 0.0

    def dist_score(self, q, p):
        """Compute the distortion penalty between q and p"""
        return self.model.dist_penalty*abs(q.r+1-p.s)

    def Next(self, q, p):
        """Generate the next state from q and p."""
        words = [q.e1, q.e2] + p.e
        M = len(p.e)
        e1p = words[M] # second last word
        e2p = words[M+1] # last word
        bp = np.copy(q.b) # new b
        for i in xrange(p.s-1,p.t):
            bp[i] = True
        rp = p.t # new r
        lm_s = self.lm_score(words)
        dist_s = self.dist_score(q, p)
        score = q.score + p.score + lm_s + dist_s
        return State(q.n,e1p,e2p,bp,rp,score)

    def Add(self, Q, qp, q, p):
        """Add the state qp to Q, with backpointer (q,p)."""
        matches = Q.find(qp)
        assert len(matches) <= 1
        if len(matches)>0:
            qpp = matches[0]
            if qp.score > qpp.score:
                Q.remove(qpp)
                Q.add(qp)
                qp.bp = (q,p)
        else:
            Q.add(qp)
            qp.bp = (q,p)

    def decode(self, source_sent):
        n = len(source_sent)
        # create set of possible phrases
        P = self.model.phrase_table.phrases(source_sent)
        # precompute phrases satifying distorition limits
        dist_index = self.index_phrases(n, P)
        # initialize n+1 stacks
        Q = []
        Q.append(StateList())
        Q[0].add(State(n)) # Q0 has initial state
        for i in xrange(n):
            Q.append(StateList()) # Qi are empty
        # star the loop
        for i in range(n):
            next_states = self.Beam(Q[i])
            print "i ", i, ", with ", len(next_states), " states"
            for q in next_states:
                next_phrases = self.Compatible(q,P,dist_index)
                for p in next_phrases:
                    qp = self.Next(q,p)
                    j = qp.len
                    self.Add(Q[j], qp, q, p)
        # return the max score in Q[n]
        print "i ", n, ", with ", len(Q[n].states()), " states"
        bestq = Q[n].topK(1)[0]
        # construct the english sentence
        eng = []
        curr = bestq.bp
        while curr:
            eng = curr[1].e + eng
            curr = curr[0].bp
        return eng
