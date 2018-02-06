#!/bin/python

class Phrase:
    """A phrase translation specific for a sentence."""
    def __init__(self,s,t,e,score):
        self.s = s
        self.t = t
        self.e = e
        self.score = score

class TranslationModel:
    """A phrase translation model, wrapper for lm+pt+hyper params."""
    def __init__(self, phrase_table, lm, dist_limit, dist_penalty):
        self.phrase_table = phrase_table
        self.lm = lm
        self.dist_limit = dist_limit
        self.dist_penalty = dist_penalty

class PhraseTable:
    """Stores the phrase table (the lexicon)."""

    def __init__(self, filename):
        self.table = dict() # of source -> list(translation,score)
        self.min_score = 0
        self.read_file(filename)

    def read_file(self, filename):
        """Read the phrase table from the file."""
        import gzip
        import math
        num_vals = 0
        vals = []
        with gzip.open(filename, 'rt') as f:
            for line in f:
                line = line.strip()
                parts = line.split("|||")
                assert len(parts) == 5
                src = parts[0].strip().decode("utf-8").split(" ")
                trg = parts[1].strip().decode("utf-8").split(" ")
                feats = parts[4].strip().split(" ")
                assert len(feats) == 5
                vals = map(lambda x: float(x), feats)
                score = 0.0
                # P(f|e)    -0.33
                score += (-math.log(vals[0]) * -0.33)
                # lex(f|e)    -0.25
                score += (-math.log(vals[1]) * -0.25)
                # P(e|f)    -1.0
                score += (-math.log(vals[2]) * -1.0)
                # lex(e|f)    -0.35
                score += (-math.log(vals[3]) * -0.35)
                # bias    -0.4
                score += (-math.log(vals[4]) * -0.4)
                if score < self.min_score:
                    self.min_score = score
                # wordBonus    2.0
                score += (len(trg) * 2.0)
                key = tuple(src)
                if key not in self.table:
                    self.table[key] = []
                num_vals += 1
                vals.append(score)
                self.table[key].append((trg,score))
            print("Phrase table read: {} keys, {} vals".format(len(self.table), num_vals))
            min_s = min(vals)
            max_s = max(vals)
            avg_s = sum(vals)/len(vals)
            print("Score dist, min: {}, max: {}, avg: {}".format(min_s, max_s, avg_s))

    def check_phrase(self, phrase):
        """Check whether a foreign phrase exists in the table.
        Returns a tuple containing whether the translations were found and the
        list of the translations, i.e {(e,score)}"""
        tp = tuple(phrase)
        if tp in self.table:
            return (True, self.table[tp])
        else:
            return (False, None)

    def phrases(self, source_sent):
        """Return the set of valid phrases for the sentence."""
        P = []
        n = len(source_sent)
        for s in xrange(n):
            # include word dropping
            # p = Phrase(s+1,s+1,[],self.min_score)
            # P.append(p)
            for t in xrange(s,n):
                phrase = source_sent[s:t+1]
                (present, trans) = self.check_phrase(phrase)
                if present:
                    for tt in trans:
                        p = Phrase(s+1,t+1,tt[0],tt[1])
                        P.append(p)
                elif t-s == 0:
                    # include word copying
                    p = Phrase(s+1,t+1,[source_sent[s]],self.min_score + 2.0)
                    P.append(p)
        return P

if __name__ == "__main__":
    pt = PhraseTable("data/phrasetable.txt.gz")
    pt.phrases(["je", "m", "\'", "appelle"])
    #pt.phrases(["ne", "vous", "en", "faites", "pas"])
