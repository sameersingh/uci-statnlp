#!/bin/python
import feat_gen

class Feats:
    def __init__(self):
        self.fmap = dict()
        self.feats = []
        self.frozen = False
        self.token2features = feat_gen.token2features
        self.preprocess_corpus = feat_gen.preprocess_corpus
        self.num_features = len(self.feats)

    def add_feature(self, ftr):
        assert self.frozen == False
        if ftr not in self.fmap:
            fidx = len(self.feats)
            self.fmap[ftr] = fidx
            self.feats.append(ftr)
            if self.num_features % 1000 == 0:
                print "--", self.num_features, "features added."
            self.num_features = len(self.feats)
        return self.fmap[ftr]

    def freeze(self):
        self.frozen = True
        self.num_features = len(self.feats)

    def get_index(self, ftr):
        return self.fmap[ftr]

    def get_ftr_name(self, findex):
        return self.feats[findex]

    def index_data(self, sents):
        self.preprocess_corpus(sents)
        idxs = []
        for s in sents:
            idxs.append(self.index_sent(s))
        self.freeze()
        assert len(idxs) == len(sents)
        return idxs

    def index_sent(self, sent):
        sentIdxs = []
        for i in xrange(len(sent)):
            tokIdxs = []
            ftrs = self.token2features(sent, i)
            for ftr in ftrs:
                idx = self.add_feature(ftr)
                tokIdxs.append(idx)
            sentIdxs.append(tokIdxs)
        assert len(sentIdxs) == len(sent)
        return sentIdxs

    def token2fidxs(self, sent, i):
        ftrs = self.token2features(sent, i)
        fidxs  = []
        for ftr in ftrs:
            if ftr in self.fmap:
                fidxs.append(self.get_index(ftr))
        return fidxs

    def fidxs2names(self, fv):
        (rows,cols) = fv.nonzero()
        fnames = []
        for i in cols:
            fnames.append(self.get_ftr_name(i))
        return fnames
