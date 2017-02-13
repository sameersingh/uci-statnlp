#!/bin/python
import feat_gen

class Feats:
    """A handy data structure to compute and index token features.

    Since the features we want to compute should be understandable, they
    have nice, human names, stored as strings. However, the classifiers 
    want an index for each feature. And thus, this class represents this 
    mapping.

    You do not need to change anything here, but I have provided comments if you
    want to understand it.

    There are many uses of this class:
    - Compute a growing index of features from a corpus before training
    - Freeze the indices so that no new features are added (once training has started)
    - Compute the features for any token, without growing the list of features
    - Get the total number of token features (to define weight dimensions)
    - Get a name for a feature from its index (currently unused, but you might want to use it)
    - Get an index of a feature from its name (currently unused, but you might want to use it)
    """
    def __init__(self):
        self.fmap = dict()
        self.feats = []
        self.frozen = False
        # uses your code in feat_gen.py to preprocess and compute features
        self.token2features = feat_gen.token2features
        self.preprocess_corpus = feat_gen.preprocess_corpus
        self.num_features = len(self.feats)

    def add_feature(self, ftr):
        """Add a new feature to our index."""
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
        """Freeze the index, no more new features allowed."""
        self.frozen = True
        self.num_features = len(self.feats)

    def get_index(self, ftr):
        """Get the index of a feature from its name."""
        return self.fmap[ftr]

    def get_ftr_name(self, findex):
        """Get the name of a feature from its index."""
        return self.feats[findex]

    def index_data(self, sents):
        """Compute and index the features of a corpus of sentences.

        Freezes the index after the corpus has been indexed.

        Returns a seq of a seq of token features, where each token
        features itself is a list of feature indexes (ints) for the token.
        """
        # call the preprocess code
        self.preprocess_corpus(sents)
        # compute and add the feature indices
        idxs = []
        for s in sents:
            idxs.append(self.index_sent(s))
        self.freeze()
        assert len(idxs) == len(sents)
        return idxs

    def index_sent(self, sent):
        """Compute and index the features of a single sentence."""
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
        """Compute the feature indices of a single token.

        Assumes that the feature indexes are frozen, i.e. does not
        add any more features.
        """
        ftrs = self.token2features(sent, i)
        fidxs  = []
        for ftr in ftrs:
            if ftr in self.fmap:
                fidxs.append(self.get_index(ftr))
        return fidxs

    def fidxs2names(self, fv):
        """Given a sparse feature vector representation of a token,
        returns a list of names of the features that are part of the
        vector.

        Useful for LogisticRegressionTagger, but not directly for CRFTagger.
        """
        (rows,cols) = fv.nonzero()
        fnames = []
        for i in cols:
            fnames.append(self.get_ftr_name(i))
        return fnames
