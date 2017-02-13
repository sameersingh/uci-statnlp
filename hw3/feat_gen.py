#!/bin/python

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this as efficient as possible."""
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    ftrs.append("IS_ALNUM=" + str(word.isalnum()))
    ftrs.append("IS_NUMERIC=" + str(word.isnumeric()))
    ftrs.append("IS_DIGIT=" + str(word.isdigit()))
    ftrs.append("IS_UPPER=" + str(word.isupper()))
    ftrs.append("IS_LOWER=" + str(word.islower()))

    # previous word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs
