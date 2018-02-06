#!/bin/py
# -*- coding: utf-8 -*-

from phrase import PhraseTable
from phrase import TranslationModel
import gzip
import bleu_score
import cProfile

def load_model(lfilename, pfilename):
    """Create the translation model.
    Does not read the language model if lfilename is None.
    """
    dist_limit = 3
    dist_penalty = -0.2
    # read the lm
    print("  language model..")
    import lm
    if lfilename:
        langm = lm.KneserNeyLm(lfilename)
    else:
        langm = lm.SimpleLangModel()
    # read the phrase table
    print("  phrase table..")
    import phrase
    table = phrase.PhraseTable(pfilename)
    # construct the model
    model = phrase.TranslationModel(table, langm, dist_limit, dist_penalty)
    return model

def read_text(filename):
    """Converts input string to a corpus of tokenized sentences.
    Assumes that the sentences are divided by newlines (but will ignore empty sentences).
    """
    import io
    corpus = []
    with io.open(filename, "rt", encoding="utf-8") as f:
        text = f.read().strip()
        sents = text.split("\n")
        for s in sents:
            toks = s.strip().split(" ")
            if len(toks) > 0:
                corpus.append(toks)
    return corpus

def trans_unigrams(source_sents, phrase_table):
    """Creates a list of translated unigrams.
    Not needed for the homework assignment, used to prune language models.
    """
    # collect list of unigrams
    unigrams = set()
    unigrams.add("<s>")
    unigrams.add("</s>")
    for s in source_sents:
        P = phrase_table.phrases(s)
        for p in P:
            for w in p.e:
                unigrams.add(w)
    print("Number of unigrams: {}".format(len(unigrams)))
    return unigrams

def prune_lm(unigrams, infile, outfile):
    """Prune the language models to only contain phrases that appear in the set
    of unigrams. Not needed for the homework submission.
    """
    out = gzip.open(outfile, 'wt')
    num_lines = 0
    with gzip.open(infile, 'rt') as f:
        order = None
        for line in f:
            num_lines += 1
            line = line.strip()
            if line == '\\data\\':
                out.write(line + "\n")
                order = 0
            elif line == '\\end\\':
                out.write(line + "\n")
                break
            elif line.startswith('\\') and line.endswith(':'):
                order = 1
                out.write(line + "\n")
            elif line:
                if order == 0:  # still in \data\ section
                    out.write(line + "\n")
                elif order > 0:
                    fields = line.split('\t')
                    add = True
                    for w in fields[1].split():
                        add = add and (w in unigrams)
                    if num_lines % 1000000 == 0:
                        print("Read {} lines".format(num_lines))
                    if add:
                        out.write(line + "\n")
                else:
                    print("ERROR: " + line)
    out.close()

def pruned_resources(fr_sents, en_sents, prefix):
    """Prunes the language model, not needed for the homework."""
    pt = PhraseTable("data/phrasetable.txt.gz")
    unigrams = trans_unigrams(fr_sents, pt)
    for s in en_sents:
        for tok in s:
            unigrams.add(tok)
    prune_lm(unigrams, "data/lm.gz", "data/" + prefix + "_lm.gz")

def str_sent(sent):
    """Pretty print a sequence of tokens."""
    string = ""
    for s in sent: string = string + " " + unicode(s)
    return string

if __name__ == "__main__":
    print("Reading sentences..")
    fr_sents = read_text("data/test.fr")
    en_sents = read_text("data/test.en")
    # Used to create the smaller language models, do not uncomment
    # pruned_resources(fr_sents, en_sents, "filtered")
    print("Reading the model..")
    #model = load_model(None, "data/phrasetable.txt.gz")
    model = load_model("data/filtered_lm.gz", "data/phrasetable.txt.gz")
    from decoder import *
    beam_w = 100
    max_length = 10
    from submission import *
    decoder = MonotonicDecoder(model, beam_w)
    #decoder = MonotonicLMDecoder(model, beam_w)
    #decoder = NonMonotonicLMDecoder(model, beam_w)
    print "---------------------"
    print("french: " + str_sent(["rapport"]))
    tr_sent = decoder.decode(["rapport"])
    print("english: " + str_sent(tr_sent))
    print "---------------------"
    print "---------------------"
    print("french: " + str_sent(["président".decode("utf-8")]))
    tr_sent = decoder.decode(["président".decode("utf-8")])
    print("english: " + str_sent(tr_sent))
    print "---------------------"
    # pr = cProfile.Profile()
    # pr.enable()
    print "---------------------"
    print("french: " + str_sent(["le", "rapport"]))
    tr_sent = decoder.decode(["le", "rapport"])
    print("english: " + str_sent(tr_sent))
    print "---------------------"
    # pr.disable()
    # pr.print_stats(sort="cumulative")
    for i in xrange(len(fr_sents)):
        fr_sent = fr_sents[i]
        if len(fr_sent) <= max_length:
            print "---------------------"
            print("french: " + str_sent(fr_sent))
            en_sent = en_sents[i]
            tr_sent = decoder.decode(fr_sent)
            print("gold: " + str_sent(en_sent))
            print("pred: " + str_sent(tr_sent))
            print("bleu: " + str(bleu_score.sentence_bleu([en_sent], tr_sent)))
            print "---------------------"
