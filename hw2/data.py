#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import numpy as np
import os
import pickle
import sys


# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)


def textToTokens(text):
    """Converts input string to a corpus of tokenized sentences.

    Assumes that the sentences are divided by newlines (but will ignore empty sentences).
    You can use this to try out your own datasets, but is not needed for reading the homework data.
    """
    corpus = []
    sents = text.split("\n")
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(sents)
    tokenizer = count_vect.build_tokenizer()
    for s in sents:
        toks = tokenizer(s)
        if len(toks) > 0:
            corpus.append(toks)
    return corpus

def file_splitter(filename, seed = 0, train_prop = 0.7, dev_prop = 0.15,
    test_prop = 0.15):
    """Splits the lines of a file into 3 output files."""
    import random
    rnd = random.Random(seed)
    basename = filename[:-4]
    train_file = open(basename + ".train.txt", "w")
    test_file = open(basename + ".test.txt", "w")
    dev_file = open(basename + ".dev.txt", "w")
    with open(filename, 'r') as f:
        for l in f.readlines():
            p = rnd.random()
            if p < train_prop:
                train_file.write(l)
            elif p < train_prop + dev_prop:
                dev_file.write(l)
            else:
                test_file.write(l)
    train_file.close()
    test_file.close()
    dev_file.close()

def read_texts(tarfname, dname):
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
    train_mem = tar.getmember(dname + ".train.txt")
    train_txt = unicode(tar.extractfile(train_mem).read(), errors='replace')
    test_mem = tar.getmember(dname + ".test.txt")
    test_txt = unicode(tar.extractfile(test_mem).read(), errors='replace')
    dev_mem = tar.getmember(dname + ".dev.txt")
    dev_txt = unicode(tar.extractfile(dev_mem).read(), errors='replace')

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(train_txt.split("\n"))
    tokenizer = count_vect.build_tokenizer()
    class Data: pass
    data = Data()
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    data.dev = []
    for s in dev_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.dev.append(toks)
    print(dname," read.", "train:", len(data.train), "dev:", len(data.dev), "test:", len(data.test))
    return data

def learn_unigram(data):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Unigram
    unigram = Unigram()
    unigram.fit_corpus(data.train)
    print("vocab:", len(unigram.vocab()))
    # evaluate on train, test, and dev
    print("train:", unigram.perplexity(data.train))
    print("dev  :", unigram.perplexity(data.dev))
    print("test :", unigram.perplexity(data.test))
    from generator import Sampler
    sampler = Sampler(unigram)
    for _ in range(2):
        print("sample: ", " ".join(str(x) for x in sampler.sample_sentence([], max_length=20)))
    return unigram

def print_table(table, row_names, col_names, latex_file = None):
    """Pretty prints the table given the table, and row and col names.

    If a latex_file is provided (and tabulate is installed), it also writes a
    file containing the LaTeX source of the table (which you can \\input into your report)
    """
    try:
        from tabulate import tabulate
        rows = list(map(lambda rt: [rt[0]] + rt[1], zip(row_names, table.tolist())))

        # compute avg in domain perplexity and add to table
        avg_in_domain_ppl = np.mean(np.diagonal(table))
        rows = [row + ['-'] for row in rows]
        rows.append(['Avg In-Domain'] + ['-']*len(rows) + [avg_in_domain_ppl])
        row_names.append('Avg In-Domain')

        print(tabulate(rows, headers = [""] + col_names))
        if latex_file is not None:
            latex_str = tabulate(rows, headers = [""] + col_names, tablefmt="latex")
            with open(latex_file, 'w') as f:
                f.write(latex_str)
                f.close()
    except ImportError as e:
        row_format ="{:>15} " * (len(col_names) + 1)
        print(row_format.format("", *col_names))
        for row_name, row in zip(row_names, table):
            print(row_format.format(row_name, *row))

def save_lms(dnames, models, output_dir):
    # write out LMs trained on different datasets in individual files
    for name, model in zip(dnames, models):
        with open(os.path.join(output_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='name of directory to write out trained language models. '
                             'If it exists, the previous contents will be overwritten.')
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Do no run, the following function was used to generate the splits
    # file_splitter("data/reuters.txt")
    dnames = ["brown", "reuters", "gutenberg"]
    datas = []
    models = []
    # Learn the models for each of the domains, and evaluate it
    for dname in dnames:
        print("-----------------------")
        print(dname)
        data = read_texts("data/corpora.tar.gz", dname)
        datas.append(data)
        model = learn_unigram(data)
        models.append(model)
    # compute the perplexity of all pairs
    n = len(dnames)
    perp_dev = np.zeros((n,n))
    perp_test = np.zeros((n,n))
    perp_train = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            perp_dev[i][j] = models[i].perplexity(datas[j].dev)
            perp_test[i][j] = models[i].perplexity(datas[j].test)
            perp_train[i][j] = models[i].perplexity(datas[j].train)

    print("-------------------------------")
    print("x train")
    print_table(perp_train, dnames, dnames, os.path.join(output_dir, "table-train.tex"))
    print("-------------------------------")
    print("x dev")
    print_table(perp_dev, dnames, dnames, os.path.join(output_dir, "table-dev.tex"))
    print("-------------------------------")
    print("x test")
    print_table(perp_test, dnames, dnames, os.path.join(output_dir, "table-test.tex"))
    print("-------------------------------")
    print("saving language models to file")
    save_lms(dnames, models, output_dir)
