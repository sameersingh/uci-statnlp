"""Data utils

Types
-----
Data:
    Class containing the train, dev, test splits for a given dataset
    but also its vocabulary (e.g., term frequencies in the training set)
    and the tokenizer used to parse the splits.

Methods
-------
textToTokens(text) --> list of sentences
    Util to parse the specified text into sequences of sentences.

file_splitter(filename, seed, train_prop, dev_prop)
    Opens the specified filename divides its lines into
    training (using train_prop), dev (using dev fraction)
    and test set (remaining lines).

read_texts(tarfname, dname) -> Data
    Given the filepath of a tar archive file and a dataset name,
    uncompress the tar file and parse the file corresponding to
    the name.

print_table
    Pretty prints the table given the table, and row and col names.
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Data:
    train: List[List[str]]
    dev: List[List[str]]
    test: List[List[str]]
    vocabulary: Dict[str, int] = None
    tokenizer: callable = None


def textToTokens(text: str) -> List[List[str]]:
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


def file_splitter(
    filename: str, seed: int = 0, train_prop: float = 0.7, dev_prop: float = 0.15
):
    """Splits the lines of a file into 3 output files."""

    import random

    rnd = random.Random(seed)
    basename = filename[:-4]
    train_file = open(basename + ".train.txt", "w")
    test_file = open(basename + ".test.txt", "w")
    dev_file = open(basename + ".dev.txt", "w")
    with open(filename, "r") as f:
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


def read_texts(
    tarfname: str, dname: str, tokenizer_kwargs: dict = None, min_freq: int = 3
) -> Data:
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    tkn_kwargs = dict(lowercase=False, stop_words=None)
    if tokenizer_kwargs is not None:
        tkn_kwargs.update(**tokenizer_kwargs)

    import tarfile

    tar = tarfile.open(tarfname, "r:gz", errors="replace")
    train_mem = tar.getmember(dname + ".train.txt")
    train_txt = tar.extractfile(train_mem).read().decode(errors="replace")
    test_mem = tar.getmember(dname + ".test.txt")
    test_txt = tar.extractfile(test_mem).read().decode(errors="replace")
    dev_mem = tar.getmember(dname + ".dev.txt")
    dev_txt = tar.extractfile(dev_mem).read().decode(errors="replace")

    from sklearn.feature_extraction.text import CountVectorizer

    count_vect = CountVectorizer(**tkn_kwargs)
    # Obtain term frequencies for training data
    tfreqs = count_vect.fit_transform(train_txt.split("\n"))
    tfreqs = np.array(tfreqs.sum(axis=0))[0]
    # Discard words that appear less than min_freq times
    vocab = {
        v: tf
        for v, tf in zip(count_vect.get_feature_names_out(), tfreqs)
        if tf >= min_freq
    }

    # Create vocab2idx: mapping between words and frequency-based
    # indexing, i.e., more frequent tokens are assigned lower ranks
    vocabulary = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    vocabulary, _ = zip(*vocabulary)

    # To apply the same mapping as the CountVectorizer, we need to apply
    # both preprocessor and tokenizer functions
    preproc = count_vect.build_preprocessor()
    tokeniz = count_vect.build_tokenizer()
    tokenizer = lambda txt: tokeniz(preproc(txt))

    data = Data([], [], [], vocabulary, tokenizer)
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    for s in dev_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.dev.append(toks)

    print(
        dname,
        " read. Num words:\n-> train:",
        len(data.train),
        "\n-> dev:",
        len(data.dev),
        "\n-> test:",
        len(data.test),
    )
    return data


def print_table(table, row_names, col_names, latex_file=None):
    """Pretty prints the table given the table, and row and col names.

    If a latex_file is provided (and tabulate is installed), it also writes a
    file containing the LaTeX source of the table (which you can \\input into your report)
    """
    try:
        from tabulate import tabulate

        rows = list(map(lambda rt: [rt[0]] + rt[1], zip(row_names, table.tolist())))

        # compute avg in domain perplexity and add to table
        avg_in_domain_ppl = np.mean(np.diagonal(table))
        rows = [row + ["-"] for row in rows]
        rows.append(["Avg In-Domain"] + ["-"] * len(rows) + [avg_in_domain_ppl])
        row_names.append("Avg In-Domain")

        print(tabulate(rows, headers=[""] + col_names))
        if latex_file is not None:
            latex_str = tabulate(rows, headers=[""] + col_names, tablefmt="latex")
            with open(latex_file, "w") as f:
                f.write(latex_str)
                f.close()
    except ImportError as e:
        row_format = "{:>15} " * (len(col_names) + 1)
        print(row_format.format("", *col_names))
        for row_name, row in zip(row_names, table):
            print(row_format.format(row_name, *row))
