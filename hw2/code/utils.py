"""Script utils

Constants
---------
DATASETS: List[str]


Methods
-------
evaluate_perplexity(data_names, datas, models):
    Given the list of models and the list of datasets computes the
    in-domain and out-of-domain perplexity of the specified models.

sample(model, temp, prefix) -> List[str]:
    Samples a few sequences from the model distribution.
    Temp is the temperature (lower leads to peakier distributions
    whereas higher leads to more uniform distribution). Prefix
    is the prompt to the model that guides generation.
"""
from typing import List

# User imports
from data import Data, print_table
from decoders import generate_sentence, DECODERS
from lm import LangModel

import os
import numpy as np

DATASETS = ["brown", "reuters", "gutenberg"]
MIN_FREQ_DEFAULT = 2
PREFIXES = [
    "",
    "United States of",
    "They danced", # brown
    "It said the government", # reuters
    "and the lord", "Harriet was not", # gutenberg

]


def evaluate_perplexity(
    dnames: List[str], datas: List[Data], models: List[LangModel], output_dir: str
):
    print(f"Evaluating {len(dnames)} datasets")
    # compute the perplexity of all pairs
    n = len(dnames)
    perp_dev = np.zeros((n, n))
    perp_test = np.zeros((n, n))
    perp_train = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            print(f"Processing dataset {dnames[j]} with model trained on {dnames[i]}...")
            dev_j = models[i].preprocess_data(datas[j].dev)
            test_j = models[i].preprocess_data(datas[j].test)
            train_j = models[i].preprocess_data(datas[j].train)
            perp_dev[i][j] = models[i].perplexity(dev_j)
            perp_test[i][j] = models[i].perplexity(test_j)
            perp_train[i][j] = models[i].perplexity(train_j)

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


def sample(
    model: LangModel,
    prefixes: List[str] = None,
    max_new_tokens: int = 10,
    decoder: DECODERS = DECODERS.GREEDY,
    **kwargs,
) -> List[str]:
    """Sample `max_new_tokens` from the model distribution given
    the prefixes and using the specified decoder algorithm.

    By default it uses the greedy decoding.
    """
    if prefixes is None:
        prefixes = [""]
    elif isinstance(prefixes, str):
        prefixes = [prefixes]

    # Obtain the preprocessed prefixes
    prefixes = [p.split() for p in prefixes]
    prefixes_dec_ids = model.preprocess_data(prefixes, add_eos=False)

    outputs = []
    for prefix, prefix_dec_ids in zip(prefixes, prefixes_dec_ids):
        # ngrams preprocessing of the data is done in terms of the words
        # however for decoding, we will deal with the vectorized representation
        # and therefore need to encode each word into their indices
        if model.is_ngram:
            prefix_dec_ids = [model.word2id(w) for w in prefix_dec_ids]

        out = generate_sentence(
            model=model,
            decoder=decoder,
            decoded_ids=prefix_dec_ids,
            max_length=len(prefix) + max_new_tokens,
            **kwargs,
        )
        out["prefix"], out["max_new_tokens"]  = prefix, max_new_tokens
        outputs.append(out)

    for output in outputs:
        print("-" * 60)
        print(output)

    return outputs


def print_sep(msg):
    print()
    print("=" * 80)
    print(msg)
    print("=" * 80)