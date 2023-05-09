"""Python script that trains and evaluates ngram models.

Methods
-------
parse_args() --> argparse.Args
    Defines the command line arguments necessary to run the script.

learn_ngram(data, n, min_freq) --> ngram.Ngram
    Fits a ngram model of size n to the specified data. It will treat
    every word that appears less than min_freq as Out-of-Vocabulary.
"""
from time import time
from typing import Any, Dict, List, Union

# User imports
from data import Data, read_texts
from utils import DATASETS, MIN_FREQ_DEFAULT, PREFIXES, evaluate_perplexity, print_sep, sample
from ngram import Ngram
from ngram_interp import InterpNgram

import argparse, os


BASE_DIR = ".."


def parse_args():
    # Usage example
    # $ python -m learn_ngram --use_interp --ngram_size 4 --min_freq 2 --alpha 0.8 --lambda 1
    # Explaining: Running the model using the above command will fit the
    # InterpNgram model using add-1 smoothing and alpha=0.8
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default=f"{BASE_DIR}/data/corpora.tar.gz",
        type=str,
        help="Path to the tar.gz file with the datasets.",
    )
    parser.add_argument(
        "--output_dir",
        default=f"{BASE_DIR}/results/ngram",
        help="name of directory to write out trained language models.",
        type=str,
    )
    parser.add_argument(
        "--use_interp",
        action="store_true",
        help="use this flag to use the interpolated ngram model version.",
    )
    parser.add_argument(
        "--eval",
        default=True,
        type=bool,
        help="use this flag to evaluate the trained models as well.",
    )
    parser.add_argument(
        "--ngram_size",
        default=3,
        help="Size of the ngram model to train.",
        type=int,
    )
    parser.add_argument(
        "--alpha",
        default=0.8,
        help="Alpha coefficient for the InterpNgram.",
        type=float,
    )
    parser.add_argument(
        "--llambda",
        default=0.2,
        help="Smoothing parameter for Ngram model. Should be non-negative.",
        type=float,
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=MIN_FREQ_DEFAULT,
        help="Mininum number of times a token should appear in"
        "the training set to be considered part of vocabulary.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="*",
        help="Specifies that datasets to train models for.",
    )
    args = parser.parse_args()

    # Create output dir
    print("Creating results directory:", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Argument verification
    assert args.ngram_size > 0, "'ngram_size' must be positive"
    assert args.min_freq > 0, "'min_freq' must be positive"
    assert args.llambda >= 0, "'lambda' must be non-negative"
    assert (
        0 < args.alpha < 1
    ), "Interpolation parameter 'alpha' must be in the range (0, 1)"

    if args.datasets == "*":
        args.datasets = DATASETS
    else:
        assert (
            args.datasets in DATASETS
        ), f"specified dataset must be one of: {DATASETS}"
        args.datasets = [args.datasets]

    print_sep(f"\n[Experiment Config]:\n {args}")
    return args


def learn_ngram_model(data: Data, ngram_model: Union[Ngram, InterpNgram]):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    print("vocab:", ngram_model.vocab_size)

    train_data = ngram_model.preprocess_data(data.train)
    print("Fitting training data...")
    ngram_model.fit_corpus(train_data)

    # -------------------------------------------------------
    # evaluate on train, test, and dev (in-domain evaluation)
    # -------------------------------------------------------
    print_sep("In domain Perplexities")
    ppl_train = ngram_model.perplexity(train_data)
    dev_data = ngram_model.preprocess_data(data.dev)
    ppl_dev = ngram_model.perplexity(dev_data)
    test_data = ngram_model.preprocess_data(data.test)
    ppl_test = ngram_model.perplexity(test_data)
    print("[PPL train]:", ppl_train)
    print("[PPL dev]  :", ppl_dev)
    print("[PPL test] :", ppl_test)


if __name__ == "__main__":
    args = parse_args()

    # List of individual corpus and corresponding models
    datas: List[Data] = []
    models: List[Ngram] = []

    # Learn the models for each of the corpus, and evaluate them in-domain
    for dname in args.datasets:
        print_sep(f"Training {dname}")
        data = read_texts(args.dataset_path, dname, tokenizer_kwargs={"lowercase": False}, min_freq=args.min_freq)
        datas.append(data)

        model_kwargs = dict(ngram_size=args.ngram_size, llambda=args.llambda)
        if args.use_interp:
            model_kwargs.update(alpha=args.alpha)
            ngram_model = InterpNgram(vocab2idx=data.vocabulary, **model_kwargs)
        else:
            ngram_model = Ngram(vocab2idx=data.vocabulary, **model_kwargs)

        start = time()
        learn_ngram_model(data, ngram_model)
        end = time()
        print(f"Training duration (min): {(end-start)/60:.2}")

        print_sep(f"Generating samples")
        results = sample(ngram_model, prefixes=PREFIXES, max_new_tokens=5)
        model_filepath = f"{args.output_dir}/{dname}__{ngram_model.name}.pkl"
        print("Persisting model at", model_filepath)
        ngram_model.save_model(model_filepath)
        models.append(ngram_model)

    if args.eval:
        # Note: use the flag --eval when running this script
        # if you'd like to conduct in-domain/out-of-domain perplexity evaluation
        print_sep("Evaluate")
        start = time()
        evaluate_perplexity(args.datasets, datas, models, args.output_dir)
        end = time()
        print(f"Evaluation duration (min): {(end-start)/60:.2}")

    print("Done!")
