import sys; sys.path.append("../code")
import numpy as np

from ngram import Ngram

VOCAB = ["A", "B", "C", "D"]
CORPUS = [["A", "A", "B", "A", "C"]]
BOS, UNK, EOS = Ngram.BOS_TOKEN, Ngram.UNK_TOKEN, Ngram.EOS_TOKEN


def assert_close_enough(res, exp, tol=1e-8):
    assert (res == -np.inf and exp == -np.inf) or (np.abs(res-exp) <= tol)

def test_unigram_no_smoothing():
    model = Ngram(vocab2idx=VOCAB, ngram_size=1, llambda=0)

    corpus = model.preprocess_data(CORPUS)
    model.fit_corpus(corpus)

    assert model.counts_totals[tuple()] == model.unigram_total
    assert sum(model.counts[tuple()].values()) == sum(model.unigram_counts.values())
    assert_close_enough(model.cond_logprob("A", []), np.log(3/7))
    assert_close_enough(model.cond_logprob("B", []), np.log(1/7))
    assert_close_enough(model.cond_logprob("C", []), np.log(1/7))
    assert_close_enough(model.cond_logprob(BOS, []), np.log(1/7))
    assert_close_enough(model.cond_logprob(EOS, []), np.log(1/7))
    assert_close_enough(model.cond_logprob(UNK, []), -np.inf)
    assert_close_enough(model.cond_logprob("D", []), -np.inf)


def test_unigram_add_1_smoothing():
    model = Ngram(vocab2idx=VOCAB, ngram_size=1, llambda=1)

    corpus = model.preprocess_data(CORPUS)
    model.fit_corpus(corpus)

    assert model.counts_totals[tuple()] == model.unigram_total
    assert sum(model.counts[tuple()].values()) == sum(model.unigram_counts.values())
    assert_close_enough(model.cond_logprob("A", []), np.log(4/14))
    assert_close_enough(model.cond_logprob("B", []), np.log(2/14))
    assert_close_enough(model.cond_logprob("C", []), np.log(2/14))
    assert_close_enough(model.cond_logprob(BOS, []), np.log(2/14))
    assert_close_enough(model.cond_logprob(EOS, []), np.log(2/14))
    assert_close_enough(model.cond_logprob(UNK, []), np.log(1/14))
    assert_close_enough(model.cond_logprob("D", []), np.log(1/14))


def test_bigram_no_smoothing():

    model = Ngram(vocab2idx=VOCAB, ngram_size=2, llambda=0)

    corpus = model.preprocess_data(CORPUS)
    model.fit_corpus(corpus)

    assert_close_enough(model.cond_logprob("A", [BOS]), 0)
    assert_close_enough(model.cond_logprob("A", ["A"]), np.log(1/3))
    assert_close_enough(model.cond_logprob("B", ["A", "A"]), np.log(1/3))
    assert_close_enough(model.cond_logprob("C", ["A", "A"]), np.log(1/3))

    assert_close_enough(model.cond_logprob(EOS, ["C"]), 0)
    assert_close_enough(model.cond_logprob("E", ["A"]), -np.inf)

    assert_close_enough(model.cond_logprob("B", ["A", "B"]), -np.inf) # b never followed b during training
    assert_close_enough(model.cond_logprob("B", ["A", "B"]), -np.inf) # b never followed b during training

    assert_close_enough(model.cond_logprob(UNK, ["C"]), -np.inf)
    assert_close_enough(model.cond_logprob("D", [BOS]), -np.inf)
    assert_close_enough(model.cond_logprob("C", [EOS]), np.log(1/7)) # backoff to unigram


def test_bigram_add_1_smoothing():
    model = Ngram(vocab2idx=VOCAB, ngram_size=2, llambda=1)
    corpus = model.preprocess_data(CORPUS)
    model.fit_corpus(corpus)
    assert_close_enough(model.cond_logprob("A", [BOS]), np.log(2/8))
    assert_close_enough(model.cond_logprob("B", [BOS]), np.log(1/8))
    assert_close_enough(model.cond_logprob("C", [BOS]), np.log(1/8))
    assert_close_enough(model.cond_logprob("D", [BOS]), np.log(1/8))
    assert_close_enough(model.cond_logprob(BOS, [BOS]), np.log(1/8))
    assert_close_enough(model.cond_logprob(UNK, [BOS]), np.log(1/8))
    assert_close_enough(model.cond_logprob(EOS, [BOS]), np.log(1/8))

    assert_close_enough(model.cond_logprob("A", [BOS, "A"]), np.log(2/10))
    assert_close_enough(model.cond_logprob("B", [BOS, "A"]), np.log(2/10))
    assert_close_enough(model.cond_logprob("C", [BOS, "A"]), np.log(2/10))
    assert_close_enough(model.cond_logprob("D", [BOS, "A"]), np.log(1/10))
    assert_close_enough(model.cond_logprob(BOS, [BOS, "A"]), np.log(1/10))
    assert_close_enough(model.cond_logprob(UNK, [BOS, "A"]), np.log(1/10))
    assert_close_enough(model.cond_logprob(EOS, [BOS, "A"]), np.log(1/10))

    assert_close_enough(model.cond_logprob("A", ["C"]), np.log(1/8))
    assert_close_enough(model.cond_logprob("B", ["C"]), np.log(1/8))
    assert_close_enough(model.cond_logprob("C", ["C"]), np.log(1/8))
    assert_close_enough(model.cond_logprob("D", ["C"]), np.log(1/8))
    assert_close_enough(model.cond_logprob(BOS, ["C"]), np.log(1/8))
    assert_close_enough(model.cond_logprob(UNK, ["C"]), np.log(1/8))
    assert_close_enough(model.cond_logprob(EOS, ["C"]), np.log(2/8))

    # Back off to unigram (also w/ smoothing for cases where D is part
    # of vocabulary but was not observed during dtraining)
    assert_close_enough(model.cond_logprob("A", ["D"]), np.log(4/14))
    assert_close_enough(model.cond_logprob("B", ["D"]), np.log(2/14))
    assert_close_enough(model.cond_logprob("C", ["D"]), np.log(2/14))
    assert_close_enough(model.cond_logprob("D", ["D"]), np.log(1/14))
    assert_close_enough(model.cond_logprob(BOS, ["D"]), np.log(2/14))
    assert_close_enough(model.cond_logprob(UNK, ["D"]), np.log(1/14))
    assert_close_enough(model.cond_logprob(EOS, ["D"]), np.log(2/14))


def test_trigram_no_smoothing():
    corpus = [["A", "A", "B", "A", "C"],
              ["A", "A", UNK, "A", UNK, "A"]]
    model = Ngram(vocab2idx=VOCAB, ngram_size=3, llambda=0)
    corpus = model.preprocess_data(corpus)
    model.fit_corpus(corpus)

    # Make sure counts for unigram backoff are correct
    assert_close_enough(model.unigram_counts.get("A"), 7)
    assert_close_enough(model.unigram_counts.get("B"), 1)
    assert_close_enough(model.unigram_counts.get("C"), 1)
    assert model.unigram_counts.get("D") is None
    assert_close_enough(model.unigram_counts.get(UNK), 2)
    assert_close_enough(model.unigram_counts.get(BOS), 2)
    assert_close_enough(model.unigram_counts.get(EOS), 2)
    assert_close_enough(model.unigram_total, 15)

    # Ensure some trigram probabilities are correct
    assert_close_enough(model.cond_logprob("A", [BOS]), 0)
    assert_close_enough(model.cond_logprob("A", [BOS, "A"]), 0)
    assert_close_enough(model.cond_logprob(UNK, ["A", "A"]), np.log(1/2))

    # UNK in conditioning term
    assert_close_enough(model.cond_logprob(EOS, [UNK, "A"]), np.log(1/2))
    assert_close_enough(model.cond_logprob(UNK, [UNK, "A"]), np.log(1/2))

    assert_close_enough(model.cond_logprob("B", [BOS, "A"]), -np.inf)
    assert_close_enough(model.cond_logprob("A", ["A", "A"]), -np.inf)
    assert_close_enough(model.cond_logprob("B", ["A", "A"]), np.log(1/2))
    assert_close_enough(model.cond_logprob("A", ["A", UNK]), 0)
    # context and word have been observed but nt sequentially
    assert_close_enough(model.cond_logprob("C", ["A", "B"]), -np.inf)
    assert_close_enough(model.cond_logprob("C", ["A", "A"]), -np.inf)
    # backoff since context is never observed
    assert_close_enough(model.cond_logprob("A", ["B", "B"]), np.log(7/15))
    assert_close_enough(model.cond_logprob("C", ["C", "C"]), np.log(1/15))
    assert_close_enough(model.cond_logprob("C", [UNK, UNK]), np.log(1/15))
    assert_close_enough(model.cond_logprob("B", [UNK, "C"]), np.log(1/15))


def test_trigram_add_1_smoothing():
    corpus = [["A", "A", "B", "A", "C"],
              ["A", "A", UNK, "A", UNK, "A"]]
    model = Ngram(vocab2idx=VOCAB, ngram_size=3, llambda=1)
    corpus = model.preprocess_data(corpus)
    model.fit_corpus(corpus)

    # Make sure counts for unigram backoff are correct
    assert_close_enough(model.unigram_counts.get("A"), 7)
    assert_close_enough(model.unigram_counts.get("B"), 1)
    assert_close_enough(model.unigram_counts.get("C"), 1)
    assert model.unigram_counts.get("D") is None
    assert_close_enough(model.unigram_counts.get(UNK), 2)
    assert_close_enough(model.unigram_counts.get(BOS), 2)
    assert_close_enough(model.unigram_counts.get(EOS), 2)
    assert_close_enough(model.unigram_total, 15)

    # Ensure some trigram probabilities are correct
    assert_close_enough(model.cond_logprob("A", [BOS]), np.log(3/9))
    assert_close_enough(model.cond_logprob("A", [BOS, "A"]), np.log(3/9))

    # UNK in conditioning term
    assert_close_enough(model.cond_logprob(EOS, [UNK, "A"]), np.log(2/9))
    assert_close_enough(model.cond_logprob(UNK, [UNK, "A"]), np.log(2/9))

    assert_close_enough(model.cond_logprob("A", ["A", UNK]), np.log(3/9))

    assert_close_enough(model.cond_logprob("A", [BOS, "A"]), np.log(3/9))
    assert_close_enough(model.cond_logprob("B", [BOS, "A"]), np.log(1/9))
    # context and word have been observed but nt sequentially
    assert_close_enough(model.cond_logprob("A", ["A", "A"]), np.log(1/9))
    assert_close_enough(model.cond_logprob("B", ["A", "A"]), np.log(2/9))
    assert_close_enough(model.cond_logprob("C", ["A", "A"]), np.log(1/9))
    assert_close_enough(model.cond_logprob("D", ["A", "A"]), np.log(1/9))
    assert_close_enough(model.cond_logprob(UNK, ["A", "A"]), np.log(2/9))
    assert_close_enough(model.cond_logprob(EOS, ["A", "A"]), np.log(1/9))
    assert_close_enough(model.cond_logprob(BOS, ["A", "A"]), np.log(1/9))
    # backoff since context is never observed
    assert_close_enough(model.cond_logprob("A", ["B", "B"]), np.log(8/22))
    assert_close_enough(model.cond_logprob("C", ["C", "C"]), np.log(2/22))
    assert_close_enough(model.cond_logprob("C", [UNK, UNK]), np.log(2/22))
    assert_close_enough(model.cond_logprob("B", [UNK, "C"]), np.log(2/22))
    assert_close_enough(model.cond_logprob("D", [UNK, "C"]), np.log(1/22))

    # backoff since A is never observed alone in a trigram model
    assert_close_enough(model.cond_logprob(EOS, ["A"]), np.log(3/22))
    # however this one is no longer backoff (but smoothing instead)
    assert_close_enough(model.cond_logprob(EOS, [BOS]), np.log(1/9))


if __name__ == "__main__":
    # ----------------------------------------------------------
    # You can execute this script in one of two ways:
    #
    # 1. You use Python command: python -m test_ngram_interp
    # The file should execute with no errors. If an assertion
    # error is detected then, you may have a bug in your
    # implementation.
    #
    # 2. You use pytest and type down in "pytest" in the terminal
    # This will tell you how many tests you failed and how many
    # you passed, as well as provide you some details on which
    # line failed and why.
    # ----------------------------------------------------------
    # Both approaches work fairly well, I'd say the advantage of
    # number 2 is that you don't have to list all the test methods
    # in the main (you are less prone to forget a test).
    # Pytest will automatically execute every method in the files
    # whose name starts with "test_" for method names starting with
    # "test_".
    # ----------------------------------------------------------
    test_unigram_no_smoothing()
    test_unigram_add_1_smoothing()
    test_bigram_no_smoothing()
    test_bigram_add_1_smoothing()
    test_trigram_no_smoothing()
    test_trigram_add_1_smoothing