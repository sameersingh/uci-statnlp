import sys; sys.path.append("../code")
import numpy as np

from ngram_interp import InterpNgram

VOCAB = ["A", "B", "C", "D"]
VOCAB_SIZE = 7
CORPUS = [
    ["A", "A", "B", "A", "C"],
    ["B", "A", "B", "A", "A", "E"],
    ["A", "E", "A", "A", "B", "A"]
]
CORPUS_SIZE = 23
BOS, UNK, EOS = InterpNgram.BOS_TOKEN, InterpNgram.UNK_TOKEN, InterpNgram.EOS_TOKEN


def assert_close_enough(res, exp, tol=1e-8):
    assert (res == -np.inf and exp == -np.inf) or (np.abs(res-exp) <= tol)


def test_interp_bigram_alpha_08_no_smoothing():
    model = InterpNgram(vocab2idx=VOCAB, ngram_size=2, llambda=0, alpha=0.8)

    corpus = model.preprocess_data(CORPUS)
    model.fit_corpus(corpus)

    # Tests backoff only
    assert_close_enough(model.cond_logprob("A", [EOS]), np.log(10/CORPUS_SIZE))
    assert_close_enough(model.cond_logprob("A", ["D"]), np.log(10/CORPUS_SIZE))

    # Tests interpolation
    assert_close_enough(model.cond_logprob("A", [BOS]), np.log(0.8 * 2/3 + 0.2 * 10/CORPUS_SIZE))
    # ^Note: np.log(alpha * p(a|bos) + (1-alpha) p(a))
    assert_close_enough(model.cond_logprob("A", ["B"]), np.log(0.8 + 0.2 * 10/CORPUS_SIZE))
    assert_close_enough(model.cond_logprob("C", ["A"]), np.log(0.8 * 1/10 + 0.2 * 1/CORPUS_SIZE))
    assert_close_enough(model.cond_logprob(UNK, ["A"]), np.log(0.8 * 2/10 + 0.2 * 2/CORPUS_SIZE))
    # Sequence "unk unk" was never observed during training
    assert_close_enough(model.cond_logprob(UNK, [UNK]), np.log(0.8 * 0 + 0.2 * 2/CORPUS_SIZE))
    assert_close_enough(model.cond_logprob(EOS, [UNK]), np.log(0.8 * 1/2 + 0.2 * 3/CORPUS_SIZE))
    assert_close_enough(model.cond_logprob("A", [UNK]), np.log(0.8 * 1/2 + 0.2 * 10/CORPUS_SIZE))

    # --------------------------------------------------------------------------------
    # Friendly note
    # --------------------------------------------------------------------------------
    # We will comment the line above because there are different ways your solution
    # it! It can either raise an exception or return the probability by replacing
    # "E" by "UNK" inside. In our case, we assume that the user will call preprocess
    # before calling model.cond_logprob and therefore this will never occur!
    # However, we incentivize you to raise an exception or have a safe guard mechanism
    # against it, as it will prevent bugs!!
    # assert_close_enough(model.cond_logprob("E", [UNK]), model.cond_logprob(UNK, [UNK]))


def test_interp_trigram_alpha_08_no_smoothing():
    model = InterpNgram(vocab2idx=VOCAB, ngram_size=3, llambda=0, alpha=0.8)

    corpus = model.preprocess_data(CORPUS)
    model.fit_corpus(corpus)

    # Tests backoff only (equivalent it should back off to unigram since
    # no lower-degree ngram has any of the conditioning terms in context)
    assert_close_enough(model.cond_logprob("A", [EOS]), np.log(10/CORPUS_SIZE))
    assert_close_enough(model.cond_logprob("A", ["D"]), np.log(10/CORPUS_SIZE))

    # Back off to bigram (SINCE "B B" was never observed in training))
    assert_close_enough(model.cond_logprob("A", ["B", "B"]), np.log(0.8 + 0.2 * 10/CORPUS_SIZE))
    assert_close_enough(model.cond_logprob("A", [UNK, UNK]), np.log(0.8 * 1/2 + 0.2 * 10/CORPUS_SIZE))

    # If neither trigram or bigram have seen the context, then it should be the unigram
    assert_close_enough(model.cond_logprob("A", ["A", "D"]), np.log(10/CORPUS_SIZE))

    # Let us go to the fun fun part! Interpolation
    assert_close_enough(model.cond_logprob("C", ["B", "A"]),
                        np.log(0.8 * 1/4 + 0.2 * (0.8 * 1/10 + 0.2 * 1/CORPUS_SIZE)))
    # ----------------------------------------------------------------------------
    # ^Explanation:
    # Let us drill down the expression above, using the handout's notation
    # ----------------------------------------------------------------------------
    # If we use I_n to represent the probability given by the Interpolated N-gram
    # and P_3 to represent the probability given by the standard trigram model, we
    # can define the probability given by an interpolated 3-gram model as:
    # I_3(C|BA) = alpha * P_3(C|BA) + (1-alpha) I_2(C|A)
    #           = alpha * P_3(C|BA) + (1-alpha) (alpha * P_2(C|A) + (1-alpha) P_1(C))
    # ----------------------------------------------------------------------------
    assert_close_enough(model.cond_logprob(EOS, ["A", UNK]),
                        np.log(0.8 * 1/2 + 0.2 * (0.8 * 1/2 + 0.2 * 3/CORPUS_SIZE)))

    assert_close_enough(model.cond_logprob("B", ["A", "A"]),
                        np.log(0.8 * 2/3 + 0.2 * (0.8 * 3/10 + 0.2 * 4/CORPUS_SIZE)))

    # We need smoothing :( or we still face the chances of having -np.inf
    # unfortunate, isn't it?
    assert_close_enough(model.cond_logprob("D", ["B", "A"]), -np.inf)


def test_interp_trigram_alpha_08_add_1_smoothing():
    model = InterpNgram(vocab2idx=VOCAB, ngram_size=3, llambda=1, alpha=0.8)

    corpus = model.preprocess_data(CORPUS)
    model.fit_corpus(corpus)

    # Tests backoff only (equivalent it should back off to unigram since
    # no lower-degree ngram has any of the conditioning terms in context)
    assert_close_enough(model.cond_logprob("A", [EOS]), np.log(11/(CORPUS_SIZE+VOCAB_SIZE)))
    assert_close_enough(model.cond_logprob("A", ["D"]), np.log(11/(CORPUS_SIZE+VOCAB_SIZE)))

    # Back off to bigram (SINCE "B B" was never observed in training))
    assert_close_enough(model.cond_logprob("A", ["B", "B"]), np.log(0.8 * 5/11 + 0.2 * 11/(CORPUS_SIZE+VOCAB_SIZE)))
    assert_close_enough(model.cond_logprob("A", [UNK, UNK]), np.log(0.8 * 2/9 + 0.2 * 11/(CORPUS_SIZE+VOCAB_SIZE)))

    # If neither trigram or bigram have seen the context, then it should be the unigram
    assert_close_enough(model.cond_logprob("A", ["A", "D"]), np.log(11/(CORPUS_SIZE+VOCAB_SIZE)))

    # Let us go to the fun fun part! Interpolation
    assert_close_enough(model.cond_logprob("C", ["B", "A"]),
                        np.log(0.8 * 2/11 + 0.2 * (0.8 * 2/17 + 0.2 * 2/(CORPUS_SIZE+VOCAB_SIZE))))
    # ----------------------------------------------------------------------------
    # ^Explanation:
    # Let us drill down the expression above, using the handout's notation
    # ----------------------------------------------------------------------------
    # If we use I_n to represent the probability given by the Interpolated N-gram
    # and P_3 to represent the probability given by the standard trigram model, we
    # can define the probability given by an interpolated 3-gram model as:
    # I_3(C|BA) = alpha * P_3(C|BA) + (1-alpha) I_2(C|A)
    #           = alpha * P_3(C|BA) + (1-alpha) (alpha * P_2(C|A) + (1-alpha) P_1(C))
    # ----------------------------------------------------------------------------
    assert_close_enough(model.cond_logprob(EOS, ["A", UNK]),
                        np.log(0.8 * 2/9 + 0.2 * (0.8 * 2/9 + 0.2 * 4/(CORPUS_SIZE+VOCAB_SIZE))))

    assert_close_enough(model.cond_logprob("B", ["A", "A"]),
                        np.log(0.8 * 3/10 + 0.2 * (0.8 * 4/17 + 0.2 * 5/(CORPUS_SIZE+VOCAB_SIZE))))

    # See how distributing a bit of the mass accross everything helps? :3
    assert_close_enough(model.cond_logprob("D", ["B", "A"]),
                        np.log(0.8 * 1/11 + 0.2 * (0.8 * 1/17 + 0.2 * 1/(CORPUS_SIZE+VOCAB_SIZE))))




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
    test_interp_bigram_alpha_08_no_smoothing()
    test_interp_trigram_alpha_08_no_smoothing()
    test_interp_trigram_alpha_08_add_1_smoothing()