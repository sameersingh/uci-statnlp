"""Language Modeling Interface

In many cases, the base implementation defaults to support
N-gram based language modeling.
"""
from typing import Dict, List

import numpy as np
import pickle
import tqdm

class LangModel:
    """Language modeling base class.

    The default implementation concerns parts of a simplified
    ngram implementation.

    Attributes
    ----------
    BOS_TOKEN: str
        Text descriptor used to mark the beginning of a sentence.

    EOS_TOKEN: str
        Text descriptor used to mark the end of a sentence.

    UNK_TOKEN: str
        Text descriptor used to represent the tokens that are out-of-vocabulary.

    Notes
    -----
    The use of a LangModel must follow a recipe for training:
    (1) Call LangModel.preprocess_data(corpus)
    (2) LangModel.fit_corpus(corpus)

    The use of LangModel also requires the preprocess_data
    method to be called before any of the inference methods is
    called, such as cond_logprob, logprob_sentence,
    cond_logprob_dist.
    """

    UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 0
    EOS_TOKEN, EOS_TOKEN_ID = "<eos>", 1
    BOS_TOKEN, BOS_TOKEN_ID = "<bos>", 2

    def __init__(self, vocab2idx: List[str]):
        self._word2id = {
            self.UNK_TOKEN: self.UNK_TOKEN_ID,
            self.EOS_TOKEN: self.EOS_TOKEN_ID,
            self.BOS_TOKEN: self.BOS_TOKEN_ID,
        }
        self._id2word = {
            self.UNK_TOKEN_ID: self.UNK_TOKEN,
            self.EOS_TOKEN_ID: self.EOS_TOKEN,
            self.BOS_TOKEN_ID: self.BOS_TOKEN,
        }

        for w in vocab2idx:
            n = len(self._word2id)
            self._word2id[w] = n
            self._id2word[n] = w

        self.is_ngram = True
        self._orig_vocab = vocab2idx # debugging purposes

    def _preprocess_data_extra(self, sentence: List[str]) -> list:
        """To be redefined by subclasses that need extra preprocessing."""
        return sentence

    @property
    def vocab(self) -> List[str]:
        """List of words supported by the language model.

        Notes
        -----
        The returned list will include the LangModel.UNK_TOKEN,
        LangModel.BOS_TOKEN, and LangModel.EOS_TOKEN, as well
        as the words that you specified during creation.
        """
        return list(self._word2id.keys())

    @property
    def vocab_size(self) -> int:
        """Vocabulary size including special tokens."""
        return len(self._word2id)

    def preprocess_data(self, corpus: List[List[str]], add_eos=True) -> list:
        """Formats the sequences and should be called prior to fit corpus
        or evaluating any sentence."""
        fmt_corpus = []

        for sentence in tqdm.tqdm(corpus, desc="Preprocessing data"):
            sentence = self.replace_unks(sentence)
            sentence = [self.BOS_TOKEN] + sentence
            if add_eos:
                sentence += [self.EOS_TOKEN]
            sentence = self._preprocess_data_extra(sentence)
            fmt_corpus.append(sentence)

        return fmt_corpus

    def fit_corpus(self, corpus: List[List[str]], **kwargs):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in tqdm.tqdm(corpus, desc="Num training sentences"):
            self.fit_sentence(s, **kwargs)

    def fit_sentence(self, sentence: List[str], **kwargs):
        """Parses a list of words."""
        pass

    def word2id(self, word: str) -> int:
        """Get the word index from the range [0, |V|].

        If the specified word does not exist, it returns
        LangModel.UNK_TOKEN_ID.
        """
        return self._word2id.get(word) or self.UNK_TOKEN_ID

    def id2word(self, word_id: int) -> str:
        """Map from index to vocabulary.

        Useful when dealing w/ vectorized representations of text.
        """
        return self._id2word[word_id]

    def is_word_oov(self, word: str) -> bool:
        """True if the word is out-of-vocabulary, false otherwise."""
        return self.word2id(word) == self.UNK_TOKEN_ID

    def replace_unks(self, words: List[str]) -> bool:
        """Replace the out-of-vocabulary words in ``words``."""
        result = []
        for w in words:
            if self.is_word_oov(w):
                result.append(self.UNK_TOKEN)
            else:
                result.append(w)
        return result

    def perplexity(self, corpus: List[str]) -> float:
        """Computes the perplexity (in nats) for the specified corpus."""
        return np.exp(self.entropy(corpus))

    def entropy(self, corpus: List[List[str]]) -> float:
        """Computes the entropy (in nats) over a given corpus."""
        num_words, sum_logprob = 0.0, 0.0
        for s in tqdm.tqdm(corpus, desc="[Entropy] Num sentences:"):
            num_words += len(s) - 1
            sum_logprob += self.logprob_sentence(s)
        return -(1.0 / num_words) * (sum_logprob)

    def logprob_sentence(self, sentence: List[str]) -> float:
        """Computes the unnormalized log probability of a sentence.

        Assumes that the provided sentence is already preprocessed
        (i.e., right format and type).
        """
        p = 0
        for i in range(1, len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        return p

    def cond_logprob_dist(self, previous: List[str]) -> np.ndarray:
        """Computes the natural log probability over the vocabulary,
        given previous words.

        Assumes that the previous is already preprocessed (i.e.,
        right format and type).
        """
        return np.array([self.cond_logprob(word, previous) for word in self.vocab])
        # ^Note: Efficiency could be improved by going over the
        # terms for which we had counts.

    def cond_logprob(self, word: str, previous: List[str]) -> float:
        """Computes the natural log conditional probability of word, given previous words."""
        raise NotImplementedError("Please override in subclass")

    def save_model(self, filepath: str):
        """Persist the current model to the specified filepath."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath: str, **kwargs) -> "LangModel":
        """Load a model from the specified filepath."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def decode(self, sentence_ids: List[int]) -> List[str]:
        """Decodes a list of indices into text"""
        return [self.id2word(sid) for sid in np.array(sentence_ids).tolist()]
