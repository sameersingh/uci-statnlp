"""Decoding utilities

Types
-----
DECODERS(enum)
    Enumerate file specifying the types of decoding algorithms.

Candidate
    A class containing information for a single candidate generation.
    Information concerns the (last decoded ids, scores, etc)
"""
from lm import LangModel
from typing import Iterable, List
from enum import Enum, auto

import copy
import random
import numpy as np
import torch


class DECODERS(Enum):
    GREEDY = auto()
    TOP_K = auto()
    NUCLEUS = auto()
    CONSTRAINED = auto()
    CONSTRAINED_NO_REP = auto()
    MULTINOMIAL = auto()


class Candidate:
    """A class containing information for a single candidate generation.

    Attributes:
    ----------
    log_prob: ``float``
        The score of the current generation. More specifically, this is
        the log-probability joint probability of the current generation.

    last_id_prob: ``float``
        The probability of generating the most recently decoded ID. This
        is not in log-space. This is useful for decoding strategies
        like top k and nucleus sampling.

    last_decoded_id: ``int``
        The ID of the most recently decoded ID.

    decoded_ids: ``list``
        A list of all generated IDs in this `Candidate`
    """

    def __init__(
        self, score: float = 0, last_id_prob: float = 0, decoded_ids: List[int] = []
    ):
        self.log_prob = score
        self.last_id_prob = last_id_prob
        self.decoded_ids = copy.deepcopy(decoded_ids)

    def __len__(self):
        return len(self.decoded_ids)

    @property
    def last_decoded_id(self):
        return self.decoded_ids[-1] if len(self.decoded_ids) else None

    def concat_id(self, idx: int) -> Iterable:
        """Abstraction to concatenate past decoded ids with idx."""
        if isinstance(self.decoded_ids, list):
            return self.decoded_ids + [idx]
        elif isinstance(self.decoded_ids, np.ndarray):
            return np.concatenate(self.decoded_ids.flatten(), [idx])
        elif isinstance(self.decoded_ids, torch.Tensor):
            concat_args = (self.decoded_ids, torch.tensor([idx]))
            return torch.concat((concat_args))
        else:
            raise RuntimeError(f"Cannot concat unknown type: {type(self.decoded_ids)}")

    def get_next_cands(self, model: LangModel) -> List["Candidate"]:
        """Returns a list of `Candidate` objects which are all possible
        continuations of the current `Candidate` object. The continuation
        IDs and scores come from calling `model`.

        Parameters
        ----------
        model: ``LangModel``
            A light-weight wrapper around an object which takes as input a
            list of the previously decoded IDs as well as any metadata, and
            returns scores for all possible continuations and updated
            metadata. See `lm.py` for more details on this class.

        Returns
        -------
        cands: ``list[Candidate]``
            Returns a list of `Candidate` objects which are all possible
            continuations of the current `Candidate`.
        """
        # Feed previous decoded ids through the model
        if model.is_ngram:
            decoded_text = model.decode(self.decoded_ids)
            next_id_logprobs = model.cond_logprob_dist(decoded_text)
        else:
            next_id_logprobs = model.cond_logprob_dist(self.decoded_ids)
        # ^Note: next_id_logprobs consists of the log probability
        # of each of the model.vocab_size tokens.
        assert isinstance(next_id_logprobs, (np.ndarray, list))

        # Generate candidates for every possible continuation of current candidate
        next_candidates = [
            Candidate(
                score=self.log_prob + id_logprob,
                last_id_prob=np.exp(id_logprob),
                decoded_ids=self.concat_id(idx),
            )
            for idx, id_logprob in enumerate(next_id_logprobs)
        ]

        return next_candidates

    def __str__(self):
        return f"Candidate(logprob={self.log_prob}, decoded_ids={self.decoded_ids})"


def is_cand_finished(cand: Candidate, max_length: int, eos_id: int) -> bool:
    """A candidate is finished generating if the number of decoded IDs of
    the candidate is at the max length or if the EOS ID has been generated
    """
    if len(cand) >= max_length or cand.last_decoded_id == eos_id:
        return True
    else:
        return False


def greedy_decoding(
    model: LangModel,
    max_length: int = 50,
    decoded_ids: Iterable[int] = None,
):
    """
    Parameters
    ----------
    model: ``Model``
        Used to get continuation probabilities
    max_length: ``int``
        Maximum allowed length of decoding
    decoded_ids: ``Iterable[int]``
        List of decoded IDs to start the generation

    Returns
    -------
    cand: ``Candidate`` The highest scoring candidate at the end of the search
    """
    decoded_ids = [] if decoded_ids is None else decoded_ids
    cand = Candidate(decoded_ids=decoded_ids)
    eos_id = model.EOS_TOKEN_ID

    # Continue decoding while the number of decoded IDs of the candidate is
    # less than the max length and while the EOS ID has not been generated.
    while not is_cand_finished(cand, max_length, eos_id):
        # Get all possible continuation candidates
        potential_cands = cand.get_next_cands(model)

        # Sort list of candidates by their respective scores
        potential_cands = sorted(
            potential_cands, key=lambda x: x.log_prob, reverse=True
        )

        # The next candidate is the best scoring potential candidate
        cand = potential_cands[0]

    return cand


def multinomial_sampling(
    model: LangModel,
    temperature: float = 1.0,
    max_length: int = 50,
    decoded_ids: Iterable[int] = None,
):
    """Multinomial sampling decoding algorithm

    Parameters
    ----------
    model: ``Model``
        Used to get continuation probabilities
    max_length: ``int``
        Maximum allowed length of decoding
    decoded_ids: ``Iterable[int]``
        List of decoded IDs to start the generation

    Returns
    -------
    cand: ``Candidate``
        The candidate at the end of top-k sampling
    """
    decoded_ids = [] if decoded_ids is None else decoded_ids
    cand = Candidate(decoded_ids=decoded_ids)
    eos_id = model.EOS_TOKEN_ID

    while not is_cand_finished(cand, max_length, eos_id):
        # Get possible continuation candidates
        potential_cands = cand.get_next_cands(model)

        # Sort continuations by their last decoded ID probabilities
        potential_cands = sorted(
            potential_cands, key=lambda x: x.last_id_prob, reverse=True
        )

        # Get probabilities for all the last decoded IDs
        # (note: last_id_prob is in probability space)
        last_id_probs = np.array([cand.last_id_prob for cand in potential_cands])
        last_id_logprobs = np.array([np.log(prob) / temperature for prob in last_id_probs])
        last_id_probs = np.exp(last_id_logprobs) / np.exp(last_id_logprobs).sum()

        # ---------------------------------------------------------------------
        # Sample a candidate based on the probability of it's last ID
        # random.choices() automatically re-weights the probabilities!
        cand = random.choices(potential_cands, weights=last_id_probs)[0]

    return cand


def top_k_sampling(
    model: LangModel,
    top_k: float,
    temperature: float = 1,
    max_length: int = 50,
    decoded_ids: Iterable[int] = None,
):
    """Top-k sampling decoding algorithm

    Parameters
    ----------
    model: ``Model``
        Used to get continuation probabilities
    top_k: ``int``
        Filters for the `top_k` candidates whose before sampling
    max_length: ``int``
        Maximum allowed length of decoding
    decoded_ids: ``Iterable[int]``
        List of decoded IDs to start the generation

    Returns
    -------
    cand: ``Candidate``
        The candidate at the end of top-k sampling
    """
    decoded_ids = [] if decoded_ids is None else decoded_ids

    cand = Candidate(decoded_ids=decoded_ids)
    eos_id = model.EOS_TOKEN_ID

    while not is_cand_finished(cand, max_length, eos_id):
        # Get possible continuation candidates
        potential_cands = cand.get_next_cands(model)

        # Sort continuations by their last decoded ID probabilities
        potential_cands = sorted(
            potential_cands, key=lambda x: x.last_id_prob, reverse=True
        )

        # Get probabilities for all the last decoded IDs
        last_id_probs = [cand.last_id_prob for cand in potential_cands]

        # ---------------------------------------------------------------------
        # TODO: finish implementing this part of the while loop to complete
        # ---------------------------------------------------------------------
        # Top-k decoding algorithm. To do this, you'll have to:
        #   * obtain the top-k ``potential_cands`` and ``last_id_probs``
        #   * Scale `last_id_probs` via temperature-scaling via the
        #  ``temperature`` parameter
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO: Implement by student")

        # ---------------------------------------------------------------------
        # Sample a candidate based on the probability of it's last ID
        # random.choices() automatically re-weights the probabilities!
        cand = random.choices(potential_cands, weights=last_id_probs)[0]

    return cand


def nucleus_sampling(
    model: LangModel,
    top_p: float,
    max_length: int = 50,
    decoded_ids: Iterable[int] = None,
):
    """
    Parameters
    ----------
    model: ``Model``
        Used to get continuation probabilities
    top_p: ``float``
        Filters for the smallest possible set of candidates whose
        cumulative probability exceeds `top_p` before sampling.
    max_length: ``int``
        Maximum allowed length of decoding
    decoded_ids: ``Iterable[int]``
        List of decoded IDs to start the generation

    Returns
    -------
    cand: ``Candidate`` The candidate at the end of nucleus sampling
    """
    decoded_ids = [] if decoded_ids is None else decoded_ids

    cand = Candidate(decoded_ids=decoded_ids)
    eos_id = model.EOS_TOKEN_ID

    # Continue decoding while the number of decoded IDs of the candidate is
    # less than the max length and while the EOS ID has not been generated.
    while not is_cand_finished(cand, max_length, eos_id):
        # Get possible continuation candidates
        potential_cands = cand.get_next_cands(model)

        # Sort continuations by their last decoded ID probabilities
        potential_cands = sorted(
            potential_cands, key=lambda x: x.last_id_prob, reverse=True
        )

        # Get probabilities for all the last decoded IDs
        last_id_probs = [cand.last_id_prob for cand in potential_cands]

        # ---------------------------------------------------------------------
        # TODO: finish implementing this part of the while loop to complete
        # ---------------------------------------------------------------------
        #  nucleus sampling. To do this you will have to:
        #  * Find where the cutoff point is that satisfies the `top_p`
        #    threshold (the cands and associated probabilities are already
        #    sorted so this should be fairly straightforward)
        #  * Update `potential_cands` and `last_id_probs` by truncating
        #    everything past this cutoff point.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO: Implement by student")

        # ---------------------------------------------------------------------
        #  We have implemented the part of sampling the next candidate given
        #  the truncated `potential_cands` and `last_id_probs` so you
        #  shouldn't touch this part of the code.
        cand = random.choices(potential_cands, weights=last_id_probs)[0]

    return cand


def constrained_decoding(
    model: LangModel,
    constraints_list: Iterable[str],
    max_length: int = 50,
    decoded_ids: Iterable[int] = [],
):
    """
    Parameters
    ----------
    model: ``Model``
        Used to get continuation probabilities.

    constraints_list: List[str]
        List of words that cannot be sampled during decoding.

    max_length: ``int``
        Maximum allowed length of decoding

    decoded_ids: ``list[int]``
        List of decoded IDs to start the generation

    Returns
    -------
    cand: ``Candidate`` The highest scoring candidate at the end of the search
    """
    cand = Candidate(decoded_ids=decoded_ids)
    eos_id = model.EOS_TOKEN_ID

    # Continue decoding while the number of decoded IDs of the candidate is
    # less than the max length and while the EOS ID has not been generated.
    while not is_cand_finished(cand, max_length, eos_id):
        # ---------------------------------------------------------------------
        # TODO: finish implementing this part of the while loop to complete
        # ---------------------------------------------------------------------
        #  constrained decoding sampling. To do this you will have to:
        #   * Obtain the log probability distribution of the next words given
        #     the decoded_ids. (Hint: use get_next_cands)
        #   * Obtain the index/dimension for each word in ``constraints_list``
        #     (Hint: use model.word2id)
        #   * Remove candidate generations that concern words in the
        #     ``constraints_list``. In this case, it will be indices.
        #     (Hint: filter out the candidate generations obtained in 1)
        #   * Sort by the log probability of the remaining options and name it
        #     potential_cands.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO: Implement by student")

        # ---------------------------------------------------------------------
        #  We have implemented the part of sampling the next candidate given
        #  the truncated `potential_cands` and `last_id_probs` so you
        #  shouldn't touch this part of the code.
        cand = random.choices(potential_cands, weights=last_id_probs)[0]
    return cand


def constrained_decoding_no_repetition(
    model: LangModel,
    max_length: int = 50,
    decoded_ids: Iterable[int] = None,
):
    """
    Parameters
    ----------
    model: ``Model``
        Used to get continuation probabilities.

    max_length: ``int``
        Maximum allowed length of decoding

    decoded_ids: ``list[int]``
        List of decoded IDs to start the generation

    Returns
    -------
    cand: ``Candidate`` The highest scoring candidate at the end of the search
    """
    decoded_ids = [] if decoded_ids is None else decoded_ids
    cand = Candidate(decoded_ids=decoded_ids)
    eos_id = model.EOS_TOKEN_ID

    # Continue decoding while the number of decoded IDs of the candidate is
    # less than the max length and while the EOS ID has not been generated.
    while not is_cand_finished(cand, max_length, eos_id):
        # ---------------------------------------------------------------------
        # TODO: finish implementing this part of the while loop to complete
        # ---------------------------------------------------------------------
        #  constrained decoding sampling. To do this you will have to:
        #   * Obtain the log probability distribution of the next words given
        #     the decoded_ids. (Hint: use get_next_cands)
        #   * Use the previously decoded_ids as the ``constraints_list``
        #     (from the previous method)
        #   * Remove candidate generations that concern words in this
        #     ``constraints_list``. In this case, it will be indices.
        #     (Hint: filter out the candidate generations obtained in 1)
        #   * Sort by the log probability of the remaining options and name it
        #     potential_cands.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO: Implement by student")

        # ---------------------------------------------------------------------
        #  We have implemented the part of sampling the next candidate given
        #  the truncated `potential_cands` and `last_id_probs` so you
        #  shouldn't touch this part of the code.
        cand = random.choices(potential_cands, weights=last_id_probs)[0]
    return cand


def generate_sentence(
    model: LangModel, decoder: DECODERS, max_length: int = 20, **decoder_kwargs
):
    assert max_length > 0
    if decoder == DECODERS.GREEDY:
        top_candidate = greedy_decoding(
            model=model,
            max_length=max_length,
            **decoder_kwargs,
        )
    elif decoder == DECODERS.MULTINOMIAL:
        top_candidate = multinomial_sampling(
            model=model,
            max_length=max_length,
            **decoder_kwargs,
        )
    elif decoder == DECODERS.TOP_K:
        top_candidate = top_k_sampling(
            model=model,
            top_k=3,
            temperature=0.5,
            max_length=max_length,
            **decoder_kwargs,
        )
    elif decoder == DECODERS.NUCLEUS:
        top_candidate = nucleus_sampling(
            model=model,
            top_p=0.2,
            max_length=max_length,
            **decoder_kwargs,
        )
    elif decoder == DECODERS.CONSTRAINED:
        top_candidate = constrained_decoding(
            model=model,
            max_length=max_length,
            **decoder_kwargs,
        )
    elif decoder == DECODERS.CONSTRAINED_NO_REP:
        top_candidate = constrained_decoding_no_repetition(
            model=model, max_length=max_length, **decoder_kwargs
        )
    else:
        raise ValueError("Unexpected error")

    seq = model.decode(top_candidate.decoded_ids)
    seq_score = top_candidate.log_prob
    return {"sequence": seq, "seq_log_prob": seq_score}
