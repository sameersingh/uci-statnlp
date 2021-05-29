import copy
import itertools
import math
import random


class Candidate:
    """ A class containing information for a single candidate generation.

    Parameters:
    ----------
    score: ``float`` The score of the current generation. More
        specifically, this is the log-probability score of the current generation.
    last_id_prob: ``float`` The probability of generating the most recently
        decoded ID. This is not in log-space. This is useful for decoding
        strategies like top k and nucleus sampling.
    decoded_ids: ``list`` A list of all generated IDs in this `Candidate`
    metadata: ``dict`` A dictionary storing any useful metadata for generation
    """
    def __init__(self, score=0, last_id_prob=0, decoded_ids=[], metadata={}):
        self.score = score
        self.last_id_prob = last_id_prob
        self.decoded_ids = decoded_ids
        self.metadata = metadata

    def __len__(self):
        return len(self.decoded_ids)

    @property
    def last_decoded_id(self):
        return self.decoded_ids[-1] if len(self.decoded_ids) else None

    def get_next_cands(self, model):
        """
        Returns a list of `Candidate` objects which are all possible
        continuations of the current `Candidate` object. The continuation IDs
        and scores come from calling `model`.

        Parameters
        ----------
        model: ``Model`` A light-weight wrapper around an object which takes as
            input a list of the previously decoded IDs as well as any metadata,
            and returns scores for all possible continuations and updated
            metadata. See `models.py` for more details on this class.

        Returns
        -------
        cands: ``list[Candidate]`` Returns a list of `Candidate` objects
            which are all possible continuations of the current `Candidate`
        """
        # Detach the current metadata from other `Candidate` objects sharing it
        self.metadata = copy.deepcopy(self.metadata)

        # Feed metadata and decoded IDs through the model
        outputs = model(decoded_ids=self.decoded_ids, metadata=self.metadata)

        # Probability of seeing every following token
        next_id_probs = outputs['next_id_probs']
        assert type(next_id_probs) == list

        # Update the metadata
        self.metadata = outputs.get('metadata', {})

        # Generate candidates for every possible continuation of current candidate
        next_candidates = [
            Candidate(
                score=self.score+math.log(id_prob),
                last_id_prob=id_prob,
                decoded_ids=self.decoded_ids+[id],
                metadata=self.metadata
            )
            for id, id_prob in enumerate(next_id_probs)
        ]

        return next_candidates


def is_cand_finished(cand, max_length, eos_id):
    """
    A candidate is finished generating if the number of decoded IDs of
    the candidate is at the max length or if the EOS ID has been generated
    """
    if len(cand) >= max_length or cand.last_decoded_id == eos_id:
        return True
    else:
        return False


def top_k_sampling(
    model,
    top_k,
    temperature=1,
    max_length=50,
    eos_id=-1,
    decoded_ids=[],
    metadata=None
):
    """
    Parameters
    ----------
    model: ``Model`` Used to get continuation probabilities
    top_k: ``int`` Filters for the `top_k` candidates whose before sampling
    max_length: ``int`` Maximum allowed length of decoding
    eos_id: ``int`` End of sentence ID
    decoded_ids: ``list[int]`` List of decoded IDs to start the generation
    metadata: ``dict`` Metadata that may be used during generation

    Returns
    -------
    cand: ``Candidate`` The candidate at the end of top-k sampling
    """

    cand = Candidate(decoded_ids=decoded_ids, metadata=metadata)

    while not is_cand_finished(cand, max_length, eos_id):
        # Get possible continuation candidates
        potential_cands = cand.get_next_cands(model)

        # Sort continuations by their last decoded ID probabilities
        potential_cands = sorted(potential_cands, key=lambda x: x.last_id_prob, reverse=True)

        # Get probabilities for all the last decoded IDs
        last_id_probs = [cand.last_id_prob for cand in potential_cands]

        # Get the top-K continuations
        potential_cands = potential_cands[:top_k]
        last_id_probs = last_id_probs[:top_k]

        # TODO: Scale `last_id_probs` via temperature-scaling via the
        #  `temperature` parameter

        # Sample a candidate based on the probability of it's last ID
        # random.choices() automatically re-weights the probabilities!
        cand = random.choices(potential_cands, weights=last_id_probs)[0]

    return cand


def nucleus_sampling(
    model,
    top_p,
    max_length=50,
    eos_id=-1,
    decoded_ids=[],
    metadata=None
):
    """
    Parameters
    ----------
    model: ``Model`` Used to get continuation probabilities
    top_p: ``float`` Filters for the smallest possible set of candidates whose
        cumulative probability exceeds `top_p` before sampling.
    max_length: ``int`` Maximum allowed length of decoding
    eos_id: ``int`` End of sentence ID
    decoded_ids: ``list[int]`` List of decoded IDs to start the generation
    metadata: ``dict`` Metadata that may be used during generation

    Returns
    -------
    cand: ``Candidate`` The candidate at the end of nucleus sampling
    """
    cand = Candidate(decoded_ids=decoded_ids, metadata=metadata)

    # Continue decoding while the number of decoded IDs of the candidate is
    # less than the max length and while the EOS ID has not been generated.
    while not is_cand_finished(cand, max_length, eos_id):
        # Get possible continuation candidates
        potential_cands = cand.get_next_cands(model)

        # Sort continuations by their last decoded ID probabilities
        potential_cands = sorted(potential_cands, key=lambda x: x.last_id_prob, reverse=True)

        # Get probabilities for all the last decoded IDs
        last_id_probs = [cand.last_id_prob for cand in potential_cands]

        # TODO: finish implementing this part of the while loop to complete
        #  nucleus sampling. To do this you will have to:
        #  * Find where the cutoff point is that satisfies the `top_p`
        #    threshold (the cands and associated probabilities are already
        #    sorted so this should be fairly straightforward)
        #  * Update `potential_cands` and `last_id_probs` by truncating
        #    everything past this cutoff point.


        #  We have implemented the part of sampling the next candidate given
        #  the truncated `potential_cands` and `last_id_probs` so you
        #  shouldn't touch this part of the code.
        cand = random.choices(potential_cands, weights=last_id_probs)[0]

    return cand


def greedy_decoding(
    model,
    max_length=50,
    eos_id=-1,
    decoded_ids=[],
    metadata=None
):
    """
    Parameters
    ----------
    model: ``Model`` Used to get continuation probabilities
    max_length: ``int`` Maximum allowed length of decoding
    eos_id: ``int`` End of sentence ID
    decoded_ids: ``list[int]`` List of decoded IDs to start the generation
    metadata: ``dict`` Metadata that may be used during generation

    Returns
    -------
    cand: ``Candidate`` The highest scoring candidate at the end of the search
    """
    cand = Candidate(decoded_ids=decoded_ids, metadata=metadata)

    # Continue decoding while the number of decoded IDs of the candidate is
    # less than the max length and while the EOS ID has not been generated.
    while not is_cand_finished(cand, max_length, eos_id):
        # Get all possible continuation candidates
        potential_cands = cand.get_next_cands(model)

        # Sort list of candidates by their respective scores
        potential_cands = sorted(potential_cands, key=lambda x: x.score, reverse=True)

        # The next candidate is the best scoring potential candidate
        cand = potential_cands[0]

    return cand


def beam_search_decoding(
    model,
    beam_size,
    max_length=50,
    eos_id=-1,
    decoded_ids=[],
    metadata=None,
):
    """
    Parameters
    ----------
    model: ``Model`` Used to get continuation probabilities
    beam_size: ``int`` Number of candidates to keep on the beam
    max_length: ``int`` Maximum allowed length of decoding
    eos_id: ``int`` End of sentence ID
    decoded_ids: ``list[int]`` List of decoded IDs to start the generation
    metadata: ``dict`` Metadata that may be used during generation

    Returns
    -------
    finished_cands: ``list[Candidate]`` A sorted list of the best scoring
        finished candidates. The length of `finished_cands` is `beam_size`
    """
    cands = [Candidate(decoded_ids=decoded_ids, metadata=metadata)]
    finished_cands = []

    # Continue generating beam candidates while we don't have `beam_size`
    # number of finished candidates
    while len(finished_cands) < beam_size:
        # Get all possible continuation candidates for all current candidates
        potential_cands = [cand.get_next_cands(model) for cand in cands]

        # Flatten the list of lists into a single list of candidates
        potential_cands = list(itertools.chain(*potential_cands))

        # Sort the continuation candidates by their respective scores
        potential_cands = sorted(potential_cands, key=lambda x: x.score, reverse=True)

        # TODO: finish implementing this part of the while loop to complete
        #  beam search. To do this you will have to:
        #  * Get the top `beam_size - len(finished_cands)` number of candidates
        #    from potential candidates.
        #  * Of these candidates, add the completed ones to `finished_cands`
        #    (hint: use `is_cand_finished` function).
        #  * Update `cands` with remaining ones.


    # Sort the finished beams by their scores, then return the candidates
    finished_cands = sorted(finished_cands, key=lambda x: x.score, reverse=True)
    return finished_cands
