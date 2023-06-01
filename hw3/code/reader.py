from data import Answer
from typing import List, Tuple, Union

import numpy as np
import torch


class Reader:
    """Simple reader class

    The default reader implementation is very simple. Given
    a set of documents and a query, this reader class assumes
    the answer to the query is located in the first paragraph
    of a document.
    """

    def __init__(self, answer_selection: str = "first", batch_size: int = 32):
        self.mode = answer_selection.lower()
        self.batch_size = batch_size

    def _select_answer(
        self, candidate_answers: List[Answer]
    ) -> Union[str, List[Answer]]:
        """Select the final subset of answers from a pool of candidate_answers.

        The provided answer selection strategies are:

        "first":
        returns the first candidate in the provided list of candidates.
        When using this mode, the output will be a string.

        "confidence":
        returns the candidate exhibiting higher score (implicit assumption
        that highest score is better).
        When using this mode, the output will be a string.

        "debug":
        returns all the candidate answers. Can be useful for debugging and
        analyzing the different scores associated with the answers.
        When using this mode, the output will be a List[Answer].
        """

        if self.mode == "first":
            return candidate_answers[0].text

        elif self.mode == "confidence":
            # ---------------------------------------------------------------------
            #  TODO - Implement confidence-based answer selection
            # ---------------------------------------------------------------------
            # To do this, you will be provided a list of candidate answers in the
            # same order as the relevant documents for a given query. The Answers
            # are data.Answer objects, constituting a text and a score.
            #
            # You should return the text of the candidate answer whose score is
            # the largest.
            # ---------------------------------------------------------------------
            raise NotImplementedError(f"To be updated by the student: {self.mode}")
            # ---------------------------------------------------------------------
            # Don't change anything below this point (: You've done enough!
            # Keep up with the good work buddy!
            # ---------------------------------------------------------------------
            return cand
        elif self.mode == "debug":
            return [cand for cand in candidate_answers]
        else:
            raise NotImplementedError(f"'{self.mode}' is currently not supported")

    def _find_candidates(
        self, query: str, documents: Union[str, List[str]]
    ) -> List[Answer]:
        """Select the first sentence of a document as the best answer
        to the specified query.

        Returns
        -------
        Answer
            The answer to the query. It will be a segment in the provided document.
            The score of how likely the model is that this is the answer.
        """
        documents = [documents] if isinstance(documents, str) else documents
        return [Answer(d.split(".")[0], 1) for d in documents]

    def find_answer(self, queries: str, documents: List[List[str]]) -> List[str]:
        """Given a set of relevant documents return the answer
        that better fits the queries."""
        answers = []

        for query, query_docs in zip(queries, documents):
            cand_answers = self._find_candidates(query, query_docs)
            answers.append(self._select_answer(cand_answers))

        return answers


class SpanReader(Reader):
    """Span-based Reader.

    This is implemented as a simple Question Answering (QA) system.
    BERT-based QA is traditionally treated in an extractive setting,
    or span prediction. Instead of generating text, the BERT model
    will produce the start and end indices of the span in the
    document that comprise the answer.

    Check the official BertForQuestionAnswering for more details on
    the model or implementation. Adapted the code from [1] to
    be more general to other model classes (e.g., RoBERTa models).

    References
    ----------
    [1] - https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/bert#transformers.BertForQuestionAnswering
    """

    def __init__(
        self, model_name: str, device: str = "cpu", max_length: int = 512, **kwargs
    ):
        """Constructor of SpanReader class.

        Parameters
        ----------
        model_name: str
            The name of the pretrained model to be used as a span extraction
            question answering. Should be BERT-based.

        device: str, defaults to "cpu"
            The name of the device to run this model on.

        max_length: int, defaults to 512
            The maximum number of tokens in the input, after which we truncate.
            This vary per model, but for most BERT-based models tends to be 512.
            Since span extraction models receive as input both the question and
            the document, this may cause some answers to be missed.
        """
        super().__init__(**kwargs)
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.model_name = model_name
        # Load the model
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

        self.model.eval()
        self.model.to(device)
        self.max_length = max_length

    def _find_candidates(
        self, query: str, documents: Union[str, List[str]]
    ) -> List[Answer]:
        """Obtain the span in the provided document that is more likely to
        be the answer to the specified query and the associated confidence
        scores in that answer.

        Parameters
        ----------
        query: str
            The question that we want to find the information for.

        documents: Union[str, List[str]]
            The list of supporting documents that we will consider when
            looking for an answer.

        Returns
        -------
        List[Answer]
            The list of candidate answers to the provided query, in the same
            order as the provided documents. For SpanReader class this matches
            a segment in each document.
        """

        def _correct_answer(answer: str) -> str:
            corrected_answer = ""
            for word in answer.split():
                corrected_answer += word[2:] if word[0:2] == "##" else " " + word
            return corrected_answer

        def _batch_find(query_doc_pairs: Tuple[str, str]) -> List[Answer]:
            encoding = self.tokenizer.batch_encode_plus(
                query_doc_pairs,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            # print(encoding["input_ids"].shape) # HELPS DEBUGGING :3
            # Input tokens will later be useful to convert the ids back to strings
            tokens = [
                self.tokenizer.convert_ids_to_tokens(enc)
                for enc in encoding["input_ids"]
            ]  #  input tokens

            # Foward through the model to obtain the ids of the predictions
            outputs = self.model(**encoding)

            start_indices = torch.argmax(outputs["start_logits"], dim=-1).tolist()
            end_indices = torch.argmax(outputs["end_logits"], dim=-1).tolist()

            start_probs = torch.softmax(outputs["start_logits"], dim=-1).tolist()
            end_probs = torch.softmax(outputs["end_logits"], dim=-1).tolist()

            answers = []
            for i, start_index, end_index in zip(
                range(len(documents)), start_indices, end_indices
            ):
                answer = " ".join(tokens[i][start_index : end_index + 1])
                corrected_answer = _correct_answer(answer)

                # scores
                start_prob = start_probs[i][start_index]
                end_prob = end_probs[i][end_index]
                answers.append(Answer(corrected_answer, start_prob * end_prob))
            return answers

        # Obtain encoding of query, document pair
        query_doc_pairs = [(query, d) for d in documents]

        # In case we have too many documents being passed to the reader
        # (e.g., when using the gold retrieved evaluation), we may have
        # to tweak the batch size of the reader class (to be able to
        # fit everything in memory)
        results = []
        for start in range(0, len(query_doc_pairs), self.batch_size):
            batch = query_doc_pairs[start : start + self.batch_size]
            out = _batch_find(batch)
            results.extend(out)

        return results


# ---------------------------------------------------------------------
#  TODO - Implement Generative QAReader
# ---------------------------------------------------------------------
# 1. Define the constructor
#   * Given a model name, your constructor should preload the model and
#     tokenizer of the corresponding model name.
#
# 2. Define the _find_candidates method:
#   * the method expects a single query and a list of supporting
#     documents.
#   * we recommend you using the method generate from huggingface to
#     generate answers using greedy decoding (num_samples=1, do_sample=False)
#   * if you install the 4.26 (or greater) version of transformers,
#     you can also consider using the compute_transition_scores method
#     to compute the scores associated with each sequence. Note that
#     this method will return the probability associated with each
#     generated token and you may want to compute the average of log
#     scores to normalize by length.
#
#  Some potentially useful resources when implementing the scores:
#
# https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14
# https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/3
#
# ---------------------------------------------------------------------
class GenerativeQAReader(Reader):
    """Generative question answering reader.

    Instead of extracting an answer directly from the provided document,
    generative QA reader will generate one. As a result, the provided
    answer may not be directly present in the provided document.
    """

    def __init__(self, **kwargs):
        pass

    def _find_candidates(
        self, query: str, documents: Union[str, List[str]]
    ) -> List[Answer]:
        pass