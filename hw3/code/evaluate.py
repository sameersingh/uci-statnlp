from typing import List

import numpy as np


def preprocess(text: str) -> str:
    """Apply the following preprocessing steps to the input text

    1. Lower case the text
    2. Remove punctuation
    3. Remove articles like "a" "an" "the"
    4. Fix "whitespace" ('  ' --> ' ')
    """
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def exact_match(ground_truth: str, prediction: str, with_preproc: bool=True):
    if with_preproc:
        return preprocess(prediction) == preprocess(ground_truth)
    else:
        return prediction == ground_truth


def evaluate_reader(gold_answers: List[str], predicted_answers: List[str]):
    assert len(gold_answers) == len(predicted_answers)

    results = []
    for gold, pred in zip(gold_answers, predicted_answers):
        assert len(gold) > 0, "Unexpected erro - Gold answer is ''" # PASTELBELEM8 REMOVE
        results.append(exact_match(gold, pred))

    return np.mean(results)


def evaluate_retriever(gold_documents: List[List[str]], retrieved_documents: List[List[str]]):
    """Evaluate the retriever's accuracy by checking whether any of the gold documents
    appear within the retrieved documents.

    Notes
    -----
    There's an assumption that the list of gold_documents comes in the same
    order as the list of retrieved documents. That is, they refer to the same
    (question, answer) pair.

    Parameters
    ----------
    gold_documents: list[list[str]]
        List of reference documents that were associated with a particular question.

    retrieved_documents: list[list[str]]
        List of retrieved documents that were associated with a particular question.
    """
    assert len(np.unique([len(docs) for docs in retrieved_documents])) == 1, "Number of retrieved documents differs"

    results = []
    for gold_lst, retrieved_lst in zip(gold_documents, retrieved_documents):
        # Check if any of the gold documents occurs in the retrieved list
        for gold in gold_lst:
            if gold in retrieved_lst:
                results.append(1)
                break
        else:
            results.append(0)

    assert len(results) == len(gold_documents), "Debugging -- shouldn't happen"
    return np.mean(results)
