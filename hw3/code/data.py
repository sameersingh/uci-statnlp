from dataclasses import dataclass
from typing import List


import json


@dataclass
class Answer:
    text: str
    score: float = 1.0


@dataclass
class ODQADataset:
    """Open-Domain QA dataset

    Attributes
    ----------
    documents: list[str]
        List of documents in the corpus. They can contain multiple sentences.

    queries: list[str]
        Each string represents one question (or query)

    gold_answers: list[str]
        Each string represents the gold truth answer that matches the
        question at the same index.

    gold_documents: list[str]
        The documents that contain the answer to a specific question.
    """

    documents: List[str]
    queries: List[str]
    gold_answers: List[str]
    _documents_mapping_per_query: List[List[int]]

    @property
    def gold_documents(self) -> List[List[str]]:
        """The textual gold documents matching each qa pair in the corpus."""
        gold_docs = []

        for query_docs_ids in self._documents_mapping_per_query:
            docs = [self.documents[idx] for idx in query_docs_ids]
            gold_docs.append(docs)

        return gold_docs

    @property
    def ndocuments(self):
        return len(self.documents)


def load_dataset(datapath: str) -> ODQADataset:
    """Loads the dataset from the specified datapath.

    Notes
    -----
    This method assumes that the file respects the following format:
    contexts: list[str]
        Each string is one document in our system. They can be composed
        of multiple sentences.
    questions: list[str]
        Each string represents one question (or query)
    answers: list[str]
        Each string represents the gold truth answer that matches the
        question at the same index.
    map_qa_pairs_to_context: list[list[int]]
      Each (question, answer) pair is mapped to a list of documents that
      contain the answer to the same question. These indices directly
      map to the contexts variable. That is, an index of 0 in this
      map_qa_pairs_to_context, will correspond to `contexts[0]`.

    Additionally, the following properties should be verified to
    guarantee that the file is structured as expected:
        len(contexts) > len(questions) = len(answers)
    """

    with open(datapath) as f:
        data = json.load(f)

    contexts = data["contexts"]
    print("Number of contexts:", len(contexts))
    questions = data.get("questions", [])
    print("Number of questions:", len(questions))
    answers = data.get("answers")
    print("Number of answers:", len(answers))
    assert len(questions) == len(answers)

    qa_pairs2context = data.get("map_qa_pairs_to_context", [])
    assert len(questions) == len(qa_pairs2context)

    return ODQADataset(contexts, questions, answers, qa_pairs2context)


def persist_dataset(dataset: ODQADataset, datapath: str):
    data_json = {
        "contexts": dataset.documents,
        "questions": dataset.queries,
        "answers": dataset.gold_answers,
        "map_qa_pairs_to_context": dataset._documents_mapping_per_query,
    }

    with open(datapath, "wt") as f:
        json.dump(data_json, f, ensure_ascii=True, indent=2)
