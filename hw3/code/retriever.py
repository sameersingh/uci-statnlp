from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Any, List, Tuple, Union

from utils import load_embeddings_from_filepath

import faiss  # useful for building fast indices
import numpy as np
import os, requests, warnings


class Retriever:
    """Base retriever class.

    It exposes the necessary methods for retrieving the most
    relevant documents from a large pool of documents.
    """

    def __init__(self, tokenizer: callable):
        self.documents = []
        self.tokenizer = tokenizer

    @property
    def size(self) -> int:
        """Size of the pool of documents stored by the retriever."""
        return len(self.documents)

    def _docs_by_id(self, ids: List[int]) -> List[str]:
        """Get documents by their indices."""
        return [self.documents[idx] for idx in ids]

    def _fit(self, embeddings: Any):
        """Extra processing that can be useful by subclasses."""
        pass

    def encode_documents(self, documents: str) -> np.array:
        """Encode provided documents, defaults to the encode_queries."""
        return self.encode_queries(documents)

    def encode_queries(self, queries: Union[str, List[str]]) -> np.array:
        """Encode the provided queries."""
        queries = [queries] if isinstance(queries, str) else queries
        return [self.tokenizer(q) for q in queries]

    def fit(self, corpus: List[str]):
        """Indexes the documents."""
        self.documents = corpus

        vect_docs = self.encode_documents(corpus)
        self._fit(vect_docs)

    def retrieve(self, queries: str, k: int) -> List[str]:
        """Finds the ``k`` most relevant documents to specific queries."""
        raise NotImplementedError("must be overriden by subclass")


class BM25Retriever(Retriever):
    """BM25 based retriever

    The BM25 is a tf-idf weighting variant that adds components
    to normalize by document length and weight the tf and idf
    parts differently.

    It is known to produce a sparse representation that relies
    on word overlap to perform well. Nevertheless it is to the
    data a very strong baseline in most retriever systems.
    """

    def __init__(
        self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25, **kwargs
    ):
        super().__init__(**kwargs)

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Model will be fit when we obtain the corpus
        self.model = None

    def _fit(self, embeddings: List[List[str]]):
        """Fits the a rank_25.BM25Okapi model using the preprocess documents."""
        self.model = BM25Okapi(
            corpus=embeddings, k1=self.k1, b=self.b, epsilon=self.epsilon
        )
        # ^Note: class receives a list of lists of strings, which are the document tokens.

    def retrieve(
        self, queries: Union[str, List[str]], k: int
    ) -> Tuple[List[str], List[float]]:
        """Finds the ``k`` most relevant documents to specific queries.

        The method accepts both one simple query, expressed as a string or
        a list of queries, expressed as a list of strings.

        Return
        ------
        list[str]
            List of documents, expressed as strings, ordered by most relevant to each query.

        list[float]
            List of assigned score to each document, expressed as floats.
        """
        # Encode the query
        vect_queries = self.encode_queries(queries)

        scores, documents = [], []
        for vq in vect_queries:
            vq_scores = self.model.get_scores(query=vq)
            vq_ids = np.argsort(vq_scores)[::-1][:k]

            scores.append(vq_scores[vq_ids])
            documents.append(self._docs_by_id(vq_ids))
        return documents, scores


class BingRetriever(Retriever):
    """Bing Web Search API based retriever.

    This class leverates the REST API for Bing's Web Search API.
    If you'd like to use it, please consider heading over to
    https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
    and setting up the free tier account. The free tier account allows your to
    make 3 Transactions Per Second (TPS) and up to 1k calls per month free of
    charge. You might have to use your student email to obtain the student
    perks from Azure.

    References
    ----------
    [1] https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
    [2] https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
    [3] https://learn.microsoft.com/en-us/azure/cognitive-services/bing-web-search/quickstarts/python
    """

    def __init__(self, api_key: str):
        super().__init__(tokenizer=lambda x: x)
        self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        self.api_key = api_key

    @property
    def size(self) -> int:
        raise NotImplementedError

    def _bing_request(self, query, k=10):
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            # "count": k,
            "textDecorations": True,
            "textFormat": "HTML",
        }
        # get response
        response = requests.get(self.search_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def _extract_text(self, json_blob):
        import re

        passages = []
        for document in json_blob["webPages"]["value"]:
            text = document["snippet"]
            text = re.sub("\[[0-9]+\]", "", text)
            text = re.sub("\<.+?\>", "", text)
            passages.append(text)
        return passages

    def retrieve(self, queries: str, k: int = None) -> Tuple[List[str], List[float]]:
        """Finds the ``n`` most relevant documents to a specific query."""
        queries = [queries] if isinstance(queries, str) else queries
        documents = []
        documents_scores = []

        for query in queries:
            payload = self._bing_request(query, k=k)
            docs = self._extract_text(payload)

            # Temporarily, we will return a score that is linear in
            # the position of the retrieved documents.
            scores = np.arange(len(docs))[::-1]

            documents.append(docs[:k])
            documents_scores.append(scores)

        return documents, documents_scores


class FaissIndexMixin:
    """Mixin class that provides indexing functionality."""

    def __init__(self, index_path: str, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim

        self.index_path = index_path
        self.index = self.load_index(index_path)

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def _fit(self, embeddings: Any):
        """Using the provided embeddings creates an index."""
        if self.index.ntotal == 0:
            if (num_emb := embeddings.shape[0]) != self.index.ntotal:
                warnings.warn(
                    f"Dimension mismatch: {num_emb} (provided embeddings) "
                    f"!= {self.index.ntotal} (loaded embeddings)"
                )

            self.index.add(embeddings)
            self.save_index(self.index_path)

    def fit(self, corpus: List[str]):
        """Indexes the documents."""
        self.documents = corpus

        if self.index.ntotal == 0:
            vect_docs = self.encode_documents(corpus)
            self._fit(vect_docs)

    def load_index(self, filepath: str) -> faiss.IndexFlatL2:
        if filepath is not None and os.path.exists(filepath):
            index = faiss.read_index(filepath)
            print(f"Loaded index from '{filepath}' with {index.ntotal} embeddings.")
            return index

    def save_index(self, filepath: str, override: bool = False):
        """Save the current index at the filepath, optionally overriding previous file."""
        # persist the index automatically
        if override or (
            self.index.ntotal == len(self.documents) and not os.path.exists(filepath)
        ):
            # create directory if it doesn't exist
            os.makedirs(Path(filepath).parent, exist_ok=True)

            print("Persisting the index at", filepath)
            faiss.write_index(self.index, filepath)

    def retrieve(
        self, queries: Union[str, List[str]], k: int
    ) -> Tuple[List[str], List[float]]:
        vect_queries = self.encode_queries(queries)

        scores_by_query, indices_by_query = self.index.search(vect_queries, k)
        if (indices_by_query == -1).any():
            warnings.warn(
                f"Insufficient documents for top-{k} docs when using"
                f" queries:\n -> {queries}"
            )

        documents, documents_scores = [], []
        for indices, scores in zip(indices_by_query, scores_by_query):
            documents.append(self._docs_by_id(indices))
            documents_scores.append(scores)

        return documents, documents_scores


class AvgWordEmbeddingRetriever(FaissIndexMixin, Retriever):
    """Average Word Embedding retriever class

    It dynamically loads the embeddings from the specified
    embedding path and computes a dense representation of
    pieces of text by averaging the embeddings of each
    corresponding word.

    Downsides to this approach is that in many cases some
    words may not exist. If no word is found for a piece
    of text, a uniform vector is created with 1/emb_dim.

    Note: for larger hit ratio, i.e., to maximize the
    number of words that get a corresponding vector, consider
    the lower case version of the text.

    Download the embeddings from:
    - https://drive.google.com/drive/folders/1RxxhmaIoBI1rA6ly5E4tDlvOET7YRUWI?usp=sharing
    """

    def __init__(self, embedding_path: str, **kwargs):
        super().__init__(**kwargs)

        self.embedding_path = embedding_path
        self.word2embeddings = load_embeddings_from_filepath(embedding_path)

    def encode_queries(self, queries: str) -> np.array:
        queries = [queries] if isinstance(queries, str) else queries

        # break down the queries into lists of individual tokens
        vect_queries = [self.tokenizer(q) for q in queries]

        avg_embeddings = []
        for query in vect_queries:
            # retrieve the embeddings associated with each word in the query
            embs = [
                self.word2embeddings[tk] for tk in query if tk in self.word2embeddings
            ]

            if len(embs) == 0:
                warnings.warn(
                    f"Query {query} has no token overlap with embeddings in {self.embedding_path}."
                    f"Assigning uniform embedding by default..."
                )
                embs = np.ones_like((1, self.embedding_dim))
            else:
                embs = np.vstack(embs)

            avg_emb = np.mean(embs, axis=0).reshape(-1, self.embedding_dim)
            avg_emb_norm = np.linalg.norm(avg_emb, axis=1)
            avg_embeddings.append(avg_emb / avg_emb_norm[:, None])

        avg_embeddings = np.vstack(avg_embeddings)
        return avg_embeddings


# ---------------------------------------------------------------------
#  TODO - Implement Sentence Encoder Retriever
# ---------------------------------------------------------------------
# 1. Define the constructor
#   * Given a model name, your constructor should preload the model and
#     tokenizer of the corresponding model name.
#   * optionally, you have two model names, one for encoding the queries
#     and one for encoding the documents.
#   * use sentence-transformers to preload the sentence encoder model.
#
# 2. Define the encode_queries method:
#   * the method expects a query (or list of queries) and should return
#     an array with the l2-normalized corresponding embeddings.
#     The shape of the output array should be len(queries) x self.embedding_dim
#
# 3. Define the encode_documents method:
#   * the method expects a document (or list of documents) and should
#     return an array with the l2-normalized vectors for each document.
#     The shape of the output array should be len(documents) x self.embedding_dim
#
# ---------------------------------------------------------------------
class SentenceEncRetriever(FaissIndexMixin, Retriever):
    """Sentence encoder retriever class.

    It encodes the documents into dense fixed-sized vectors.
    By default, it will use the average embeddings of each subword
    in the document as the final embedding for each document.

    We will use FAISS [1] for efficient indexing of these vectors
    thus avoiding the bootstrap time you would spend at systematically
    indexing these vectors. For search, we encode a new sentence into a
    semantic vector query and pass it to the FAISS index. FAISS will
    retrieve the closest matching semantic vectors and return the most
    similar sentences. Compared to linear search, which scores the query
    vector against every indexed vector, FAISS enables much faster
    retrieval times that typically scale logarithmically with the number
    of indexed vectors. Additionally, the indexes are highly memory-
    -efficient because they compress the original dense vectors.

    References
    ----------
    [1] https://towardsdatascience.com/master-semantic-search-at-scale-index-millions-of-documents-with-lightning-fast-inference-times-fa395e4efd88
    """

    def __init__(self, **kwargs):
        pass

    def encode_queries(self, queries: Union[str, List[str]]) -> np.array:
        pass

    def encode_documents(self, documents: str) -> np.array:
        pass
