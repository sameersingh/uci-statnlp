"""Utility file containing the building blocks for LSTM-inspired
language modeling.

Exposed classes:
LSTMWrapper:
    Language modeling wrapper around pytorch's default LSTM module.

LMDataset:
    Pytorch dataset class for loading data.
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_embeddings(embedding_dim: int, vocab: List[str], padding_idx: int, embedding_path: str =None, init_range: float=0.1):
    # initialize embeddings randomly
    if embedding_path is None:
        embeddings = torch.nn.Embedding(num_embeddings=len(vocab),
                                        embedding_dim=embedding_dim)

    # read in pretrained embeddings
    else:
        word2embeddings = {}
        with open(embedding_path, encoding='utf-8') as f:
            for line in f:
                line = line.split()
                word = line[0]
                embedding = torch.Tensor(list(map(float, line[1:])))
                word2embeddings[word] = embedding

        # Since there may be some missing embeddings for some words
        # we will default initialize the embeddings
        ordered_embeddings = []
        for idx, word in enumerate(vocab):
            if idx == padding_idx:
                embeds = torch.FloatTensor(embedding_dim).zero_()
            else:
                embeds = word2embeddings.get(word, torch.FloatTensor(embedding_dim).uniform_(-init_range, init_range))
            ordered_embeddings.append(embeds)

        ordered_embeddings = torch.vstack(ordered_embeddings)
        embeddings = nn.Embedding.from_pretrained(ordered_embeddings, freeze=False, padding_idx=padding_idx)

    return embeddings


def create_object_from_class_string(module_name: str, class_name: str, parameters: dict):
    import importlib
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**parameters)
    return instance


def load_object_from_dict(parameters: dict, **kwargs):
    parameters.update(kwargs)
    type = parameters.get('type')
    if type is None:
        return None
    else:
        type = type.split('.')
        module_name, class_name = '.'.join(type[:-1]), type[-1]
        params = {k: v for k, v in parameters.items() if k != "type"}
        return create_object_from_class_string(module_name, class_name, params)


class LSTMWrapper(nn.Module):
    """LSTM Wrapper class for language modeling

    It is a wrapper class around the torch.LSTM model. We tailor
    it by adding word_embeddings and dropout to achieve better performance
    at language modeling objectives. You can feed torch.nn.LSTM keyword
    arguments during construction to make it arbitrarily more complex.

    We use part of the code from the blogpost [1] as a start and make
    some tweaks according to our needs, such as handling padding.


    Reference
    ---------
    [1](https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf)
    """

    def __init__(
        self,
        vocab: List[str],
        vocab_size: int,
        embeddings: Dict[str, Any],
        encoder: Dict[str, Any],
        projection: Dict[str, Any],
        padding_idx,
        device: str = None,
        **kwargs,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._out_dim = vocab_size - 1 # discount padding
        self._padding_idx = padding_idx

        self._embeddings = load_embeddings(**embeddings, vocab=vocab, padding_idx=padding_idx)
        self._emb_dim = embeddings["embedding_dim"]

        encoder["input_size"] = self._emb_dim
        encoder["batch_first"] = True
        self._encoder = load_object_from_dict(encoder)
        self._hid_dim = encoder["hidden_size"]

        projection["in_features"] = self._hid_dim
        projection["out_features"] = self._out_dim
        self._projection = load_object_from_dict(projection)

        assert padding_idx is not None
        self.loss = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, inputs: torch.Tensor, labels: torch.Tensor=None) -> tuple:
        inputs = inputs.to(self.device) # shape: batch_size x seq_len
        embeddings = self._embeddings(inputs)  # shape: batch_size x seq_len x embed_size
        encoder_outputs = self._encoder(embeddings)[0] if self._encoder else embeddings
        logits = self._projection(encoder_outputs) # shape: batch_size x seq_len x out_size

        if labels is None:
            return None, logits

        loss = self.loss(logits.view(-1, self._out_dim), labels.to(self.device).view(-1))
        return loss, logits

    def save(self, filepath: str):
        # save the structure of this class together with the model
        # (to store just the weights, we would use self.state_dict() instead)
        torch.save(self, filepath)

    @staticmethod
    def load(filepath: str) -> "LSTMWrapper":
        return torch.load(filepath)