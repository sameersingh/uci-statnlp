from lm import LangModel
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pickle
import torch
import torch.optim as optim

import neural_utils as utils
import neural_data_utils as data


def compute_norm_metadata(parameters) -> Dict[str, float]:
    from collections import defaultdict
    metadata = defaultdict(list)

    with torch.no_grad():
        for params in parameters:
            p_grad = params.grad.detach()

            # metadata["l1_norm"].append(torch.norm(p_grad, 1).item())
            metadata["l2_norm"].append(torch.norm(p_grad, 2).item())
            # metadata["frobenius_norm"].append(torch.norm(p_grad, "fro").item())
            # metadata["nucl_norm"].append(torch.norm(p_grad, "nuc").item())
            metadata["-inf_norm"].append(torch.norm(p_grad, -torch.inf).item())
            metadata["+inf_norm"].append(torch.norm(p_grad, torch.inf).item())
            metadata["avg_grad"].append(torch.mean(p_grad).item())
            metadata["std_grad"].append(torch.std(p_grad).item())

    return metadata


class NeuralLM(LangModel):
    """Seq2seq Language Modeling class

    It is a wrapper class around the trainer class.

    We based off this implementation on the code from the blogpost [1]
    and tweak it to fit our ``lm.LangModel` implementation and
    support other features, such as handling padding.

    The default loss function is cross-entropy loss, and the base neural
    module is LSTM (potentially stacked).

    References
    ----------
    [1 - LM with LSTMs in Pytorch](https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf)
    [2 - Taming LSTMs variable sized mini batches](https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e)
    [3 - BucketIterator for grouping text sequences by length](https://gmihaila.medium.com/better-batches-with-pytorchtext-bucketiterator-12804a545e2a)
    [4 - Recent version of BucketIterator](https://medium.com/@bitdribble/migrate-torchtext-to-the-new-0-9-0-api-1ff1472b5d71)
    [5 - Pytorch Official tutorials]
    """
    _NAME_ = "neural"
    PAD_TOKEN = "<pad>"

    def __init__(self, model_configs: Dict[str, Any], filepath=None, **kwargs):
        super().__init__(**kwargs)
        self.is_ngram = False

        # Add pad token
        self.pad_token_id = len(self._word2id)
        self._word2id[self.PAD_TOKEN] = self.pad_token_id
        self._id2word[self.pad_token_id] = self.PAD_TOKEN

        self.model_configs = model_configs
        self.model_configs["padding_idx"] = self.pad_token_id

        self.running_loss = None
        self.grad_metadata = None
        self.loss_by_step = []

        # Initalize the model
        if filepath is not None:
            self.model = utils.LSTMWrapper.load(filepath, **deepcopy(model_configs))
        else:
            self.model = utils.LSTMWrapper(vocab=self.vocab, vocab_size=self.vocab_size, **deepcopy(model_configs))
        self.model.to(self.model.device)


    @property
    def name(self):
        return self._NAME_

    def _preprocess_data_extra(self, sentence: List[str]) -> torch.LongTensor:
        """Maps the words (in textual representation) to corresponding
        indices in the vocabulary."""
        return torch.LongTensor([self.word2id(w) for w in sentence])

    def parameters(self):
        return self.model.parameters()

    def fit_sentence(self, sentence: List[str], **kwargs):
        """Wrapper around the fit corpus."""
        data = [sentence]
        self.fit_corpus(data, **kwargs)

    def fit_corpus(
        self,
        corpus: List[List[torch.LongTensor]],
        optimizer: optim.Optimizer,
        batch_size: int,
        max_seq_len: int,
        clip: float = None,
        clip_mode: str = None,
    ):
        # We assume that self.preprocess_data was called before calling training.
        train_dataset = data.LMDataset(corpus, max_seq_len)
        # https://torchtext.readthedocs.io/en/latest/data.html#bucketiterator
        train_dataloader = data.get_dataloader(train_dataset, batch_size, self.pad_token_id)

        # Initializations
        self.model.train()

        running_loss, num_tokens = 0, 0
        self.loss_by_step, self.grad_metadata = [], []
        for batch in train_dataloader:
            self.model.zero_grad()  # zero-out gradient
            # Step 1. Obtain the inputs, targets
            # inputs is list of array-like of shape (seq_len,)
            # target is list of array-like of shape (seq_len,)
            inputs_len, inputs, targets = batch
            batch_tokens = sum(inputs_len)

            # prediction is array-like of shape [batch_size, seq_len, output_dim]
            loss, _ = self.model(inputs, targets)
            (loss / batch_tokens).backward()
            # Avoid gradient explosion by clipping the gradient above clip
            if clip_mode == "grad":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            elif clip_mode == "val":
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip)

            self.grad_metadata += [compute_norm_metadata(self.model.parameters())]

            optimizer.step() # update parameters
            self.loss_by_step += [loss.detach().sum().item()]
            num_tokens += batch_tokens - len(inputs_len)
            running_loss += self.loss_by_step[-1]
        self.running_loss = running_loss / num_tokens

    def cond_logprob_dist(self, context: torch.LongTensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            context = context.view(1, -1).to(self.model.device)
            _, logits = self.model(context)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)

        return logits[0, -1, :].cpu().numpy().flatten()

    def cond_logprob(self, word: str, context: List[str]) -> float:
        word_id = self.word2id(word)
        dist = self.cond_logprob_dist(context)
        return dist[word_id]

    def logprob_sentence(self, sentence: torch.LongTensor) -> float:
        self.model.eval()
        with torch.no_grad():
            inputs, targets = sentence[:-1], sentence[1:]
            loss, _ = self.model(inputs.view(1, -1), targets)

        return - loss.sum().cpu().numpy()

    def evaluate(self, sentences: List[torch.Tensor]) -> float:
        """Computes the average log loss per token in the specified data."""
        loss = 0
        num_tokens = 0
        for sentence in sentences:
            loss += self.logprob_sentence(sentence)
            num_tokens += len(sentence) - 1

        return - loss / num_tokens

    def save_model(self, filepath: str):
        """Persist the current model to the specified filepath."""
        if filepath.endswith(".pkl"):
            filepath = filepath[:-4]

        # Save model
        self.model.save(f"{filepath}__model.pkl")

        # Save base class (without model)
        model = self.model
        self.model = None
        super().save_model(f"{filepath}__base.pkl")
        # note: we may want to keep using this instance, so we
        # recover the original model
        self.model = model

    @staticmethod
    def load_model(filepath: str, **kwargs) -> "NeuralLM":
        """Load a model from the specified filepath."""
        if filepath.endswith(".pkl"):
            filepath = filepath[:-4]

        # Load base class
        with open(f"{filepath}__base.pkl", "rb") as f:
            model = pickle.load(f)

        # Load LSTM module
        model.model = utils.LSTMWrapper.load(
            f"{filepath}__model.pkl", vocab=model.vocab, vocab_size=model.vocab_size, **model.model_configs
        )
        model.model.to("cuda" if torch.cuda.is_available() else "cpu")

        return model
