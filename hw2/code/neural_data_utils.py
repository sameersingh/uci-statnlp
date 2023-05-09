"""Utility file containing the building blocks for LSTM-inspired
language modeling.

Exposed classes:
LSTMWrapper:
    Language modeling wrapper around pytorch's default LSTM module.

LMDataset:
    Pytorch dataset class for loading data.
"""
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from typing import Dict, List

import torch


class LMDataset(Dataset):
    """Dataset class to load the data and apply some further preprocessing"""
    def __init__(self, train_data: List[torch.Tensor], max_seq_len: int=None):
        assert max_seq_len is None or max_seq_len > 0

        self.targets, self.inputs = [], []

        for t in train_data:
            if max_seq_len is None:
                max_seq_len = len(t)-1

            target = t[1:1+max_seq_len]
            inpt = t[:len(target)]


            self.targets.append(target)
            self.inputs.append(inpt)

        self.max_seq_len = max_seq_len

    def __len__(self):
        """Number of examples in the dataset"""
        return len(self.inputs)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        """Given an index return an example from the position.

        Parameters
        ----------
        item: int
            Index position to pick an example to return.

        Returns
        -------
        Dict[str, tensor]
            Dictionary of inputs that are used to feed to a model
        """

        return {
            "inputs": self.inputs[item],
            "targets": self.targets[item],
        }


def get_dataloader(lm_dataset: LMDataset, batch_size: int, padding_idx: int) -> DataLoader:
    def collate_batch(batch):
        targets, inputs = [], []
        lengths = []

        for example in batch:
            t, i = example["targets"], example["inputs"]

            assert len(t) == len(i), f"Length of target and input does not match: {len(t)} vs {len(i)}"
            targets.append(t)
            inputs.append(i)
            lengths.append(len(t))

        # Pad batch to dynamically amtch the longest sentence in a batch
        return (
            lengths,
            pad_sequence(inputs, padding_value=padding_idx, batch_first=True),
            pad_sequence(targets, padding_value=padding_idx, batch_first=True),
        )

    bucket_loader = DataLoader(
        lm_dataset,
        shuffle=True, batch_size=batch_size,
        collate_fn=collate_batch,
    )
    return bucket_loader
