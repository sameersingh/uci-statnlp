"""Utilities for loading/processing the Shakespeare data."""

from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from .vocab import Vocab


class ShakespeareDataset(Dataset):

    def __init__(self, mode, config):
        # Check mode.
        assert mode in ['train', 'dev', 'eval']
        self.mode = mode
        # Load data.
        with open(config['data']['src'][mode], 'r') as f:
            self.src_data = f.readlines()
        with open(config['data']['src']['vocab'], 'r') as f:
            self.src_vocab = Vocab.load(f)
        with open(config['data']['tgt'][mode], 'r') as f:
            self.tgt_data = f.readlines()
        with open(config['data']['tgt']['vocab'], 'r') as f:
            self.tgt_vocab = Vocab.load(f)
        # Check src and tgt datasets are the same length.
        assert len(self.src_data) == len(self.tgt_data)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # Split sentence into words.
        src_words = self.src_data[idx].split()
        tgt_words = self.tgt_data[idx].split()
        # Add <SOS> and <EOS> tokens.
        src_words = [self.src_vocab.sos_token] + src_words + [self.src_vocab.eos_token]
        tgt_words = [self.tgt_vocab.sos_token] + src_words + [self.tgt_vocab.eos_token]
        # Lookup word ids in vocabularies.
        src_ids = [self.src_vocab.word2id(word) for word in src_words]
        tgt_ids = [self.tgt_vocab.word2id(word) for word in tgt_words]
        # Convert to tensors.
        src_tensor = Variable(torch.LongTensor(src_ids))
        tgt_tensor = Variable(torch.LongTensor(tgt_ids))
        if torch.cuda.is_available():
            src_tensor = src_tensor.cuda()
            tgt_tensot = tgt_tensor.cuda()
        return src_tensor, tgt_tensor

