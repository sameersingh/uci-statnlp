"""Neural machine translation model implementation."""

from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embedding_dim):
        """Initializes the encoder.

        Args:
            src_vocab_size: Number of words in source vocabulary.
            embedding_dim: Size of word embeddings.
        """
        # Always do this when defining a Module.
        super(Encoder, self).__init__()

        # Track input parameters.
        self.src_vocab_size = src_vocab_size
        self.embedding_dim = embedding_dim

        # Define the model layers.
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, embedding_dim)

        # Initialize the embedding weights.
        nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, src, hidden=None):
        """Computes the forward pass of the encoder.

        Args:
            src: LongTensor w/ dimension [seq_len].
            hidden: FloatTensor w/ dimension [embedding_dim].

        Returns:
            net: FloatTensor w/ dimension [seq_len, embedding_dim].
                The encoder outputs for each element in src.
            hidden: FloatTensor w/ dimension [embedding_dim]. The final hidden
                activations.
        """
        # If no initial hidden state provided, then use all zeros.
        if hidden is None:
            hidden = Variable(torch.zeros(1, 1, self.embedding_dim))
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        # Forward pass.
        net = self.embedding(src).view(1, 1, -1) # Embed
        net, hidden = self.rnn(net, hidden) # Feed through RNN

        return net, hidden


class Decoder(nn.Module):
    def __init__(self,
                 tgt_vocab_size,
                 embedding_dim):
        """Initializes the decoder.

        Args:
            tgt_vocab_size: Number of words in target vocabulary.
            embedding_dim: Size of word embeddings.
        """
        # Always do this when defining a Module.
        super(Decoder, self).__init__()

        # Track input parameters.
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim

        # Define the model layers.
        self.embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, tgt_vocab_size)

        # Initialize the embedding weights.
        nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, tgt, hidden):
        """Computes the forward pass of the decoder.

        Args:
            tgt: LongTensor w/ dimension [seq_len, batch_size].
            lengths: LongTensor w/ dimension [seq_len]. Contains the lengths of
                the sequences in tgt.
            hidden: FloatTensor w/ dimension [TBD].

        Returns:
            net: FloatTensor w/ dimension [seq_len, batch_size, embedding_dim].
                The decoder outputs for each element in tgt.
            hidden: FloatTensor w/ dimension [TBD]. The final hidden
                activations.
        """
        # Forward pass.
        net = self.embedding(tgt).view(1, 1, -1) # Embed
        net, hidden = self.rnn(net, hidden) # Feed through RNN
        net = self.fc(net) # Feed through fully connected layer
        net = F.log_softmax(net[0], dim=-1) # Transform to log-probabilities

        return net, hidden

