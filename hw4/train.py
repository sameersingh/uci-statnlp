"""Model training script.

Usage:
    python train.py --config config.yaml
"""
from __future__ import division
from __future__ import print_function

import argparse
import logging
import numpy as np
import os
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from model import Encoder, Decoder
from utils import ShakespeareDataset


FLAGS = None


def main(_):
    # Load the configuration file.
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    # Load the training and dev datasets.
    train_data = ShakespeareDataset('train', config)
    dev_data = ShakespeareDataset('dev', config)
    src_vocab_size = len(train_data.src_vocab)
    tgt_vocab_size = len(train_data.tgt_vocab)

    # Build the model.
    encoder = Encoder(src_vocab_size, config['model']['embedding_dim'])
    decoder = Decoder(tgt_vocab_size, config['model']['embedding_dim'])
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    # Define the loss function + optimizer.
    loss_weights = torch.ones(decoder.tgt_vocab_size)
    loss_weights[0] = 0
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()
    criterion = torch.nn.NLLLoss(loss_weights)

    learning_rate = config['training']['learning_rate']
    encoder_optimizer = torch.optim.Adam(encoder.parameters(),
                                        lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(),
                                        lr=learning_rate)

    # Restore saved model (if one exists).
    ckpt_path = config['data']['ckpt']
    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        encoder_optimizer.load_state_dict(ckpt['encoder_optimzer'])
        decoder_optimizer.load_state_dict(ckpt['decoder_optimzer'])
    else:
        epoch = 0

    log_string = 'Epoch %i: train loss - %0.4f, dev loss - %0.4f'
    while epoch < config['training']['num_epochs']:

        # Main training loop.
        train_loss = []
        sampler = RandomSampler(train_data)
        for train_idx in sampler:
            src, tgt = train_data[train_idx]

            # Clear gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Feed inputs one by one from src into encoder.
            src_length = src.size()[0]
            hidden = None
            for i in range(src_length):
                encoder_output, hidden = encoder(src[i], hidden)

            # Feed desired outputs one by one from tgt into decoder
            # and measure loss.
            tgt_length = tgt.size()[0]
            loss = 0
            for i in range(tgt_length - 1):
                decoder_output, hidden = decoder(tgt[i], hidden)
                loss += criterion(decoder_output, tgt[i+1])

            # Backpropagate the loss and update the model parameters.
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            train_loss.append(loss)
            print(loss)

        # Evaluation loop.
        dev_loss = []
        for src, tgt in dev_data:
            # Clear gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Feed inputs one by one from src into encoder.
            src_length = src.size()[0]
            hidden = None
            for i in range(src_length):
                encoder_output, hidden = encoder(src[i], hidden)

            # Feed desired outputs one by one from tgt into decoder
            # and measure loss.
            tgt_length = tgt.size()[0]
            loss = 0
            for i in range(tgt_length - 1):
                decoder_output, hidden = decoder(tgt[i], hidden)
                loss += criterion(decoder_output, tgt[i+1])

            dev_loss.append(loss)

        print(log_string % (epoch, np.mean(train_loss), np.mean(dev_loss)))

        state_dict = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict()
        }
        torch.save(state_dict, ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

