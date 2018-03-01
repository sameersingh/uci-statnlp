"""Model training script.

Usage:
    python train.py --config config.yaml
"""
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import shutil
import torch
import yaml
from datetime import datetime
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data.sampler import RandomSampler

from model import Encoder, Decoder
from utils import ShakespeareDataset, Vocab


FLAGS = None


def main(_):
    # Load the configuration file.
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    # Create the checkpoint directory if it does not already exist.
    ckpt_dir = os.path.join(config['data']['ckpt'], config['experiment_name'])
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # Check if a pre-existing configuration file exists and matches the current
    # configuration. Otherwise save a copy of the configuration to the
    # checkpoint directory.
    prev_config_path = os.path.join(ckpt_dir, 'config.yaml')
    if os.path.exists(prev_config_path):
        with open(prev_config_path, 'r') as f:
            prev_config = yaml.load(f)
        assert config == prev_config
    else:
        shutil.copyfile(FLAGS.config, prev_config_path)

    # Load the vocabularies.
    with open(config['data']['src']['vocab'], 'r') as f:
        src_vocab = Vocab.load(f)
    with open(config['data']['tgt']['vocab'], 'r') as f:
        tgt_vocab = Vocab.load(f)

    # Load the training and dev datasets.
    train_data = ShakespeareDataset('train', config, src_vocab, tgt_vocab)
    dev_data = ShakespeareDataset('dev', config, src_vocab, tgt_vocab)

    # Build the model.
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
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
    ckpt_path = os.path.join(ckpt_dir, 'model.pt')
    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        encoder_optimizer.load_state_dict(ckpt['encoder_optimizer'])
        decoder_optimizer.load_state_dict(ckpt['decoder_optimizer'])
    else:
        epoch = 0

    train_log_string = '%s :: Epoch %i :: Iter %i / %i :: train loss: %0.4f'
    dev_log_string = '/n%s :: Epoch %i :: dev loss: %0.4f'
    while epoch < config['training']['num_epochs']:

        # Main training loop.
        train_loss = []
        sampler = RandomSampler(train_data)
        for i, train_idx in enumerate(sampler):
            src, tgt = train_data[train_idx]

            # Clear gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Feed inputs one by one from src into encoder (in reverse).
            src_length = src.size()[0]
            hidden = None
            for j in reversed(range(src_length)):
                encoder_output, hidden = encoder(src[j], hidden)

            # Feed desired outputs one by one from tgt into decoder
            # and measure loss.
            tgt_length = tgt.size()[0]
            loss = 0
            for j in range(tgt_length - 1):
                decoder_output, hidden = decoder(tgt[j], hidden)
                loss += criterion(decoder_output, tgt[j+1])

            # Backpropagate the loss and update the model parameters.
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            train_loss.append(loss.data.cpu())

            # Every once and a while check on the loss
            if ((i+1) % 100) == 0:
                print(train_log_string % (datetime.now(), epoch, i+1, len(train_data), np.mean(train_loss)), end='\r')
                train_loss = []

        # Evaluation loop.
        dev_loss = []
        for src, tgt in dev_data:

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

            dev_loss.append(loss.data.cpu())

        print(dev_log_string % (datetime.now(), epoch, np.mean(dev_loss)))

        state_dict = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict()
        }
        torch.save(state_dict, ckpt_path)

        epoch += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

