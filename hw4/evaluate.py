"""Model evaluation script.

Usage:
    python evaluate.py --config config.yaml
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import sys
import torch
import yaml
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable

from model import Encoder, Decoder
from utils import ShakespeareDataset, Vocab


FLAGS = None


class GreedyTranslator(object):
    def __init__(self,
                 encoder,
                 decoder,
                 tgt_vocab,
                 max_length=80):
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = tgt_vocab.word2id(tgt_vocab.sos_token)
        self.eos_id = tgt_vocab.word2id(tgt_vocab.eos_token)
        self.max_length = max_length

    def __call__(self, src):
        # Feed inputs one by one (backwards) from src into encoder.
        src_length = src.size()[0]
        hidden = None
        for i in reversed(range(src_length)):
            encoder_output, hidden = self.encoder(src[i], hidden)

        # Greedily translate.
        decoder_input = Variable(torch.LongTensor([self.sos_id]))
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()

        translation = [self.sos_id]

        for _ in range(self.max_length):

            # Feed data into decoder.
            decoder_output, hidden = self.decoder(decoder_input, hidden)

            # Find most likely word id.
            _, word_id = decoder_output.data.topk(1)
            word_id = word_id[0][0]

            translation.append(word_id)
            if word_id == self.eos_id:
                break

            # Convert word id to tensor to be used as next input to the decoder.
            decoder_input = Variable(torch.LongTensor([word_id]))
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()

        return translation


def main(_):
    # Load the configuration file.
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    # Load the vocabularies.
    src_vocab = Vocab.load(config['data']['src']['vocab'])
    tgt_vocab = Vocab.load(config['data']['tgt']['vocab'])

    # Load the training and dev datasets.
    test_data = ShakespeareDataset('test', config, src_vocab, tgt_vocab)

    # Restore the model.
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    encoder = Encoder(src_vocab_size, config['model']['embedding_dim'])
    decoder = Decoder(tgt_vocab_size, config['model']['embedding_dim'])
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    ckpt_path = os.path.join(config['data']['ckpt'], config['experiment_name'], 'model.pt')
    if os.path.exists(ckpt_path):
        print('Loading checkpoint: %s' % ckpt_path)
        ckpt = torch.load(ckpt_path)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
    else:
        print('Unable to find checkpoint. Terminating.')
        sys.exit(1)
    encoder.eval()
    decoder.eval()

    # Initialize translator.
    greedy_translator = GreedyTranslator(encoder, decoder, tgt_vocab)

    # Qualitative evaluation - print translations for first couple sentences in
    # test corpus.
    for i in range(10):
        src, tgt = test_data[i]
        translation = greedy_translator(src)
        src_sentence = [src_vocab.id2word(id) for id in src.data.cpu().numpy()]
        tgt_sentence = [tgt_vocab.id2word(id) for id in tgt.data.cpu().numpy()]
        translated_sentence = [tgt_vocab.id2word(id) for id in translation]
        print('---')
        print('Source: %s' % ' '.join(src_sentence))
        print('Ground truth: %s' % ' '.join(tgt_sentence))
        print('Model output: %s' % ' '.join(translated_sentence))
    print('---')

    # Quantitative evaluation - compute corpus level BLEU scores.
    hypotheses = []
    references = []
    for src, tgt in test_data:
        translation = greedy_translator(src)
        tgt_sentence = [tgt_vocab.id2word(id) for id in tgt.data.cpu().numpy()]
        translated_sentence = [tgt_vocab.id2word(id) for id in translation]
        # Remove start and end of sentence tokens.
        tgt_sentence = tgt_sentence[1:-1]
        translated_sentence = translated_sentence[1:-1]
        hypotheses.append(tgt_sentence)
        references.append([translated_sentence])
    print("Corpus BLEU score: %0.4f" % corpus_bleu(references, hypotheses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

