"""Tools for creating / loading vocabularies.

This module defines the Vocab() object which is used to map text data to
integers.

In addition, it provides a command-line utility for creating vocabularies from
text corpora and serializing them to disk. Example usage:

    python utils/vocab.py [input] [output]
"""
from __future__ import division
from __future__ import print_function

import argparse
from collections import Counter


FLAGS = None


class Vocab(object):

    def __init__(self,
                 pad_token='<pad>',
                 sos_token='<s>',
                 eos_token='</s>',
                 unk_token='<unk>'):
        """Initializes the vocabulary.

        Args:
            pad_token: Token used for padding.
            sos_token: Token used to denote the start of a sentence.
            eos_token: Token used to denote the end of a sentence.
            unk_token: Token used to denote an out-of-vocabulary word.
        """
        # Special tokens.
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # Initialize word mappings.
        self._counter = Counter()
        self._id2word = [pad_token, sos_token, eos_token]
        self._word2id = {y: x for x, y in enumerate(self._id2word)}

    def __len__(self):
        return len(self._word2id) + 1 # +1 for <UNK> token

    def word2id(self, word):
        """Maps a word to its corresponding index in the vocabulary.

        Args
            word: Word to lookup.
        """
        if word in self._word2id:
            return self._word2id[word]
        else:
            return len(self._word2id)

    def id2word(self, id):
        """Maps an index to its corresponding word in the vocabulary.

        Args:
            id: Index to lookup.
        """
        if id < len(self) - 1:
            return self._id2word[id]
        elif id == len(self) - 1:
            return self.unk_token
        else:
            raise KeyError('Index larger than vocab size.')

    def from_counter(self, counter):
        """Builds the vocabulary from a Counter object.

        Args:
            counter: The Counter object.
        """
        self._counter = counter
        pairs = counter.items()
        pairs = sorted(pairs, key = lambda x: x[1], reverse=True)
        words = [x[0] for x in pairs]
        self._id2word = [self.pad_token, self.sos_token, self.eos_token] + words
        self._word2id = {y: x for x, y in enumerate(self._id2word)}

    @classmethod
    def load(cls, f):
        """Loads the vocabulary from a file.

        Args:
            f: File whose lines contain the vocab words.

        Returns:
            The loaded vocabulary.
        """
        vocab = cls()
        word_count_tuples = [line.strip().split('\t') for line in f]
        word_count_tuples = [(x, int(y)) for x, y in word_count_tuples]
        counter = Counter()
        for word, count in word_count_tuples:
            counter[word] = count
        vocab.from_counter(counter)
        return vocab

    def dump(self, f):
        """Dumps the vocabulary to a file.

        Args:
            f: File to dump the vocabulary to.
        """
        template = '%s\t%i\n'
        for word, count in self._counter.most_common():
            f.write(template % (word, count))


def main(_):
    counter = Counter()

    print('Parsing %s' % FLAGS.input)
    with open(FLAGS.input, 'r') as f:
        for line in f:
            words = line.split()
            counter.update(words)

    vocab = Vocab()
    vocab.from_counter(counter)

    print('Writing to %s' % FLAGS.output)
    with open(FLAGS.output, 'w') as f:
        vocab.dump(f)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='The input file.')
    parser.add_argument('output', type=str, help='The output file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

