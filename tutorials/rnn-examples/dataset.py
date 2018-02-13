import re
import torch
from collections import Counter
from torch.utils.data import Dataset
from torch.autograd import Variable


def pad(sequences, max_length, pad_value=0):
    """Pads a list of sequences.

    Args:
        sequences: A list of sequences to be padded.
        max_length: The length to pad to.
        pad_value: The value used for padding.

    Returns:
        A list of padded sequences.
    """
    out = []
    for sequence in sequences:
        padded = sequence + [0]*(max_length - len(sequence))
        out.append(padded)
    return out


def collate_annotations(batch):
    """Function used to collate data returned by CoNLLDataset."""
    # Get inputs, targets, and lengths.
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    # Sort by length.
    sort = sorted(zip(inputs, targets, lengths),
                  key=lambda x: x[2],
                  reverse=True)
    inputs, targets, lengths = zip(*sort)
    # Pad.
    max_length = max(lengths)
    inputs = pad(inputs, max_length)
    targets = pad(targets, max_length)
    # Transpose.
    inputs = list(map(list, zip(*inputs)))
    targets = list(map(list, zip(*targets)))
    # Convert to PyTorch variables.
    inputs = Variable(torch.LongTensor(inputs))
    targets = Variable(torch.LongTensor(targets))
    lengths = Variable(torch.LongTensor(lengths))
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
        lengths = lengths.cuda()
    return inputs, targets, lengths


class Vocab(object):
    def __init__(self, iter, max_size=None, sos_token=None, eos_token=None, unk_token=None):
        """Initialize the vocabulary.

        Args:
            iter: An iterable which produces sequences of tokens used to update
                the vocabulary.
            max_size: (Optional) Maximum number of tokens in the vocabulary.
            sos_token: (Optional) Token denoting the start of a sequence.
            eos_token: (Optional) Token denoting the end of a sequence.
            unk_token: (Optional) Token denoting an unknown element in a
                sequence.
        """
        self.max_size = max_size
        self.pad_token = '<pad>'
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        id2word = [self.pad_token]
        if sos_token is not None:
            id2word.append(self.sos_token)
        if eos_token is not None:
            id2word.append(self.eos_token)
        if unk_token is not None:
            id2word.append(self.unk_token)

        counter = Counter()
        for x in iter:
            counter.update(x)

        if max_size is not None:
            counts = counter.most_common(max_size)
        else:
            counts = counter.items()
            counts = sorted(counts, key=lambda x: x[1], reverse=True)
        words = [x[0] for x in counts]
        id2word.extend(words)
        word2id = {x: i for i, x in enumerate(id2word)}

        self._id2word = id2word
        self._word2id = word2id

    def __len__(self):
        return len(self._id2word)

    def word2id(self, word):
        """Map a word in the vocabulary to its unique integer id.

        Args:
            word: Word to lookup.

        Returns:
            id: The integer id of the word being looked up.
        """
        if word in self._word2id:
            return self._word2id[word]
        elif self.unk_token is not None:
            return self._word2id[self.unk_token]
        else:
            raise KeyError('Word "%s" not in vocabulary.' % word)

    def id2word(self, id):
        """Map an integer id to its corresponding word in the vocabulary.

        Args:
            id: Integer id of the word being looked up.

        Returns:
            word: The corresponding word.
        """
        return self._id2word[id]


class Annotation(object):
    def __init__(self):
        self.tokens = []
        self.pos_tags = []


class CoNLLDataset(Dataset):
    def __init__(self, fname, target):
        """Initializes the CoNLLDataset.

        Args:
            fname: The .conllu file to load data from.
            target: Either 'lm' or 'pos'.
        """
        assert target in ['lm', 'pos'], 'Invalid target "%s".' % target
        self.target = target
        self.fname = fname
        self.annotations = self.process_conll_file(fname)
        self.token_vocab = Vocab([x.tokens for x in self.annotations],
                                  sos_token='<s>',
                                  eos_token='</s>',
                                  unk_token='<unk>')
        self.pos_vocab = Vocab([x.pos_tags for x in self.annotations])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        if self.target == 'lm':
            tokens = ['<s>', *annotation.tokens, '</s>']
            ids = [self.token_vocab.word2id(x) for x in tokens]
            input = ids[:-1]
            target = ids[1:]
        elif self.target == 'pos':
            input = [self.token_vocab.word2id(x) for x in annotation.tokens]
            target = [self.pos_vocab.word2id(x) for x in annotation.pos_tags]
        return input, target

    def process_conll_file(self, fname):
        # Read the entire file.
        with open(fname, 'r') as f:
            raw_text = f.read()
        # Split into chunks on blank lines.
        chunks = re.split(r'^\n', raw_text, flags=re.MULTILINE)
        # Process each chunk into an annotation.
        annotations = []
        for chunk in chunks:
            annotation = Annotation()
            lines = chunk.split('\n')
            # Iterate over all lines in the chunk.
            for line in lines:
                # If line is empty ignore it.
                if len(line)==0:
                    continue
                # If line is a commend ignore it.
                if line[0] == '#':
                    continue
                # Otherwise split on tabs and retrieve the token and the
                # POS tag fields.
                fields = line.split('\t')
                annotation.tokens.append(fields[1])
                annotation.pos_tags.append(fields[3])
            annotations.append(annotation)
        return annotations


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = CoNLLDataset('./data/en-ud-dev.conllu', 'pos')
    dataloader = DataLoader(ds, batch_size=12, shuffle=True,
                            collate_fn=collate_annotations)
    for i, batch in enumerate(dataloader):
        print(batch)
        if i > 20:
            break

