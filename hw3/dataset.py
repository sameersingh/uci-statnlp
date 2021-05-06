from typing import List

import torch
from torch.utils.data import Dataset


class Vocabulary():
    """ Object holding vocabulary and mappings
    Args:
        word_list: ``list`` A list of words. Words assumed to be unique.
        add_unk_token: ``bool` Whether to create an token for unknown tokens.
    """
    def __init__(self, word_list, add_unk_token=False):
        self._pad_token = '<pad>'
        self._unk_token = '<unk>' if add_unk_token else None

        self._special_tokens = [self._pad_token]
        if self._unk_token:
            self._special_tokens += [self._unk_token]

        self.word_list = word_list

    def __len__(self):
        return len(self._token_to_id)

    @property
    def special_tokens(self):
        return self._special_tokens

    @property
    def pad_token_id(self):
        return self.map_token_to_id(self._pad_token)

    @property
    def word_list(self):
        return self._word_list

    @word_list.setter
    def word_list(self, wl):
        self._word_list = wl
        self._init_vocab()

    def _init_vocab(self):
        self._id_to_token = self._word_list + self._special_tokens
        self._token_to_id = {token: id for id, token in
                             enumerate(self._id_to_token)}

    def map_token_to_id(self, token: str):
        """ Maps a single token to its token ID """
        if token not in self._token_to_id:
            token = self._unk_token
        return self._token_to_id[token]

    def map_id_to_token(self, id: int):
        """ Maps a single token ID to its token """
        return self._id_to_token[id]

    def map_tokens_to_ids(self, tokens: List[str], max_length: int = None):
        """ Maps a list of tokens to a list of token IDs """
        # truncate extra tokens and pad to `max_length`
        if max_length:
            tokens = tokens[:max_length]
            tokens = tokens + [self._pad_token]*(max_length-len(tokens))
        return [self.map_token_to_id(token) for token in tokens]

    def map_ids_to_tokens(self, ids: List[int], filter_padding=True):
        """ Maps a list of token IDs to a list of token """
        tokens = [self.map_id_to_token(id) for id in ids]
        if filter_padding:
            tokens = [t for t in tokens if t != self._pad_token]
        return tokens


class TwitterDataset(Dataset):
    def __init__(self, data_path, max_length=30):
        self._max_length = max_length
        self._dataset = []
        self._load_dataset(data_path)

        self.token_vocab = None
        self.tag_vocab = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item: int):
        instance = self._dataset[item]
        return self.tensorize(tokens=instance['tokens'],
                              tags=instance['tags'],
                              max_length=self._max_length)

    def _load_dataset(self, dataset_file):
        # Hacky way to distinguish between POS and NER tasks.
        # We look for the word 'ner' or 'pos' in the data file, then
        # use it to determine which column contains the labels
        task = dataset_file.split('.')[-1]
        assert task == 'pos' or task == 'ner'
        label_column = 1 if task == 'pos' else 3

        # read the dataset file, extracting tokens and tags
        with open(dataset_file, 'r') as f:
            tokens, tags = [], []
            for line in f:
                elements = line.strip().split('\t')
                # empty line means end of sentence
                if elements == [""]:
                    self._dataset.append({'tokens': tokens, 'tags': tags})
                    tokens, tags = [], []
                else:
                    tokens.append(elements[0].lower())
                    tags.append(elements[label_column])

    def get_tokens_list(self):
        tokens = [token for d in self._dataset for token in d['tokens']]
        return sorted(set(tokens))

    def get_tags_list(self):
        tags = [tag for d in self._dataset for tag in d['tags']]
        return sorted(set(tags))

    def set_vocab(self, token_vocab: Vocabulary, tag_vocab: Vocabulary):
        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab

    def tensorize(self, tokens: list, tags: list = None, max_length: int = None):
        assert self.token_vocab is not None
        assert self.tag_vocab is not None
        if tags is not None:
            assert len(tokens) == len(tags)

        token_ids = self.token_vocab.map_tokens_to_ids(tokens, max_length)
        tensor_dict = {'token_ids': torch.LongTensor(token_ids)}
        if tags:
            tag_ids = self.tag_vocab.map_tokens_to_ids(tags, max_length)
            tensor_dict['tag_ids'] = torch.LongTensor(tag_ids)

        return tensor_dict