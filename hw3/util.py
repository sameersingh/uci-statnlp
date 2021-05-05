def load_embeddings(embedding_dim, token_vocab, embedding_path=None):
    import torch

    # initialize embeddings randomly
    if embedding_path is None:
        embeddings = torch.nn.Embedding(len(token_vocab), embedding_dim)
    # read in pretrained embeddings
    else:
        word_list = []
        embeddings_list = []
        with open(embedding_path, encoding='utf-8') as f:
            for line in f:
                line = line.split()
                word_list.append(line[0])
                embeddings_list.append(torch.Tensor(list(map(float, line[1:]))))

        # create embeddings for special tokens (e.g. UNK)
        for _ in range(len(token_vocab.special_tokens)):
            embeddings_list.append(torch.FloatTensor(embedding_dim).uniform_(-0.1, 0.1))

        # init a random Embedding object
        embeddings = torch.nn.Embedding(len(embeddings_list), embedding_dim)
        # set embedding weights to the embeddings we loaded
        embeddings.weight.data.copy_(torch.vstack(embeddings_list))
        # update word list in token vocabulary with words from embedding file
        token_vocab.word_list = word_list

    return embeddings


def create_object_from_class_string(module_name, class_name, parameters):
    import importlib
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**parameters)
    return instance


def load_object_from_dict(parameters: dict, **kwargs):
    parameters.update(kwargs)
    type = parameters.pop('type')
    if type is None:
        return None
    else:
        type = type.split('.')
        module_name, class_name = '.'.join(type[:-1]), type[-1]
        return create_object_from_class_string(module_name, class_name, parameters)
