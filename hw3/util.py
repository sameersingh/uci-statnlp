def load_embeddings(embedding_dim, word_list=[], embedding_path=None):
    import torch

    # initialize embeddings randomly
    if embedding_path is None:
        embeddings = torch.nn.Embedding(len(word_list)+2, embedding_dim)
    # read in pretrained embeddings
    # TODO: only load in embeddings of words in our word list
    else:
        word_list = []
        embeddings_list = []
        with open(embedding_path, encoding='utf-8') as f:
            for line in f:
                line = line.split()
                word_list.append(line[0])
                embeddings_list.append(torch.Tensor(list(map(float, line[1:]))))
        embeddings_list.append(torch.FloatTensor(embedding_dim).uniform_(-0.1, 0.1)) # <pad>
        embeddings_list.append(torch.FloatTensor(embedding_dim).uniform_(-0.1, 0.1)) # <unk>

        # init a random Embedding object
        embeddings = torch.nn.Embedding(len(embeddings_list), embedding_dim)
        # set embedding weights to the embeddings we loaded
        embeddings.weight.data.copy_(torch.vstack(embeddings_list))

    return embeddings, word_list


def create_object_from_class_string(module_name, class_name, params):
    import importlib
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**params)
    return instance


def load_torch_object(params: dict):
    type = params.pop('type')
    if type is None:
        return None
    else:
        type = type.split('.')
        module_name, class_name = '.'.join(type[:-1]), type[-1]
        return create_object_from_class_string(module_name, class_name, params)
