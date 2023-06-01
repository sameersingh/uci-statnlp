from typing import List

import importlib, nltk, re, numpy


def create_object_from_class_string(
    module_name: str, class_name: str, parameters: dict
):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**parameters)
    return instance


def load_object_from_dict(parameters: dict, **kwargs):
    parameters.update(kwargs)
    type = parameters.get("type")
    if type is None:
        return None
    else:
        type = type.split(".")
        module_name, class_name = ".".join(type[:-1]), type[-1]
        params = {k: v for k, v in parameters.items() if k != "type"}
        return create_object_from_class_string(module_name, class_name, params)


## A few tokenization methods:
def whitespace_tokenizer(text: str) -> List[str]:
    return text.split(" ")


def default_tokenizer(text) -> str:
    # remove punctuation from string
    text = re.sub(r"[^\w\s]", "", text)
    return nltk.word_tokenize(text)


def default_tokenizer_lower(text) -> str:
    return default_tokenizer(text.lower())


def load_tokenizer(name: str = None) -> callable:
    if name is None:
        return None
    elif name == "nltk-punct":
        return default_tokenizer
    elif name == "nltk-punct-lower":
        return default_tokenizer_lower
    elif name == "whitespace":
        return whitespace_tokenizer
    elif name == "nltk":
        return nltk.word_tokenize
    else:
        raise NotImplementedError(f"'{name}' is currently not supported...")


def load_embeddings_from_filepath(filepath: str) -> numpy.array:
    word2embeddings = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.split()
            word = line[0]
            embedding = numpy.array([float(e) for e in line[1:]], dtype=numpy.float32)
            word2embeddings[word] = embedding

    return word2embeddings
