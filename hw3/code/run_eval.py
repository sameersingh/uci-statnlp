from data import ODQADataset, load_dataset
from evaluate import evaluate_reader, evaluate_retriever
from retriever import Retriever
from reader import Reader


import argparse, json, time, tqdm
import utils as ut


BASE_DIR = ".."


def print_sep(msg):
    print("=" * 80, msg, "=" * 80)


def load_retriever(filepath: str) -> Retriever:
    with open(filepath) as f:
        configs = json.load(f)

    tokenizer = ut.load_tokenizer(configs.pop("tokenizer", None))
    params = {} if tokenizer is None else {"tokenizer": tokenizer}
    retriever = ut.load_object_from_dict(configs, **params)
    return retriever


def load_reader(filepath: str) -> Reader:
    with open(filepath) as f:
        configs = json.load(f)

    reader = ut.load_object_from_dict(configs)
    return reader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        default=f"{BASE_DIR}/data/bioasq_dev.json",
        help="Filepath to the json file with the data.",
        type=str,
    )
    parser.add_argument(
        "--retriever_filepath",
        default=f"{BASE_DIR}/configs/rt_default.json",
        help="Path to the config file of the retriever",
        type=str,
    )
    parser.add_argument(
        "--reader_filepath",
        default=f"{BASE_DIR}/configs/rd_default.json",
        help="Path to the config file of the reader.",
        type=str,
    )
    parser.add_argument(
        "--reader_gold_eval",
        action="store_true",
        help="Specify this flag if you'd like to report the reader performance when using gold documents.",
    )
    parser.add_argument(
        "--k",
        default=10,
        help="Number of documents to retrieve",
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        help="Process queries in batches of 32 queries",
        type=int,
    )
    args = parser.parse_args()
    # CLI arguments validation
    assert args.k > 0, "--k argument should be a positive integer"
    return args


if __name__ == "__main__":
    args = parse_args()

    print_sep("Conduct default evaluation")
    print(args)

    dataset: ODQADataset = load_dataset(args.datapath)

    reader: Reader = load_reader(args.reader_filepath)
    if not args.reader_gold_eval:
        retriever: Retriever = load_retriever(args.retriever_filepath)

        print(f"Fitting {dataset.ndocuments} documents to retriever")
        start = time.time()
        retriever.fit(dataset.documents)
        print("Duration (min):", (time.time() - start) / 60)

    predicted_answers = []
    retrieved_documts = []

    print_sep("Evaluating ODQA Pipeline")
    start = time.time()

    for i in tqdm.tqdm(range(0, len(dataset.queries), args.batch_size)):
        queries = dataset.queries[i : i + args.batch_size]

        if args.reader_gold_eval:
            retr_docs = dataset.gold_documents[i : i + args.batch_size]
        else:
            retr_docs, retr_scores = retriever.retrieve(queries, args.k)
        retrieved_documts.extend(retr_docs)

        answers = reader.find_answer(queries, retr_docs)
        predicted_answers.extend(answers)

    print("Duration (min):", (time.time() - start) / 60)
    if not args.reader_gold_eval:
        retr_eval = evaluate_retriever(dataset.gold_documents, retrieved_documts)
        print(f"Retriever R@{args.k}: {retr_eval:.2%}")

    read_eval = evaluate_reader(dataset.gold_answers, predicted_answers)
    print(f"Reader Exact Match: {read_eval:.2%}")
