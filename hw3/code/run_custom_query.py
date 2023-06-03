from data import ODQADataset, load_dataset
from run_eval import load_reader, load_retriever, print_sep


import argparse, json, os, tqdm


BASE_DIR = ".."


def print_sep(msg):
    print("=" * 80, msg, "=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=f"{BASE_DIR}/results",
        help="Directory to write the results",
        type=str,
    )
    parser.add_argument(
        "--datapath",
        default=f"{BASE_DIR}/data/bioasq.json",
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
        "--query",
        required=True,
        help="Query or semicolon-separated list of queries to execute.",
        type=str,
    )
    parser.add_argument(
        "--k",
        default=1,
        help="Number of documents to retrieve",
        type=int,
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # CLI arguments validation
    assert args.k > 0, "--k argument should be a positive integer"
    return args


if __name__ == "__main__":
    args = parse_args()

    print_sep("Conduct CUSTOM EXPERIMENT")
    print(args)
    dataset: ODQADataset = load_dataset(args.datapath)

    reader = load_reader(args.reader_filepath)
    retriever = load_retriever(args.retriever_filepath)

    print(f"Fitting {dataset.ndocuments} documents to retriever")
    retriever.fit(dataset.documents)

    predicted_answers = []
    retrieved_documts = []

    print_sep("Experiments")
    queries = args.query.split(";")
    # Note: You can specify multiple queries through the use of the colon
    # --query "example query 1; example query 2"
    print("\n".join(queries))

    results = []
    for query in tqdm.tqdm(queries):
        retr_docs, retr_scores = retriever.retrieve(query, args.k)
        answer = reader.find_answer(query, retr_docs)

        results.append(
            {
                "query": query,
                "answer": answer,
                "retrieved_docs": retr_docs,
            }
        )

    with open(f"{args.output_dir}/results.jsonl", "w", encoding="utf-8") as f:
        for l in results:
            f.write(json.dumps(l, ensure_ascii=False, sort_keys=True) + "\n")
