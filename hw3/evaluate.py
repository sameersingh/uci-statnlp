import argparse
import json
from tqdm import tqdm
import torch
from dataset import TwitterDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("serialization_dir", help="save directory to load model")
    parser.add_argument("file", help="file to evaluate on")
    args = parser.parse_args()

    model = torch.load(f"{args.serialization_dir}/model.pt").eval()
    dataset = TwitterDataset(args.file)
    dataset.set_vocab(model.token_vocab, model.tag_vocab)
    dataloader = torch.utils.data.DataLoader(dataset, 1)

    for batch in tqdm(dataloader):
        _ = model(**batch)
    print(json.dumps(model.get_metrics(), indent=4))

if __name__ == "__main__":
    main()