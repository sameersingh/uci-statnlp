"""Python script that trains and evaluates neural models.

Methods
-------
parse_args() --> argparse.Args:
    Defines the command line arguments necessary to run the script.

learn_neural_model(
        data, model_configs, batch_size, sequence_len, num_epochs,
        seed, learn_rate, patience, checkpoints_dir
    ) --> neural.NeuralLM:
    Fits an LSTM model with configurations model_configs to the
    specified data for the specified epochs. Creating a temporary
    dir with the best checkpoints of each epoch.
"""
from data import Data, read_texts
from neural import NeuralLM
from neural_utils import load_object_from_dict
from utils import *

from time import time
from torch.utils.tensorboard import SummaryWriter
from typing import List

import argparse, copy, datetime, os, json, random, tqdm, functools
import numpy as np

import torch


BASE_DIR = ".."
BASE_DIR = "/home/kat/Projects/PhD/TA/uci-statnlp-private/hw2-langmodels-decoding"


def parse_args():
    # Usage example
    # $ python -m learn_neural_model --config_filepath ../configs/default.json --min_freq 10
    # Explaining: Running the model using the above command will fit a
    # simple LSTM using the model_configs and data_configs.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default=f"{BASE_DIR}/data/corpora.tar.gz",
        type=str,
        help="Path to the tar.gz file with the datasets.",
    )
    parser.add_argument(
        "--output_dir",
        default=f"{BASE_DIR}/results/neural",
        help="name of directory to write out trained language models.",
        type=str,
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        # default=f"{BASE_DIR}/configs/lstm.json",
        default=f"{BASE_DIR}/configs/lstm_w_embeddings.json",
        help="Configuration filepath with the model and data configs for "
        "the experiment.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="*",
        help="datasets to train models for. It defaults to using all datasets.",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=MIN_FREQ_DEFAULT,
        help="Mininum number of times a token should appear in"
        "the training set to be considered part of vocabulary.",
    )
    parser.add_argument(
        "--train",
        default=True,
        type=bool,
        help="use this flag to train the models.",
    )
    parser.add_argument(
        "--model_filepath",
        default=f"",
        type=str,
        help="use this flag to load a model. must be specified if not training.",
    )
    args = parser.parse_args()

    # -------------------------------------------------
    # Create output dir
    # -------------------------------------------------
    if args.datasets == "*":
        args.datasets = DATASETS
    else:
        assert args.datasets in DATASETS
        args.datasets = [args.datasets]
    # -------------------------------------------------
    # Read configs
    # -------------------------------------------------
    with open(args.config_filepath, encoding="utf-8") as f:
        args.configs = json.load(f)

    print_sep(f"[Experiment configs]\n{args}")
    print("Creating results directory:", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def init_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def get_splits(
    data: Data, preprocess_fn: callable, train_eval_fraction: float = None
) -> tuple:
    train_data = preprocess_fn(data.train)
    dev_data = preprocess_fn(data.dev)
    test_data = preprocess_fn(data.test)

    # Split train into train_eval
    if train_eval_fraction is None or train_eval_fraction == 1.0:
        return train_data, train_data, dev_data, test_data

    num_examples_train = int(np.round(len(train_data) * train_eval_fraction))
    train_data_ids = np.arange(len(train_data))
    np.random.shuffle(train_data_ids)

    return (
        train_data[:num_examples_train],
        train_data[num_examples_train:],
        dev_data,
        test_data,
    )


def log2tensorboard(
    writer: SummaryWriter,
    epoch: int,
    train_loss=None,
    dev_loss=None,
    text: dict = None,
    lr: float = None,
    seq_len: int=None,
    train_loss_by_step: List[float] = None,
    grads=None,
):
    if train_loss is not None:
        writer.add_scalar("loss_train", train_loss, epoch)
        writer.add_scalar("perplexity_train", np.exp(train_loss), epoch)

    if dev_loss is not None:
        writer.add_scalar("loss_val", dev_loss, epoch)
        writer.add_scalar("perplexity_val", np.exp(dev_loss), epoch)

    if lr is not None:
        writer.add_scalar("learning_rate", lr, epoch)

    if seq_len is not None:
        writer.add_scalar("seq_len", seq_len, epoch)

    if text is not None:
        writer.add_text("generated samples", str(text), epoch)

    if train_loss_by_step is not None:
        step = epoch * len(train_loss_by_step)
        for step_i, loss in enumerate(train_loss_by_step):
            writer.add_scalar("loss_by_step", loss, step+step_i)

    if grads is not None:
        step = epoch * len(grads)
        for step_i, grad_metadata in enumerate(grads):
            for norm, norm_values in grad_metadata.items():
                [writer.add_scalar(f"grads_{norm}/param_{idx}", val, step+step_i)
                 for idx, val in enumerate(norm_values)]


def persist_if_improved(
    output_dir: str,
    model: NeuralLM,
    new_loss: float,
    new_epoch: int,
    previous_loss: float,
    previous_epoch: int,
    early_stop_patience: int = 100,
):
    if new_loss >= previous_loss:
        return previous_loss, previous_epoch, ((new_epoch - (previous_epoch or 0)) > early_stop_patience)

    # Otherwise we want to remove previous best model
    if previous_epoch is not None:
        os.remove(f"{output_dir}/model_epoch_{previous_epoch}__base.pkl")
        os.remove(f"{output_dir}/model_epoch_{previous_epoch}__model.pkl")
    # ^Explanation of the code above: To avoid wasting too much disk, we eliminate
    # the previous best model, every time we find a new one. If you don't want this
    # behavior, please comment the three lines above...

    print(f"\t[{new_epoch}] Saving best model")
    model.save_model(f"{output_dir}/model_epoch_{new_epoch}.pkl")
    return new_loss, new_epoch, False


def bptt_regularization(seq_len: int, scale: int=5) -> int:
    p = np.random.uniform()
    loc = seq_len if p <= 0.95 else seq_len // 2
    return max(scale, int(np.round(np.random.normal(loc, scale))))

def learn_neural_model(
    data: Data,
    configs: dict,
    model_filepath: str,
    output_dir: str,
) -> NeuralLM:
    init_seed(configs.get("random_seed", 8273))

    checkpoint_dir = output_dir + "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Creating directory to dump checkpoints at", checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(checkpoint_dir)
    write = functools.partial(log2tensorboard, writer=writer)
    sample_text = functools.partial(
        sample, prefixes=PREFIXES, max_new_tokens=5, decoder=DECODERS.MULTINOMIAL
    )
    def get_lr(optimizer) -> float:
        return [param_group["lr"] for param_group in optimizer.param_groups][0]

    # Create the neural LM
    model = NeuralLM(
        vocab2idx=data.vocabulary, model_configs=configs["model"],
    )
    print("vocab:", model.vocab_size)

    # Create optimizer and learning rate scheduler
    # (if the loss does not reduce in the next patience epochs, lr decreases by half)
    train_configs = configs["training"]
    optimizer = load_object_from_dict(train_configs["optimizer"], params=model.parameters())
    lr_scheduler = load_object_from_dict(train_configs["scheduler"], optimizer=optimizer)

    # Preprocess the training and eval data
    train_data, train_val_data, dev_data, test_data = get_splits(
        data, model.preprocess_data, train_configs.get("train_eval_frac")
    )

    print_sep("BEFORE TRAINING LSTM (uniform initialization)")
    train_loss, dev_loss = model.evaluate(train_data), model.evaluate(train_val_data)
    text = sample_text(model)
    write(epoch=0, train_loss=train_loss, dev_loss=dev_loss, text=text)

    orig_seq_len = train_configs.get("seq_len", 64)
    best_loss, best_model_epoch = dev_loss, None
    for epoch in tqdm.tqdm(range(train_configs["num_epochs"])):
        # Step 0. Apply regularization
        if train_configs.get("apply_bptt_reg", False):
            seq_len = bptt_regularization(orig_seq_len)
        else:
            seq_len = orig_seq_len

        # Step 1. Fit model for one epoch
        model.fit_corpus(
            corpus=train_data,
            optimizer=optimizer,
            batch_size=train_configs.get("batch_size", 32),
            max_seq_len=seq_len,
            clip=train_configs.get("clip", 1),
            clip_mode=train_configs.get("clip_mode"),
        )

        # Step 2. Track down perplexity
        train_loss = model.evaluate(train_data)
        dev_loss = model.evaluate(train_val_data)

        lr = get_lr(optimizer)
        write(epoch=epoch + 1, train_loss=train_loss, dev_loss=dev_loss, lr=lr, seq_len=seq_len)
        write(epoch=epoch+1, train_loss_by_step=model.loss_by_step, grads=model.grad_metadata)
        lr_scheduler.step(dev_loss)

        if (epoch + 1) % train_configs["log_interval"] == 0:
            print_sep(f"After {epoch+1} Epoch:")
            text = sample_text(model)
            write(epoch=epoch + 1, text=text)

        best_loss, best_model_epoch, early_stopped = persist_if_improved(
            output_dir=checkpoint_dir,
            model=model,
            new_loss=dev_loss,
            new_epoch=epoch + 1,
            previous_loss=best_loss,
            previous_epoch=best_model_epoch,
            early_stop_patience=train_configs["early_stopping_patience"],
        )

        if early_stopped or lr < train_configs.get("early_stopping_min_lr", 1e-8):
            print(f"Stop the network after {epoch+1} techniques!")
            break

    # In case the last model is not the same as the best model, re-load the best one
    if best_model_epoch != (epoch + 1):
        print("\tLoading best model, from epoch", best_model_epoch)
        model = NeuralLM.load_model(f"{checkpoint_dir}/model_epoch_{best_model_epoch}.pkl")

    # evaluate on train, test, and dev
    print("PPL train:", model.perplexity(train_data + train_val_data))
    print("PPL dev  :", model.perplexity(dev_data))
    print("PPL test :", model.perplexity(test_data))

    print("Persisting Final checkpoint at", model_filepath)
    model.save_model(model_filepath)
    writer.close()

    return model


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    configs = args.configs

    datas = []
    models: List[NeuralLM] = []

    for dname in args.datasets:
        # . Load data
        print_sep(f"Training {dname}")
        data = read_texts(
            args.dataset_path,
            dname,
            tokenizer_kwargs={"lowercase": False},
            min_freq=args.min_freq,
        )
        datas.append(data)

        if args.train:
            # Dump configs to output directory
            output_dir_name = f"{output_dir}/{dname}"
            os.makedirs(output_dir_name, exist_ok=True)

            with open(f"{output_dir_name}/config.json", "w") as f:
                f.write(json.dumps(configs, indent=4))

            print_sep("Training model")
            neural_model = learn_neural_model(
                copy.deepcopy(data),
                copy.deepcopy(configs),
                output_dir_name + f"/{NeuralLM._NAME_}.pkl",
                output_dir_name,
            )
            models.append(neural_model)

    if not args.train:
        neural_model = NeuralLM.load_model(args.model_filepath, **configs["model"])
        models.append(neural_model)

    print_sep("Evaluation")
    start = time()
    evaluate_perplexity(args.datasets, datas, models, args.output_dir)
    end = time()
    print(f"Evaluation duration (min): {(end-start)/60:.2}")
