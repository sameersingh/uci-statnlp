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
from typing import List, Tuple

import argparse, copy, datetime, os, json, random, tqdm, functools
import numpy as np

import torch


BASE_DIR = ".."


def parse_args():
    # ------------------------------------------------------------------------------------------
    # Usage example for evaluation purposes:
    # ------------------------------------------------------------------------------------------
    # $ python -m learn_neural --model_dir ../results/neural --datasets brown
    # The command above will load the model available at ../results/neural/brown/neural.pkl
    # (note that this code expects to find in two pkl files in the directory, the "__base.pkl"
    # and the "__model.pkl" file).
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
        action="store_true",
        help="Use this flag to train the models.",
    )
    parser.add_argument(
        "--model_dir",
        default=f"{BASE_DIR}/results/neural",
        type=str,
        help="use this flag to load a model. Must be specified if not training.",
    )
    parser.add_argument("--device", default="cpu", type=str)
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

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.train:
        assert args.model_dir is not None, "Must define a path to the directory with models"

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

    train_ids = train_data_ids[:num_examples_train]
    train_eval_ids = train_data_ids[num_examples_train:]

    return (
        [train_data[i] for i in train_ids],
        [train_data[i] for i in train_eval_ids],
        dev_data,
        test_data,
    )


def log2tensorboard(
    writer: SummaryWriter,
    epoch: int,
    train_loss: Tuple[float, float]=None,
    dev_loss: Tuple[float, float]=None,
    text: dict = None,
    lr: float = None,
    seq_len: int = None,
    train_loss_by_step: List[float] = None,
    grads=None,
):
    if train_loss is not None:
        writer.add_scalar("loss_train_avg_token", train_loss[0], epoch)
        writer.add_scalar("loss_train_avg_sequence", train_loss[1], epoch)

        writer.add_scalar("perplexity_train", np.exp(train_loss[0]), epoch)

    if dev_loss is not None:
        writer.add_scalar("loss_val_avg_token", dev_loss[0], epoch)
        writer.add_scalar("loss_val_avg_sequence", dev_loss[1], epoch)
        writer.add_scalar("perplexity_val", np.exp(dev_loss[0]), epoch)

    if lr is not None:
        writer.add_scalar("learning_rate", lr, epoch)

    if seq_len is not None:
        writer.add_scalar("seq_len", seq_len, epoch)

    if text is not None:
        writer.add_text("generated samples", str(text), epoch)

    if train_loss_by_step is not None:
        step = epoch * len(train_loss_by_step)
        for step_i, loss in enumerate(train_loss_by_step):
            writer.add_scalar("loss_train_by_step", loss, step + step_i)

    if grads is not None:
        step = epoch * len(grads)
        for step_i, grad_metadata in enumerate(grads):
            for norm, norm_values in grad_metadata.items():
                [
                    writer.add_scalar(f"grads_{norm}/param_{idx}", val, step + step_i)
                    for idx, val in enumerate(norm_values)
                ]


def persist_if_improved(
    output_dir: str,
    model: NeuralLM,
    new_loss: float,
    new_epoch: int,
    previous_loss: float,
    previous_epoch: int,
    early_stop_patience: int = 100,
) -> Tuple[float, int, bool]:
    """Determines whether we should the new model is better than the previous.
    Removing the old one and persisting the new one if it is.
    """
    if new_loss >= previous_loss:
        return (
            previous_loss,
            previous_epoch,
            ((new_epoch - previous_epoch) > early_stop_patience), # early stopping
        )

    # Otherwise we want to remove previous best model
    if previous_epoch != 0:
        os.remove(f"{output_dir}/model_epoch_{previous_epoch}__base.pkl")
        os.remove(f"{output_dir}/model_epoch_{previous_epoch}__model.pkl")
    # ^Explanation of the code above: To avoid wasting too much disk, we eliminate
    # the previous best model, every time we find a new one. If you don't want this
    # behavior, please comment the three lines above...

    print(f"\t[{new_epoch}] Saving best model")
    model.save_model(f"{output_dir}/model_epoch_{new_epoch}.pkl")
    return new_loss, new_epoch, False


def setup(configs: dict, output_dir: str) -> str:
    """Setups the experiment, creating the checkpoint dir and init random seeds."""
    checkpoint_dir = (
        output_dir + "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    print("Creating directory to dump checkpoints at", checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # For reproducibility
    with open(f"{checkpoint_dir}/config.json", "w") as f:
        f.write(json.dumps(configs, indent=4))

    init_seed(configs.get("random_seed", 8273))
    return checkpoint_dir


def learn_neural_model(
    data: Data,
    configs: dict,
    model_filepath: str,
    output_dir: str,
    device: str=None,
) -> NeuralLM:
    """Fits a neural LM.

    It fits a neural LM while logging different aspects of the training
    during training, including greedy samples of the models, and non-
    -adjusted perplexity.
    """
    checkpoint_dir = setup(configs, output_dir)

    writer = SummaryWriter(checkpoint_dir)
    # syntactic sugar, avoid verbose calls later in the codes
    write = functools.partial(log2tensorboard, writer=writer)
    sample_text = functools.partial(
        sample, prefixes=PREFIXES, max_new_tokens=10, decoder=DECODERS.GREEDY
    )
    def get_lr(optimizer) -> float:
        return [param_group["lr"] for param_group in optimizer.param_groups][0]

    # -------------------------------------------------------------------------s
    # 1. Create the neural language model
    # -------------------------------------------------------------------------
    model = NeuralLM(
        vocab2idx=data.vocabulary,
        model_configs=configs["model"],
        device=device,
    )

    train_configs = configs["training"]
    print("vocab:", model.vocab_size)
    # -------------------------------------------------------------------------
    # Create optimizer and learning rate scheduler
    # -------------------------------------------------------------------------
    optimizer = load_object_from_dict(
        train_configs["optimizer"], params=model.parameters()
    )
    lr_scheduler = load_object_from_dict(
        train_configs["scheduler"], optimizer=optimizer
    )
    # -------------------------------------------------------------------------
    # Preprocess the training and eval data
    # -------------------------------------------------------------------------
    # We do not want to overfit to dev_data to make it a comparable model to
    # the ngram models, thus we use a fraction of the training data as the
    # train eval set for setting up the hyperparameters and defining early
    # stopping.
    train_data, train_val_data, dev_data, test_data = get_splits(
        data, model.preprocess_data, train_configs.get("train_eval_frac")
    )

    # -------------------------------------------------------------------------
    #                           TRAIN
    # -------------------------------------------------------------------------
    print_sep("BEFORE TRAINING LSTM (uniform initialization)")
    train_loss, dev_loss = model.evaluate(train_data), model.evaluate(train_val_data)
    # ^Note: evaluate returns (1) avg logprob by token and (2) avg logprob by sequence
    text = sample_text(model)
    write(epoch=0, train_loss=train_loss, dev_loss=dev_loss, text=text)

    best_loss, best_model_epoch, epoch = 10e10, 0, -1
    for epoch in tqdm.tqdm(range(train_configs["num_epochs"])):

        # Fit model for one epoch =============================================
        model.fit_corpus(
            corpus=train_data,
            optimizer=optimizer,
            batch_size=train_configs.get("batch_size", 32),
            max_seq_len=train_configs.get("seq_len", 64),
            clip=train_configs.get("clip", 0.25),
            clip_mode=train_configs.get("clip_mode"),
        )

        writer.add_scalar("running_avg_token_loss", model.running_loss, epoch+1)
        writer.add_scalar("running_avg_seq_loss", model.running_train_loss, epoch+1)

        # Track down loss =============================================
        train_loss = model.evaluate(train_data)
        dev_loss = model.evaluate(train_val_data)

        lr = get_lr(optimizer)
        write(
            epoch=epoch + 1,
            train_loss=train_loss,
            dev_loss=dev_loss,
            lr=lr
        )
        write(
            epoch=epoch + 1,
            train_loss_by_step=model.loss_by_step,
            grads=model.grad_metadata,
        )
        lr_scheduler.step(train_loss[0])

        # Sample text =============================================
        if (epoch + 1) % train_configs["log_interval"] == 0:
            print_sep(f"After {epoch+1} Epoch:")
            text = sample_text(model)
            write(epoch=epoch + 1, text=text)

        # Early stopping and best model ===========================
        best_loss, best_model_epoch, early_stopped = persist_if_improved(
            output_dir=checkpoint_dir,
            model=model,
            new_loss=train_loss[0],
            # ^Note: during training there was a mismatch between validation loss
            # and quality of the text generated by the LSTM model, hence we
            # decided, for the time being, to use training loss as the main
            # guide towards better LSTM-based models. This may lead to overfitting
            # but we conducted a few manual experiments to guarantee the model's
            # output were not verbatim of the training data.
            new_epoch=epoch + 1,
            previous_loss=best_loss,
            previous_epoch=best_model_epoch,
            early_stop_patience=train_configs["early_stopping_patience"],
        )

        if early_stopped or lr < train_configs.get("early_stopping_min_lr", 1e-8):
            print(f"Stop the network after {epoch+1} epochs!")
            break

    # -------------------------------------------------------------------------
    #                      POS-TRAINING LOAD BEST MODEL
    # -------------------------------------------------------------------------
    # Reload the best model (if it's not the last one)
    if best_model_epoch != (epoch + 1):
        print("\tLoading best model, from epoch", best_model_epoch)
        model = NeuralLM.load_model(
            f"{checkpoint_dir}/model_epoch_{best_model_epoch}.pkl"
        )

    # Evaluate perplexity on train, test, and dev (comparable results to ngram models)
    print("PPL train:", model.perplexity(train_data + train_val_data))
    print("PPL dev  :", model.perplexity(dev_data))
    print("PPL test :", model.perplexity(test_data))

    print("Persisting Final checkpoint at", model_filepath)
    model.save_model(model_filepath)
    writer.close()
    return model


if __name__ == "__main__":
    args = parse_args()

    # List of individual corpus and corresponding models
    datas: List[Data] = []
    models: List[NeuralLM] = []

    # Learn the models for each of the corpus, and evaluate them in-domain
    for dname in args.datasets:
        # Load data
        print_sep(f"Training {dname}")
        data = read_texts(args.dataset_path, dname, tokenizer_kwargs={"lowercase": False}, min_freq=args.min_freq)
        datas.append(data)

        if args.train:
            print_sep("Training model")
            output_dir_name = f"{args.output_dir}/{dname}"
            os.makedirs(output_dir_name, exist_ok=True)

            neural_model = learn_neural_model(
                copy.deepcopy(data),
                copy.deepcopy(args.configs),
                output_dir_name + f"/{NeuralLM._NAME_}.pkl",
                output_dir_name,
                device=args.device,
            )
            models.append(neural_model)
        else:
            # it looks for the models in a directory like /some/path/brown/neural.pkl
            model_filepath = f"{args.model_dir}/{dname}/{NeuralLM._NAME_}.pkl"
            neural_model = NeuralLM.load_model(model_filepath, device=args.device)
            models.append(neural_model)

    # Note: unlike the ngram model, since you're not supposed to train your
    # ngram model, we didn't make a flag --eval available to command line
    # You should be running this script only with the purpose of evaluating
    # the model.
    print_sep("Evaluation")
    start = time()
    evaluate_perplexity(args.datasets, datas, models, args.output_dir)
    end = time()
    print(f"Evaluation duration (min): {(end-start)/60:.2}")
    print("Done!")
