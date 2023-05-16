from tqdm import tqdm
from data import textToTokens
from lm import LangModel
from ngram import Ngram
from ngram_interp import InterpNgram
from neural import NeuralLM
from decoders import DECODERS, generate_sentence

import argparse, json, os


BASE_DIR = ".."


def parse_args():
    # ------------------------------------------------------------------------------
    # note on specifying neural model filepath
    # If you've used the provided code to store the neural model you'll notice that
    # you won't find any model_path named "../results/neural/brown/neural.pkl
    # but instead you have a base path: ../results/neural/brown/neural__base.pkl
    # and a model path: ../results/neural/brown/neural__model.pkl
    # This separates the base wrapper class we created from the actual pytorch
    # model defined in neural_utils.LSTMWrapper.
    # To correctly load this model, you'd have to specify the option:
    # --model_filepath ../results/neural/brown/neural.pkl
    # (Note that we avoid the suffix "__base" and "__model", since this is done
    # on our behalf by the provided code)
    # -------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_filepath",
        default=f"{BASE_DIR}/results/neural/brown/neural.pkl",
        type=str,
        help="Filepath to trained neural model.",
    )
    parser.add_argument(
        "--output_dir",
        default=f"{BASE_DIR}/results/generations",
        type=str,
        help="Directory to place the results.",
    )
    parser.add_argument("--n", default=1, type=int, help="Number of sequences.")
    parser.add_argument(
        "--max_length", default=10, type=int, help="Maximum number of tokens to decode."
    )
    parser.add_argument(
        "--prompt",
        default="the department of",
        type=str,
        help="Prefix to use for generation.",
    )
    parser.add_argument(
        "--constraints_list",
        default="<unk>,the",
        type=str,
        help="List of tokens used in constrained decoding. Tokens should be comma-separated.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="The device to run the neural models on."
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.model_filepath):
        ValueError(f"No file exists at the specified location: {args.model_filepath}")

    if args.constraints_list is not None:
        args.constraints_list = args.constraints_list.split(",")

    return args


def load_model(model_filepath: str, device: str=None) -> LangModel:
    if "neural" in model_filepath:
        return NeuralLM.load_model(model_filepath, device)
    elif "interp" in model_filepath:
        return InterpNgram.load_model(model_filepath)
    else:
        return Ngram.load_model(model_filepath)


if __name__ == "__main__":
    args = parse_args()

    # -------------------------------------------------------------------------
    # Step 1. Load model from file
    # -------------------------------------------------------------------------
    model = load_model(args.model_filepath, args.device)

    # -------------------------------------------------------------------------
    # Step 2. Tokenize the prompt
    # -------------------------------------------------------------------------
    prompt = textToTokens(args.prompt) if args.prompt else [[]]
    print("Prompt (after default tokenization):", prompt)

    encoded_prompt = model.preprocess_data(prompt, add_eos=False)[0]

    # ngrams preprocessing of the data is done in terms of the words
    # however for decoding, we will deal with the vectorized representation
    # and therefore need to encode each word into their indices
    if model.is_ngram:
        encoded_prompt = [model.word2id(w) for w in prompt[0]]

    print("Decoded prompt:", encoded_prompt)
    # -------------------------------------------------------------------------
    # Step 3. Generate N sequences with each decoding algorithm
    # -------------------------------------------------------------------------
    for decoder in DECODERS:
        decoder_kwargs = {}
        output_filepath = f"{args.output_dir}/{decoder.name}.json"

        if decoder == DECODERS.CONSTRAINED:
            decoder_kwargs = {"constraints_list": args.constraints_list}

        # Greedy decoding always decodes to the same sequence
        n = 1 if decoder == DECODERS.GREEDY else args.n
        print(f"Generating {n} sequences with:", decoder.name)

        outputs = []
        for _ in tqdm(range(n)):
            output = generate_sentence(
                model, decoder, max_length=args.max_length, decoded_ids=encoded_prompt, **decoder_kwargs
            )
            outputs.append(output)
            print(f"[{decoder.name}] :{output}")

        # Step 4. Persist generated sequences by decoding algorithm
        with open(output_filepath, "w", encoding="utf-8") as f:
            for l in outputs:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
