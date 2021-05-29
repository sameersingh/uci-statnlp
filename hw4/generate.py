import argparse
import json
import random

import jsonlines
import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration

import decoders
from models import TransformerModel

random.seed(0)


def generate_summary(model, tokenizer, document, decoder):
    """ Generates a summary for a single document

    Parameters
    ----------
    model: ``BartForConditionalGeneration`` A BART model that has been
        fine-tuned for summarization
    tokenizer: ``BartForConditionalGeneration``: A corresponding BART tokenizer
    document: ``str`` A single document to be summarized
    decoder: ``str`` The decoder to use for decoding

    Returns:
    ----------
    summary: ``str`` A generated summary of the input document
    summary_score: ``float`` The log-probability score of the summary
    """
    input_ids = tokenizer(document, truncation=True, return_tensors='pt')['input_ids']
    metadata = {'input_ids': input_ids}
    model_wrapper = TransformerModel(model)

    if decoder == 'greedy':
        top_candidate = decoders.greedy_decoding(
            model=model_wrapper,
            max_length=50,
            eos_id=tokenizer.eos_token_id,
            decoded_ids=[tokenizer.bos_token_id],
            metadata=metadata
        )
    elif decoder == 'beam_search':
        top_candidate = decoders.beam_search_decoding(
            model=model_wrapper,
            beam_size=3,
            max_length=50,
            eos_id=tokenizer.eos_token_id,
            decoded_ids=[tokenizer.bos_token_id],
            metadata=metadata
        )[0]
    elif decoder == 'random':
        # Random sampling
        top_candidate = decoders.top_k_sampling(
            model=model_wrapper,
            top_k=int(1e9), # random sampling is top-K with large K
            temperature=1,
            max_length=50,
            eos_id=tokenizer.eos_token_id,
            decoded_ids=[tokenizer.bos_token_id],
            metadata=metadata
        )
    elif decoder == 'top_k':
        top_candidate = decoders.top_k_sampling(
            model=model_wrapper,
            top_k=3,
            temperature=0.5,
            max_length=50,
            eos_id=tokenizer.eos_token_id,
            decoded_ids=[tokenizer.bos_token_id],
            metadata=metadata
        )
    elif decoder == 'nucleus':
        top_candidate = decoders.nucleus_sampling(
            model=model_wrapper,
            top_p=0.2,
            max_length=50,
            eos_id=tokenizer.eos_token_id,
            decoded_ids=[tokenizer.bos_token_id],
            metadata=metadata
        )

    summary_ids = top_candidate.decoded_ids
    summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
    summary_score = top_candidate.score
    return summary, summary_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    parser.add_argument("--decoder",
                        choices=['greedy', 'beam_search', 'random', 'top_k', 'nucleus'])
    args = parser.parse_args()

    model_name = 'sshleifer/distilbart-xsum-1-1'
    model = BartForConditionalGeneration.from_pretrained(model_name).eval()
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Iterate through input file documents, generating summaries
    outputs = []
    for line in tqdm.tqdm(jsonlines.open(args.input_file)):
        summary, summary_score = generate_summary(model=model,
                                                  tokenizer=tokenizer,
                                                  document=line['document'],
                                                  decoder=args.decoder)

        outputs.append({'id': line['id'],
                        'generated_summary': summary,
                        'generated_summary_score': summary_score})

    # Write out the generated summaries to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for l in outputs:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
