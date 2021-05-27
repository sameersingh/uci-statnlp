import argparse
import json

import jsonlines
import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration

import decoders
from models import TransformerModel


def generate_summary(model, tokenizer, document):
    """ Generates a summary for a single document

    Parameters
    ----------
    model: ``BartForConditionalGeneration`` A BART model that has been
        fine-tuned for summarization
    tokenizer: ``BartForConditionalGeneration``: A corresponding BART tokenizer
    document: ``str`` A single document to be summarized

    Returns:
    ----------
    summary: ``str`` A generated summary of the input document
    summary_score: ``float`` The log-probability score of the summary
    """
    input_ids = tokenizer(document, truncation=True, return_tensors='pt')['input_ids']
    metadata = {'input_ids': input_ids}
    model_wrapper = TransformerModel(model)

    # TODO: Modify this function call to change the decoding algorithm used.
    #  Be sure not to remove any of the parameters (e.g. model, eos_id,
    #  etc.) since these are all needed. Just added necessary
    #  parameters (e.g. beam_size for beam search decoding).
    # NOTE: For beam_search_decoding, we get a list out, so you will have to
    #  set top_candidate to the first element of the returned list.
    top_candidate = decoders.greedy_decoding(
        model=model_wrapper,
        max_length=51,
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
    args = parser.parse_args()

    model_name = 'sshleifer/distilbart-xsum-1-1'
    model = BartForConditionalGeneration.from_pretrained(model_name).eval()
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Iterate through input file documents, generating summaries
    outputs = []
    for line in tqdm.tqdm(jsonlines.open(args.input_file)):
        summary, summary_score = generate_summary(model=model,
                                                  tokenizer=tokenizer,
                                                  document=line['document'])

        outputs.append({'id': line['id'],
                        'generated_summary': summary,
                        'generated_summary_score': summary_score})

    # Write out the generated summaries to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for l in outputs:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
