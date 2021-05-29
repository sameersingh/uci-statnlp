import argparse
import json

import jsonlines
from rouge_score import rouge_scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file")
    parser.add_argument("--predictions_file")
    parser.add_argument("--output_file", required=False)
    args = parser.parse_args()

    # Load the Rouge metric
    metric = rouge_scorer.RougeScorer(['rougeL'])

    # Load the gold and predictions file
    gold_lines = {l['id'] : l for l in jsonlines.open(args.gold_file)}
    pred_lines = {l['id']: l for l in jsonlines.open(args.predictions_file)}

    scores = []
    for id in gold_lines:
        if id not in pred_lines:
            print(f'Could not find a generated summary for ID {id}. '
                  f'Assigning a score of 0 for this instance.')
            scores.append(0)
        else:
            score = metric.score(
                gold_lines[id]['summary'],
                pred_lines[id]['generated_summary']
            )['rougeL'].fmeasure
            scores.append(score)
            pred_lines[id]['rougeL'] = score

    print(f'Rouge-L score: {sum(scores)/len(scores)*100:.2f}')

    # If an output_file is provided, we write out the instance-wise
    # rouge scores to file
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for l in pred_lines.values():
                f.write(json.dumps(l, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
