#!/bin/python

""" Runs a small test to check scores from AccuracyPerLabel Metric"""

def run_metric_test():
    from metric import AccuracyPerLabel
    import torch

    metric = AccuracyPerLabel(num_labels=5)

    predictions = torch.Tensor([[4, 1, 2, 0, 0],
                                [2, 0, 2, 1, 0],
                                [4, 2, 3, 4, 3]]
                               )
    labels = torch.Tensor([[0, 1, 3, 3, 2],
                           [4, 0, 2, 1, 0],
                           [4, 1, 0, 0, 2]])
    mask = torch.Tensor([[1, 1, 1, 0, 1],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0]])

    metric(predictions, labels, mask)

    your_scores = metric.get_metric()
    gold_scores = {0: 0.25, 1: 0.5, 2: 0.5, 3: 0.0, 4: 0.5}
    print(f'Your metric scores {your_scores}')
    print(f'Gold metric scores {gold_scores}')
    assert gold_scores == your_scores


if __name__ == "__main__":
    run_metric_test()
