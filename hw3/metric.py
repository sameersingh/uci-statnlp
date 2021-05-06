""" Code for computing different metrics one may be interested in

Class structure for `Metric()` is taken from the allennlp library.
"""

import torch

class Metric():
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self, predictions, gold_labels, mask=None):
        """
        predictions : `torch.Tensor`, required.
            A tensor of predictions.
        gold_labels : `torch.Tensor`, required.
            A tensor corresponding to some gold label to evaluate against.
        mask : `torch.Tensor`, optional (default = `None`).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        pass

    def get_metric(self, reset: bool):
        """ Compute and return the metric. Optionally also call `self.reset`. """
        pass

    def reset(self):
        """ Reset any accumulators or internal state. """
        pass

    @staticmethod
    def detach_tensors(*tensors):
        """ This method ensures the tensors are detached. """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)


class Average(Metric):
    """ A metric class which a list of scores and returns the average """
    def __init__(self):
        self.scores = []

    def __call__(self, scores):
        self.scores += scores

    def get_metric(self, reset=False):
        """ Returns the accumulated accuracy. """
        if len(self.scores) > 0:
            avg_score = sum(self.scores)/len(self.scores)
        else:
            avg_score = 0

        if reset:
            self.reset()

        return float(avg_score)

    def reset(self):
        self.scores = []


class Accuracy(Metric):
    """ A metric class which stores the accuracy of predictions. """
    def __init__(self):
        # keeps a running tab on the number of correct predictions
        self.correct_count = 0
        # keeps a running tab on the number of predictions
        self.total_count = 0

    def __call__(self, predictions, gold_labels, mask=None):
        """
        Parameters
        ----------
        predictions: ``torch.Tensor`` size: [batch_size, seq_len]
        gold_labels: ``torch.Tensor`` size: [batch_size, seq_len]
        mask: ``torch.Tensor`` size: [batch_size, seq_len]
        """
        assert predictions.size() == gold_labels.size()
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        if mask is None:
            mask = torch.ones(predictions.size())

        # Make the tensors 1 dimensional
        predictions = predictions.view(-1).tolist()
        gold_labels = gold_labels.view(-1).tolist()
        mask = mask.view(-1).tolist()

        for pred, gold, m in zip(predictions, gold_labels, mask):
            # if the current mask value has a 1
            if m == 1:
                self.total_count += 1
                # if gold label matches predicted label
                if pred == gold:
                    self.correct_count += 1

    def get_metric(self, reset=False):
        """ Returns the accumulated accuracy. """
        if self.total_count > 1e-12:
            accuracy = self.correct_count/self.total_count
        else:
            accuracy = 0

        if reset:
            self.reset()

        return float(accuracy)

    def reset(self):
        self.correct_count = 0
        self.total_count = 0


class AccuracyPerLabel(Metric):
    """ A metric class which stores the accuracy of predictions per class """
    def __init__(self, num_labels, label_vocab=None):
        self.num_labels = num_labels
        self.label_vocab = label_vocab

        # keeps a running tab on the number of correct predictions per label
        self.correct_count = {label_id: 0 for label_id in range(self.num_labels)}
        # keeps a running tab on the total number of times we see each gold label
        self.total_count = {label_id: 0 for label_id in range(self.num_labels)}

    def __call__(self, predictions, gold_labels, mask=None):
        """
        Parameters
        ----------
        predictions: ``torch.Tensor`` size: [batch_size, seq_len]
            A tensor of predictions.
        gold_labels: ``torch.Tensor`` size: [batch_size, seq_len]
            A tensor of the gold labels.
        mask: ``torch.Tensor`` size: [batch_size, seq_len]
            A tensor of 0's and 1's. Accuracy should only be computed on the
            positions has a value of 1. If the value is 0, ignore the
            gold label and predicted label at that position.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # If a mask isn't provided, make the mask all ones (i.e. compute
        # accuracy at all positions)
        if mask is None:
            mask = torch.ones(predictions.size())

        # Turn the tensors into a single list for easier computation
        predictions = predictions.view(-1).tolist()
        gold_labels = gold_labels.view(-1).tolist()
        mask = mask.view(-1).tolist()
        assert len(predictions) == len(gold_labels) == len(mask)

        # TODO: Update self.correct_count and self.total_count with counts
        # You should add to the values already stored there.
        # See the `Accuracy.__call__()` to see an example of this.

    def get_metric(self, reset=False):
        # accuracies per label
        accuracy = {}
        for label_id in self.total_count:
            if self.total_count[label_id] > 1e-12:
                accuracy[label_id] = float(self.correct_count[label_id]/self.total_count[label_id])
            else:
                accuracy[label_id] = 0

        if reset:
            self.reset()

        # Convert from label IDs to label names
        if self.label_vocab is not None:
            accuracy = {self.label_vocab.map_id_to_token(label_id): value
                        for label_id, value in accuracy.items()}

        return accuracy

    def reset(self):
        self.correct_count = {label_id: 0 for label_id in range(self.num_labels)}
        self.total_count = {label_id: 0 for label_id in range(self.num_labels)}
