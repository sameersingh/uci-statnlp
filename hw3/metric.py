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
        self.reset()

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
        self.reset()

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
        predictions = predictions.view(-1)
        gold_labels = gold_labels.view(-1)
        mask = mask.view(-1)

        # all positions where the prediction is correct
        correct = predictions == gold_labels
        # use mask to make all masked positions incorrect
        correct = correct*mask

        self.correct_count += torch.sum(correct)
        # only count positions that aren't masked
        self.total_count += torch.sum(mask)

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
        self.reset()

    def __call__(self, predictions, gold_labels, mask=None):
        """
        Parameters
        ----------
        predictions: ``torch.Tensor`` size: [batch_size, seq_len]
        gold_labels: ``torch.Tensor`` size: [batch_size, seq_len]
        mask: ``torch.Tensor`` size: [batch_size, seq_len]
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        if mask is None:
            mask = torch.ones(predictions.size())

        # Make the tensors 1 dimensional
        predictions = predictions.view(-1).tolist()
        gold_labels = gold_labels.view(-1).tolist()
        mask = mask.view(-1).tolist()

        # TODO: Update self.correct_count and self.total_count with counts


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
        # the number of correct predictions per label
        self.correct_count = {label_id: 0 for label_id in range(self.num_labels)}
        # the total number of times we see each gold label
        self.total_count = {label_id: 0 for label_id in range(self.num_labels)}
