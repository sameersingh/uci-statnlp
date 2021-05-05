from typing import Dict
import torch
import torch.nn as nn

from dataset import Vocabulary
from metric import Accuracy, AccuracyPerLabel, Average
from util import load_embeddings, load_torch_object


class SimpleTagger(nn.Module):
    def __init__(self,
        num_tags: int,
        token_vocab: Vocabulary,
        tag_vocab: Vocabulary,
        embeddings: Dict,
        encoder: Dict,
        tag_projection: Dict
    ):
        super(SimpleTagger, self).__init__()
        # in addition to loading embeddings, update the vocabulary word list
        embeddings['word_list'] = token_vocab.word_list
        self._embeddings, token_vocab.word_list = load_embeddings(**embeddings)
        self._encoder = load_torch_object(encoder)
        self._tag_projection = load_torch_object(tag_projection)

        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.num_tags = num_tags
        self.loss = nn.CrossEntropyLoss(ignore_index=num_tags) # ignore pad
        self.metrics = {
            'accuracy': Accuracy(),
            'accuracy_per_label': AccuracyPerLabel(self.num_tags, self.tag_vocab),
            'loss': Average()
        }

    def forward(self, token_ids, tag_ids=None) -> Dict:
        embeddings = self._embeddings(token_ids).permute(1, 0, 2)
        encoder_outputs = self._encoder(embeddings)[0] if self._encoder else embeddings
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        tag_logits = self._tag_projection(encoder_outputs)
        pred_tag_ids = torch.max(tag_logits, dim=-1)[1]

        output_dict = {
            'pred_tag_ids': pred_tag_ids,
            'tag_logits': tag_logits,
            'tag_probs': nn.functional.softmax(tag_logits, dim=-1)
        }
        if tag_ids is not None:
            loss = self.loss(tag_logits.view(-1, self.num_tags), tag_ids.view(-1))
            output_dict['loss'] = loss

            # compute accuracy on non-pad tokens
            mask = tag_ids != self.num_tags
            self.metrics['accuracy'](pred_tag_ids, tag_ids, mask)
            self.metrics['accuracy_per_label'](pred_tag_ids, tag_ids, mask)
            self.metrics['loss'](loss.repeat(token_ids.size(0)))

        return output_dict

    def get_metrics(self, header='', reset=True):
        return {f'{header}{metric_name}': metric.get_metric(reset)
                for metric_name, metric in self.metrics.items()}