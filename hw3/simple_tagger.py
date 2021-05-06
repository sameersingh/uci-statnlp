from typing import Dict
import torch

from dataset import Vocabulary
from metric import Accuracy, AccuracyPerLabel, Average
from util import load_embeddings, load_object_from_dict


class SimpleTagger(torch.nn.Module):
    def __init__(
        self,
        token_vocab: Vocabulary,
        tag_vocab: Vocabulary,
        embeddings: Dict,
        encoder: Dict,
        tag_projection: Dict
    ):
        super(SimpleTagger, self).__init__()
        self._embeddings = load_embeddings(**embeddings, token_vocab=token_vocab)
        self._encoder = load_object_from_dict(encoder)
        self._tag_projection = load_object_from_dict(tag_projection)

        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab
        self.num_tags = len(self.tag_vocab)
        assert self.num_tags == self._tag_projection.out_features

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.tag_vocab.pad_token_id)
        self.metrics = {
            'accuracy': Accuracy(),
            'accuracy_per_label': AccuracyPerLabel(self.num_tags, self.tag_vocab),
            'loss': Average()
        }

    def forward(self, token_ids, tag_ids=None) -> Dict:
        mask = token_ids != self.token_vocab.pad_token_id
        embeddings = self._embeddings(token_ids).permute(1, 0, 2)
        encoder_outputs = self._encoder(embeddings)[0] if self._encoder else embeddings
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        tag_logits = self._tag_projection(encoder_outputs)
        pred_tag_ids = torch.max(tag_logits, dim=-1)[1]

        output_dict = {
            'pred_tag_ids': pred_tag_ids,
            'tag_logits': tag_logits,
            'tag_probs': torch.nn.functional.softmax(tag_logits, dim=-1)
        }
        if tag_ids is not None:
            loss = self.loss(tag_logits.view(-1, self.num_tags), tag_ids.view(-1))
            output_dict['loss'] = loss
            self.metrics['accuracy'](pred_tag_ids, tag_ids, mask)
            self.metrics['accuracy_per_label'](pred_tag_ids, tag_ids, mask)
            self.metrics['loss'](loss.repeat(token_ids.size(0)))

        return output_dict

    def get_metrics(self, header='', reset=True):
        return {f'{header}{metric_name}': metric.get_metric(reset)
                for metric_name, metric in self.metrics.items()}