from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('informed_seq2seq_predictor')
class InformedSeq2SeqPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    :class:`~allennlp.models.encoder_decoder.informed_seq2seq` and
    :class:`~allennlp.models.encoder_decoder.copynet_seq2seq`.
    """

    def predict(self, source: str, extra: str) -> JsonDict:
        return self.predict_json({"source": source, "extra": extra})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "...", "extra": "..."}``.
        """
        source = json_dict["source"]
        extra = json_dict["extra"]
        return self._dataset_reader.text_to_instance(source, extra_seq=extra)
