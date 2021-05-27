import torch


class Model:
    pass


class TransformerModel(Model):
    def __init__(self, model):
        self._model = model

    def __call__(self, decoded_ids, metadata):
        model_inputs = {
            'input_ids': metadata['input_ids'],
            'past_key_values': metadata.get('past_key_values', None),
            'decoder_input_ids': torch.LongTensor([[decoded_ids[-1]]])
        }

        # Feed inputs through the Transformer model
        with torch.no_grad():
            model_outputs = self._model(**model_inputs)
            logits = model_outputs['logits'].squeeze()
            next_id_probs = torch.nn.functional.softmax(logits, dim=-1).tolist()

        # Update the `past_key_values` key. Used for fast decoding.
        metadata['past_key_values'] = model_outputs['past_key_values']

        # Structure outputs as a dictionary, then return
        outputs = {
            'next_id_probs': next_id_probs,
            'metadata': metadata
        }
        return outputs


class FixedModel(Model):
    def __init__(self, model):
        self._model = model

    def __call__(self, decoded_ids, metadata=None):
        time_step = len(decoded_ids)
        outputs = {'next_id_probs': self._model[time_step]}
        return outputs