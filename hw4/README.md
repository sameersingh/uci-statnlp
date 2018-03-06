HW4: Neural Machine Translation
===

The assignment description is available [here]().
In order to work on this assignment, you will need to download the `data.tar.gz` file from the course website.
By default we assume that the contents of this file will be extracted into the `hw4` folder.
If you choose to store the data in another location, make sure to update the appropriate parameters in the `config.yaml` file.


Dependencies
---

This code is Python 2.7 and Python 3.5+ compatible. You will also need the following libraries:

- PyTorch
- NLTK


Files
---

- `model.py`: Code for the neural machine translation model.
    Following the basic sequence-to-sequence structure, the model is comprised of two parts: an `Encoder` and a `Decoder`.
    `Encoder` embeds words from the source sentence and then feeds these embeddings through an RNN.
    `Decoder` generates the target sentence using another LSTM, whose initial hidden state is the final hidden state output by the `Encoder`.

- `config.yaml`: Model configuration parameters.

- `train.py`: Model training script.
    To use it, run: `python train.py --config config.yaml`.

- `evaluate.py`: Script for evaluating the model output.
    Measures the BLEU score of the translations as provides a few sample outputs from the model.
    To use it, run: `python evaluate.py --config config.yaml`.

- `utils.py`: Utilities for working with text data.
    Contains the code used to build the word vocabularies, as well as for generating data from the datasets.

