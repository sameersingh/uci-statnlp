HW4: Generating Function Descriptions
===

The assignment description is available [here](https://canvas.eee.uci.edu/courses/14385/assignments/270638).
In order to work on this assignment, you will need to download the `hw4-data.tar.gz` file from the course website.
By default we assume that the contents of this file will be extracted into the `data` folder. You will also need to install AllenNLP.

## Reference Commands

To train a model, you should be able to directly run:
```
allennlp train config/naive_{java,python} -s models/naive_{java,python} --include-package informed_seq2seq --include-package informed_seq2seq_reader
```

To serve a demo after training, you need to run:
```
python -m allennlp.service.server_simple --archive-path [MODEL_DIR]/model.tar.gz --predictor informed_seq2seq_predictor --include-package informed_seq2seq --include-package informed_seq2seq_reader --include-package informed_seq2seq_predictor --title "Code to Text" --field-name source --field-name extra
```

## Files

There are a few files in this folder:

### Files you should modify

* `viterbi.py`: General purpose interface to a sequence Viterbi decoder, which currently has an incorrect implementation. Once you have implemented the Viterbi implementation, running `python viterbi_test.py` should result in successful execution without any exceptions.

* `informed_seq2seq.py`: The main model file containing an implementation of the \textbf{naive} sequence to sequence model, simlar to the default sequence decoder in AllenNLP available [here](https://github.com/allenai/allennlp/blob/master/allennlp/models/encoder_decoders/simple_seq2seq.py). 
We have included additional hooks into the code for supporting the "informed" version, however the implementation of the extra part is mostly incomplete; the additional hooks exist to support the extra column in the data, the extra fields in the configuration, etc. This is the only Python file you need to change.
* `naive_{java,python}.json`: The configuration files for training and running the naive sequence to sequence model, on respective language.
These configuration files "turn off" the extra embedding part, and run the model in `informed_seq2seq.py`.
* `informed_{java,python}.json`:
  Configuration files that "enable" the extra encoder/embedding part, which, if your implementation is correct, should provide much higher gains. The command to run it will be identical to the one above, with a different configuration file.

### Files you need not modify

* `informed_seq2seq_reader.json`: provides a _reader_ to read the the data
* `informed_seq2seq_predictor.json`: set up a _predictor_ that will be used to set up the demo.
