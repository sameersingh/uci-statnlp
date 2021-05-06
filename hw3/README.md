# HW3: Sequence Tagging on Tweets

You will need to download `data.tar.gz` file from the course website, and *uncompress* it into the `data` folder inside `hw3` (if you put it elsewhere, change the location in the code). 
You will need Python3 and the Python packages `torch` and `tqdm`. 

To train a model using a configuration file:
```
python train.py [JSON file path] -s [save point]
```

E.g., `SimpleTagger` model for POS
```
python train.py config/simple_tagger_pos.json -s out/simple_tagger_pos/
```
E.g., `NeuralCrf` model for POS
```
python train.py config/neural_crf_pos.json -s out/neural_crf_pos/
```

## Files

There are a few files in this folder:

### Files you should modify

* `viterbi.py`: General purpose interface to a sequence Viterbi decoder, which currently has an incorrect implementation. Once you have implemented the Viterbi implementation, running `python viterbi_test.py` should result in successful execution without any exceptions.

* `metric.py`: Contains metric classes used for evaluating models. Your job will be to complete the implementation of `AccuracyPerLabel.__call__()`. Once you have finished, running `python metric_test.py` should result in successful execution without any exceptions.

* `config/simple_tagger_{pos,ner}.json`: Configuration file for `simple_tagger` model for POS and NER.

* `config/neural_crf_{pos,ner}.json`: Configuration file for `neural_crf` model for POS and NER.

### Files you need not modify

* `simple_tagger.py`: Simple Tagger implementation.
* `neural_crf.py`: Neural CRF implementation working with `viterbi.py`.


* `metric_test.py`: Tests your implementation of `AccuracyPerLabel`.
* `viterbi_test.py`: Tests your implementation of Viterbi algorithm.


* `util.py`: Helper functions to load PyTorch objects.
* `dataset.py`: Code to load and tensorize data.


* `train.py` and `evaluate.py`: Code to train/evaluate model.

