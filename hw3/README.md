# HW3: Sequence Tagging on Tweets

You will need to download `data.tar.gz` file from the course website, and *uncompress* it into the `data` folder inside `hw3` (if you put it elsewhere, change the location in the code). Once you activate `allennlp` (e.g., `conda activate allennlp`), you should be then able to run:

```
allennlp train [JSON file path] -s [save point file path] --include-package neural_crf
```

E.g., `simple_tagger` model for POS
```
allennlp train ./config/simple_tagger_pos.json -s ./model/simple_tagger.pt
```
E.g., `neural_crf` model for POS
```
allennlp train ./config/neural_crf.json -s ./model/neural_crf_pos.pt --include-package neural_crf
```

The current assignment description is available [here](https://canvas.eee.uci.edu/courses/14385/assignments/270636).

## Files

There are a few files in this folder:

### Files you should modify

* `viterbi.py`: General purpose interface to a sequence Viterbi decoder, which currently has an incorrect implementation. Once you have implemented the Viterbi implementation, running `python viterbi_test.py` should result in successful execution without any exceptions.

* `config/simple_tagger_{pos,ner}.json`: Configuration file for `simple_tagger` model for POS and NER.

* `config/neural_crf_{pos,ner}.json`: Configuration file for `neural_crf` model for POS and NER.

### Files you need not modify

* `neural_crf.py`: Neural CRF implementation working with `viterbi.py`.

* `viterbi_test.py`: Code to test your implementation of Viterbi algorithm.
