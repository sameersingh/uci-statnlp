# HW3: Sequence Tagging on Tweets

You will need to download `data.tar.gz` file from the course website, and *uncompress* it into the `data` folder inside `hw3` (if you put it elsewhere, change the location in the code). You should be then able to run:

 ```
 python data.py
 ```

The current assignment description is available [here](http://sameersingh.org/courses/statnlp/wi17/assignments.html#hw3).

## Files

There are quite a few files in this folder:

* `tagger.py`: Code for two sequence taggers, logistic regression and CRF. Both of these taggers rely on `feats.py` and `feat_gen.py` to compute the features for each token. The CRF tagger also relies on `viterbi.py` to decode (which is currently incorrect), and on `struct_perceptron.py` for the training algorithm (which also needs Viterbi to be working).

* `feats.py` (and `feat_gen.py`): Code to compute, index, and maintain the token features. The primary purpose of `feats.py` is to map the boolean features computed in `feats_gen.py` to integers, and do the reverse mapping (if you want to know the name of a feature from its index). `feats_gen.py` is used to compute the features of a token in a sentence, which you will be extending. The method there returns the computed features for a token as a list of string (so does not have to worry about indices, etc.).

* `struct_perceptron.py`: A direct port (with negligible changes) of the structured perceptron trainer from the `pystruct` project. Only used for the CRF tagger. The description of the various hyperparameters of the trainer are available here, but you should change them from the constructor in `tagger.py`.

* `viterbi.py` (and `viterbi_test.py`): General purpose interface to a sequence Viterbi decoder in `viterbi.py`, which currently has an incorrect implementation. Once you have implemented the Viterbi implementation, running `python viterbi_test.py` should result in succesful execution without any exceptions.

* `data.py`: The primary entry point that reads the data, and trains and evaluates the tagger implementation.
