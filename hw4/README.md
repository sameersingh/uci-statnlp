# HW4: Phrase-Based Translation

You will need to download `data.tar.gz` file from the course website, and *uncompress* it into the `data` folder inside `hw4` (if you put it elsewhere, change the location in the code). You should be then able to run:

 ```
 python data.py
 ```

The current assignment description is available [here](http://sameersingh.org/courses/statnlp/wi17/assignments.html#hw4).

## Files

There are quite a few files in this folder:

* `lm.py`: Similar to the assignment in HW2, this code provides an implementation of a Trigram language model with Kneser-Ney smoothing, along with the parameters of such a model trained on a really large corpus of English documents. Note, since we are computing $P(f|e)P(e)$, we do not require a language model of French in order to perform the decoding. The format of the language model file, known as the ARPA model, consists of 1-, 2-, and 3-grams, with their log probabilities and backoff scores.
You can load and query the language model using the main function of this file.

* `phrase.py`: Code for the French to English phrase table. Each line in the file contains a pair of these phrases, along with a number of scores for different *features* of the pair. The code reads this file and computes the single score $g_p$ for each pair of phrases. This code also provides a handy method to get all the possible phrase translations for a given sentence, i.e. `phrases()` corresponds to `Phrases` in the pseudocode.
You can investigate the translation table as shown in the main function.

* `decoder.py`: Implementation of the multiple stack-based decoding algorithm.
This implementation attempts to follow the above notation of the pseudocode (and Collins' notes) as much as possible, deviating as needed for an optimized implementation.
The code implements a working monotonic decoder that does not take the language model into account.
This is especially important when you are looking at the code for finding compatible phrases (`Compatible`), computing the language model score (`lm_score`), and the distortion score (`dist_score`).
Some code that differs from the pseudocode includes precomputing the set of phrases that should be considered for position $r$ in `index_phrases` and extra fields in the state to make equality comparisons efficient (`key` in `State`). You will need to develop a reasonable understanding of this code, so please post privately or publicly on Piazza if you are not able to understand something.

* `submission.py`: Skeleton code for the submission. It contains the three types of decoders, out of which only the first one, `MonotonicDecoder`, works as intended. You have to implement the other functions in this skeleton.

* `data.py`: This is the code that reads in the files related to the translation model, reads French sentences from `test.fr`, corresponding English translations from `test.en`, runs the model on the French sentences, and computes the Bleu score on the predictions.
It also contains some simple words and phrases to translate into French, just to test your decoder.

* `bleu_score.py`: Code for computing the Bleu score for each translation/prediction pair (this code was ported from NLTK by Zhengli Zhao).
