# HW1: Semi-supervised Text Classification

You will need to download `speech.tar.gz` file from the Kaggle website, and put it in the `data` folder inside `hw1` (if you put it elsewhere, change the location in the code). You should be then able to run:

 ```
 python speech.py
 ```

 This will train a default logistic regression classifier, and save the output predictions in `data/speech-basic.csv`. If you like, you can upload this file to Kaggle, and make sure you are getting the same/similar performance as the benchmarks on Kaggle.

The current assignment description is available [here](http://sameersingh.org/courses/statnlp/wi17/assignments.html#hw1).

## Files

There are only two files in this folder:

    * `speech.py`: All the I/O related functionality. See the main function for how to read the training and dev data, how to train a classifier, how to read the unlabeled data, and how to save the output predictions to file. You should not really be modifying this file, but instead calling these functions from your code.

    * `classify.py`: Two simple methods to train and evaluate a classifier. You can either write all your code in this file, or create your different one with these methods copied over.
