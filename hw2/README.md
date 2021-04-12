# HW2: Comparing Language Models

You will need to download `corpora.tar.gz` file from the course website, and put it in the `data` folder inside `hw2` (if you put it elsewhere, change the location in the code). You should be then able to run:

 ```
 python data.py
 ```

The current assignment description is available [here](https://canvas.eee.uci.edu/courses/22668/assignments/414255).

## Files

There are only four files in this folder:

* `lm.py`: This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. An implementation of a simple back-off based unigram model is also included, that implements all of the functions
of the interface.

* `generator.py`: This file contains a simple word and sentence sampler for any language model. Since it supports arbitarily complex language models, it is not very efficient. If this sampler is incredibly slow for your language model, you can consider implementing your own (by caching the conditional probability tables, for example, instead of computing it for every word).

* `data.py`: The primary file to run. This file contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language models (by calling `lm.py`), and generate sample sentences from all the models (by calling `generator.py`). It  saves the result tables into LaTeX files and writes out the trained language models into `saved_lms.pkl`. Note that each run overwrites the previously saved files, so be sure to keep track of which configurations are performing well.

* `demo.py`: This file contains the code to launch an interactive Streamlit demo with your trained language models in `saved_lms.pkl`. You will use this demo to sample text from your language model as well as to score text that you provide as input. To launch the demo, run `streamlit run demo.py`, then in your browser go to http://localhost:8501.

## Tabulate

The one *optional* dependency I have in this code is `tabulate` ([documentation](https://pypi.python.org/pypi/tabulate)), which you can install using a simple `pip install tabulate`.
This package is quite useful for generating the results table in LaTeX directly from your python code, which is a practice I encourage all of you to incorporate into your research as well.
If you do not install this package, the code does not write out the results to file (there's no runtime error).
