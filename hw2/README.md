# Homework 2 - Spring 2023 - Language Modeling and Decoding Algorithms

# Table of Contents

In this homework, we will ask you to explore different language modeling techniques, namely, statistical and neural language modeling. We will also ask you to implement and evaluate different decoding algorithms.
This README is organized as follows:

1. [Setup and Dependencies](#setup)
2. [Tasks](#tasks)
3. [Code structure](#code-structure)
4. [Testing your implementation](#verifying-your-implementation)

## Setup

The code is setup to work with [Python 3.9](https://www.python.org/downloads/release/python-390/). 
The current version of the code requires installing some packages, including `scikit-learn`, `tqdm`, `PyTorch`, `jsonlines`, `numpy`, `tensorboard`, and `tabulate`.
Please refer to the Section [Required Dependencies](#required-dependencies) for more information on the version of these packages.

When running the code, we recommend using a virtual environment framework like [Anaconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html) or [Virtual env (Venv)](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). 
If you're familiar with this, feel free to jump to the next section on [Homework Tasks](#Tasks) or to [Code structure](#code-structure) if you want to know more about how the code is organized.

### Setting up your environment using Anaconda

In this section, we guide you through the commands necessary to successfully run the code using Anaconda. 
For information on how to setup Anaconda, check the [Official Anaconda Install Website](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html).

After installing Anaconda, you can either setup the environment automatically by compiling all the dependencies into a YAML file or you can setup up the environment manually. 
Manually setting up your environment, requires you to first create a virtual environment using the command below (replace `<name_env>` with your desired name): 

```shell
conda create -n <name_env> python=3.9
```

After creating the environment, activate it using the command:
```shell
conda activate <name_env>
```

Install all the dependencies using the command `conda install -c <channel> <package1> ... <packagen>`, where `<channel>` is the repository to look for the packages and `<package1>` and `<packagen>` are meant to be replaced by packages listed in the next section. For example, `conda install -c conda-forge scikit-learn tqdm jsonlines tabulate tensorboard`. Install PyTorch by following the instructions in the [Start Locally Offical pytorch](https://pytorch.org/get-started/locally/) page. In your specific setup, we used the command `conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia` but this should differ slightly depending on your hardware.

### Required dependencies

For reference, this code was runned using the dependencies below. You can try to upgrade some of these, but do it at your own risk!

```
- python==3.9.13
- pip==23.0.1
- joblib==1.2.0
- jsonlines==3.1.0
- numpy==1.23.3
- scikit-learn==1.2.2
- scipy==1.9.3
- tabulate==0.9.0
- torch==2.0.0
- tqdm==4.64.1
```

## Tasks

In this homework you will implement and evaluate n-gram models, comparing their performance against neural language modeling techniques. You will also implement several state-of-the-art decoding algorithms, which can be thought as functions of the next-word probability distribution obtained from each language model. Using different decoding algorithms, you can achieve text that is more diverse, while also being grammatically correct.

### Data 

Throughout the homework we will be using the same three corpora to conduct the analysis and training of the language models.
These are compiled and compressed in the file `./data/corpora.tar.gz` in this repository. 
The corpora constitutes three datasets **Brown**, **Reuters**, and **Gutenberg**.
Find more information about the datasets in the handout.

**Note**: In the `data` directory, you will also find other files that you can use to quickstart your analysis.
The files `<DATASET_NAME>_prompts.json` is a JSON file with sequences of words (and their corresponding counts) that occur with the highest frequency in the training set of each of the datasets.
Other sequences may exist and we encourage you to use sequences that are interesting in your analysis, describing them in your report.

The files `<DATASET_NAME>_constraints.jsonl` represent some combinations of prompts and constraints that you can use to get a sense of how the constrained decoding algorithm works. 
`<DATASET_NAME>` should be replaced by one of the following `brown`, `reuters`, or `gutenberg`.

---

### Part 1. Ngram language models

The **first milestone** in this part is to implement the n-gram with additive-k-smoothing algorithm in the `./code/ngram.py` file, as well as the more advanced version that uses linear interpolation and backoff in the `./code/ngram_interp.py` file. 
In each of these files, we provide you a partial implementation of `Ngram` and `InterpNgram` and you are required to implement a single method `cond_logprob`.

The **second milestone** is to conduct training and evaluation of these ngram and interpolated ngram models.
After training a few models for each of the datasets, you have to report the train and test perplexity in both in-domain (i.e., model is trained and evaluated on samples from the same dataset) and out-of-domain (i.e., model is trained on train set of corpus X and evaluated on eval set of corpus Y).
We provide you a python script `learn_ngram.py` that you can use for this task, if you so find useful!

**Note**: To better understand the trade-offs between neural and statistical language modeling, we provide you pretrained neural models (that you should download from [this google drive folder](https://drive.google.com/drive/folders/1VYCoWymLhLCiv9DpXT68em3SZdENaDsU?usp=share_link)), comparing their perplexity with those of the n-gram models.

----

### Part 2. Decoding Algorithms 

The second part of this assignment concerns the implementation and evaluation of several decoding algorithms.

The **first milestone** is to implement three decoding algorithms, namely _top-p_ (or nucleus sampling), _top-k_, and _constrained decoding_. You will find four methods in the file `decoders.py` that raise an `NotImplementedError` exception. These are the ones you will have to complete. 
To give you a better idea of the potential of constrained decoding we ask you to implement two variants of such decoding algorithm: the one where no token in the generated sequence is repeated and another one that receives a list of words which shouldn't appear in the generation. 
Effectively, this list works as a blacklist of terms that cannot be generated. 

The **milestone** is to conduct an analysis and comment on the observed results. 
This evaluation should be based on already pretrained models, so we suggest you use two or three n-gram models that achieved strong results in part 1.
For neural models, we ask you to use the provided models. 
Try to use decoding algorithms with both n-gram and neural models.
What do you observe? Are the decoding algorithms equally useful in both models? Why is that? 
How are the generations of neural models compared with those of the n-gram models?
How different are the generations of different decoding algorithms for the same model?


## Code Structure

The repository is organized in terms of three main set of files:
1. `learn_{X}.py` scripts: scripts that train and evaluate the perplexity of models `X`, persisting them in disk.
2. `generate_{X}.py` scripts: scripts that using the models persisted in 1., generate sequences using different decoding algorithms.
3. Other files concern the  base classes to make the code work. For example, the classes that define the language modeling behavior are `ngram.py`, `ngram_interp.py`, `neural.py`, whereas the `decoders.py` contains the different decoding algorithms. 

In summary, you'll have to implement code in the following files: `ngram.py`, `ngram_interp.py`, and `decoders.py`.
You can run the scripts `learn_{X}.py` to train and evaluate these models' perplexity or the `generate.py` to generate some sequences using different decoding algorithms.

### Fine-grained description of each folder

For those of you interested in getting more fine-grained description of the files:

- `data.py`: constitutes methods for loading the data, the most useful being `read_texts`. You can use its tokenizer_kwargs parameter to adapt the properties of the tokenization that is being applied by default to the tokenizer.
- `lm.py`: contains the definition for the base class `LanguageModeling`
- `ngram.py` (and `ngram_interp.py`): defines the classes for `Ngram` (and `InterpolatedNgram`). You will have to implement the methods `cond_logprob` for both classes.
- `neural_utils.py`: defines one LSTM wrapper for language modeling that leverages word embeddings. It also contains a `LMDataset` and methods related to data loading/processing. While not optimal, we decided to include it in this file to avoid having more files.
- `decoders.py`: implements different decoding algorithms in function of a `Candidate` class, including greedy decoding. You will have to implement the other decoding algorithms such as `top-k`, `nucleus sampling`, and `constrained decoding`.


### Running the scripts

`learn_ngram.py`: loads the data, trains one ngram-model `ngram_size` for each of the datasets, persisting them in the specified `output_dir`. 
It also evaluates the in-domain x out-of-domain perplexity of the different models when evaluated across the different corpora. 
You can run this script with the command: 

```shell
python -m learn_ngram.py --output_dir <OUTPUT_DIR> --ngram_size <NGRAM_SIZE> --min_freq <MIN_FREQ>
```

`learn_neural.py`: loads the data, trains one LSTM-based model for each of the datasets, persisting them in the specified `output_dir`. 
It also evaluates the in-domain x out-of-domain perplexity of the different models when evaluated across the different corpora. 
You can run this script via the command line using:

```shell
python -m learn_neural.py --output_dir <OUTPUT_DIR> --model_configs configs/default.json
```

`generate.py`: loads an ngram model from the specified path and generates the specified number of sequences with each decoding algorithm, dumping the resulting sequences in the specified `OUTPUT_DIR`. To facilitate conducting analysis with different lexical constraints we provide the command argument `CONSTRAINTS_LIST`. which should be expressed in terms of a list of tokens separated by comma. The constraint list `"the,of,and"` will be converted into the list `["the", "of", "and"]` and fed into the `constrained_decoding`, which should avoid sampling any of these words. 
You can run this file with  the command: 

```shell
python -m generate --model_filepath <MODEL_FILEPATH> --output_dir <OUTPUT_DIR> --n <NUM_SEQUENCES> --prompt <PROMPT> --constraints_list <CONSTRAINTS_LIST>
```



## Verifying your implementation

In order to ground your implementation, we report of the achieved perplexity using varying n-gram models when using the `learn_ngram.py` script, in these logs you can find the configuration of the specified script.

This section is organized as follows:

- [Starting small with python tests](#testing-your-implementation-with-tests)

- [Unigram model with add-0.2 smoothing](#example-of-execution-log-of-unigram-model-with-add-02-smoothing)

- [Bigram model with add-1 smoothing](#example-of-execution-log-of-bigram-model-with-add-1-smoothing)

- [Trigram model with add-1 smoothing](#example-of-execution-log-of-trigram-model-with-add-1-smoothing)

- [Trigram model with interpolation and backoff](#example-of-execution-log-of-trigram-model-with-interpolation-and-backoff)

Besides the non-adjusted perplexities, we also include the corresponding python command and the resulting log. 
You can observe the greedy generations of the different models when provided different prefixes (or prompts).


### Running tests

In order to run the tests in the folder [./tests](./tests), you will need to install `pytest`.
Having it installed, you can run the tests by executing the command `pytest -k .` in the folder `./tests`. 
We will iteratively add more tests to validate your implementation. 
Note that these tests are meant to give you an idea of what to expect but are not comprehensive or do not replace proper unit tests.


### Example of execution log of unigram model with add-0.2 smoothing

Python command:
```shell 
python -m learn_ngram --ngram_size 1 --min_freq 2 --llambda 0.2
```

The resulting non-adjusted perplexities obtained using the code in `learn_ngram.py`, were:

| Data split | Brown    | Reuters  | Gutenberg |
| ---------- | -------- | -------- | --------- |
| Train PPL  | 1258.98  | 1329.68  | 912.19    |
| Dev PPL    | 1102.81  | 1213.00  | 866.97    |

Consider the full log below in case you'd like to check the configuration and/or the generations of the resulting models.

```
Creating results directory: ../results/ngram

================================================================================

[Experiment Config]:
 Namespace(dataset_path='../data/corpora.tar.gz', output_dir='../results/ngram', use_interp=False, eval=False, ngram_size=1, alpha=0.8, llambda=0.2, min_freq=2, datasets=['brown', 'reuters', 'gutenberg'])
================================================================================

================================================================================
Training brown
================================================================================
brown  read. Num words:
-> train: 39802 
-> dev: 8437 
-> test: 8533
vocab: 23985
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 1258.9847135673383
[PPL dev]  : 1102.8171223220381
[PPL test] : 1096.3532777001842
Training duration (min): 0.039

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.489561362080813, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.489561362080813, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.489561362080813, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.489561362080813, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', 'lord', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.489561362080813, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/brown__1-gram.pkl

================================================================================
Training reuters
================================================================================
reuters  read. Num words:
-> train: 38183 
-> dev: 8083 
-> test: 8199
vocab: 21774
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 1329.6813146814832
[PPL dev]  : 1212.9957916019607
[PPL test] : 1222.687297707015
Training duration (min): 0.056

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -12.956368691826995, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -12.956368691826995, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'the', 'the', 'the', 'the'], 'seq_log_prob': -12.956368691826995, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -12.956368691826995, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', '<unk>', 'the', 'the', 'the', 'the'], 'seq_log_prob': -12.956368691826995, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/reuters__1-gram.pkl

================================================================================
Training gutenberg
================================================================================
gutenberg  read. Num words:
-> train: 68767 
-> dev: 14667 
-> test: 14861
vocab: 25458
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 912.1859985871547
[PPL dev]  : 866.9728793444675
[PPL test] : 852.5457482691778
Training duration (min): 0.086

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.62569714991939, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.62569714991939, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.62569714991939, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.62569714991939, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', 'lord', 'the', 'the', 'the', 'the'], 'seq_log_prob': -11.62569714991939, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/gutenberg__1-gram.pkl
Done!
```

### Example of execution log of bigram model with add-1 smoothing

Python command:
```shell 
python -m learn_ngram --ngram_size 2 --min_freq 2 --llambda 1
```

| Data split | Brown    | Reuters  | Gutenberg |
| ---------- | -------- | -------- | --------- |
| Train PPL  | 2806.47  | 1454.83  | 1540.87   |
| Dev PPL    | 3421.91  | 1706.49  | 1830.19   |

Consider the full log below in case you'd like to check the configuration and/or the generations of the resulting models.

```
Creating results directory: ../results/ngram

================================================================================
[Experiment Config]:
 Namespace(dataset_path='../data/corpora.tar.gz', output_dir='../results/ngram', use_interp=False, eval=False, ngram_size=2, alpha=0.8, llambda=1.0, min_freq=2, datasets=['brown', 'reuters', 'gutenberg'])
================================================================================

================================================================================
Training brown
================================================================================
brown  read. Num words:
-> train: 39802 
-> dev: 8437 
-> test: 8533
vocab: 23985
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 2806.4703984769167
[PPL dev]  : 3421.9127053038337
[PPL test] : 3406.1732657056214
Training duration (min): 0.049

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', '<unk>', '<eos>'], 'seq_log_prob': -8.784823708081035, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', '<unk>', '<eos>'], 'seq_log_prob': -8.784823708081035, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', '<eos>'], 'seq_log_prob': -6.201185078090051, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', '<unk>', '<eos>'], 'seq_log_prob': -8.784823708081035, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', 'lord', 'of', 'the', '<unk>', '<eos>'], 'seq_log_prob': -18.176985511809487, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/brown__2-gram.pkl

================================================================================
Training reuters
================================================================================
reuters  read. Num words:
-> train: 38183 
-> dev: 8083 
-> test: 8199
vocab: 21774
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 1454.8255644480262
[PPL dev]  : 1706.4926788163161
[PPL test] : 1702.7426059031484
Training duration (min): 0.065

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', 'company', 'said', '<eos>'], 'seq_log_prob': -11.383874600614911, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', 'company', 'said', '<eos>'], 'seq_log_prob': -11.383874600614911, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'countries', '<eos>'], 'seq_log_prob': -11.628411794067834, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', 'company', 'said', '<eos>'], 'seq_log_prob': -11.383874600614911, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>'], 'seq_log_prob': -14.880939049096234, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/reuters__2-gram.pkl

================================================================================
Training gutenberg
================================================================================
gutenberg  read. Num words:
-> train: 68767 
-> dev: 14667 
-> test: 14861
vocab: 25458
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 1540.8736498113267
[PPL dev]  : 1830.1926171635857
[PPL test] : 1792.4019490403355
Training duration (min): 0.1

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -9.169997535796973, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -9.169997535796973, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', '<eos>'], 'seq_log_prob': -5.4639423368681035, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -9.169997535796973, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', 'lord', 'the', 'LORD', '<eos>'], 'seq_log_prob': -13.931410379530053, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/gutenberg__2-gram.pkl
Done!

```


### Example of execution log of trigram model with add-1 smoothing

Python command:
```shell 
python -m learn_ngram --ngram_size 3 --min_freq 2 --llambda 1
```

| Data split | Brown    | Reuters  | Gutenberg |
| ---------- | -------- | -------- | --------- |
| Train PPL  | 7604.10  | 4932.11  | 6087.90   |
| Dev PPL    | 4493.31  | 4310.01  | 4762.90   |

Consider the full log below in case you'd like to check the configuration and/or the generations of the resulting models.


```
Creating results directory: ../results/ngram

================================================================================

[Experiment Config]:
 Namespace(dataset_path='../data/corpora.tar.gz', output_dir='../results/ngram', use_interp=False, eval=False, ngram_size=3, alpha=0.8, llambda=1.0, min_freq=2, datasets=['brown', 'reuters', 'gutenberg'])
================================================================================

================================================================================
Training brown
================================================================================
brown  read. Num words:
-> train: 39802 
-> dev: 8437 
-> test: 8533
vocab: 23985
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 7604.106321480532
[PPL dev]  : 4493.310415928523
[PPL test] : 4477.879401527372
Training duration (min): 0.052

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'America', 'in', 'the', '<unk>'], 'seq_log_prob': -29.11857622964424, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'matter', 'and', 'energy', '<eos>'], 'seq_log_prob': -36.4705768160969, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'hand', 'the', '<unk>', 'of'], 'seq_log_prob': -28.424956575110585, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', '<unk>', 'of', 'the'], 'seq_log_prob': -22.53119363088048, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', 'lord', 'and', 'master', '<unk>', '<unk>'], 'seq_log_prob': -37.56852214456187, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/brown__3-gram.pkl

================================================================================
Training reuters
================================================================================
reuters  read. Num words:
-> train: 38183 
-> dev: 8083 
-> test: 8199
vocab: 21774
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 4932.105198350126
[PPL dev]  : 4310.008954147226
[PPL test] : 4250.0848248483235
Training duration (min): 0.07

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'Malaya', 'Chamber', 'of', 'Commerce'], 'seq_log_prob': -34.35633797737946, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'Sao', 'Paulo', 'said', 'in'], 'seq_log_prob': -34.86739317694181, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'hand', 'the', 'official', 'said'], 'seq_log_prob': -32.11061421956763, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', '<unk>', 'and', '<unk>', '<eos>'], 'seq_log_prob': -27.871737458025798, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', '<unk>', 'of', 'the', 'company', 'said'], 'seq_log_prob': -20.15671697649519, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/reuters__3-gram.pkl

================================================================================
Training gutenberg
================================================================================
gutenberg  read. Num words:
-> train: 68767 
-> dev: 14667 
-> test: 14861
vocab: 25458
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 6087.903922503255
[PPL dev]  : 4762.902237726482
[PPL test] : 4724.310624191728
Training duration (min): 0.11

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'Louisiana', 'always', 'the', 'first'], 'seq_log_prob': -37.11426926888753, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -17.127861767662697, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'side', 'of', 'the', 'LORD'], 'seq_log_prob': -21.15620054923185, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -15.423624170026079, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', 'lord', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -24.565460845817707, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/gutenberg__3-gram.pkl
Done!
```


### Example of execution log of trigram model with interpolation and backoff

Python command:
```shell 
python -m learn_ngram --ngram_size 3 --min_freq 2 --llambda 1 --use_interp --alpha 0.5
```

| Data split | Brown    | Reuters  | Gutenberg |
| ---------- | -------- | -------- | --------- |
| Train PPL  | 1446.16  | 1219.74  | 1039.59  |
| Dev PPL    | 1373.72  | 1211.23  | 1042.19   |

Consider the full log below in case you'd like to check the configuration and/or the generations of the resulting models.

```
Creating results directory: ../results/ngram

================================================================================

[Experiment Config]:
 Namespace(dataset_path='../data/corpora.tar.gz', output_dir='../results/ngram', use_interp=True, eval=False, ngram_size=3, alpha=0.5, llambda=1.0, min_freq=2, datasets=['brown', 'reuters', 'gutenberg'])
================================================================================

================================================================================
Training brown
================================================================================
brown  read. Num words:
-> train: 39802 
-> dev: 8437 
-> test: 8533
vocab: 23985
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 1446.159076001153
[PPL dev]  : 1373.7273660111916
[PPL test] : 1369.462325455672
Training duration (min): 0.2

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', '<unk>', '<eos>'], 'seq_log_prob': -10.789378007557989, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', '<unk>', '<eos>'], 'seq_log_prob': -10.789377718813864, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'the', 'the', 'the', 'the'], 'seq_log_prob': -17.111049493037168, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', '<unk>', '<eos>'], 'seq_log_prob': -10.7880787632349, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', 'lord', 'the', 'the', 'the', 'the'], 'seq_log_prob': -16.428099707011846, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/brown__interp_3-gram.pkl

================================================================================
Training reuters
================================================================================
reuters  read. Num words:
-> train: 38183 
-> dev: 8083 
-> test: 8199
vocab: 21774
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 1219.7363227621559
[PPL dev]  : 1211.234341214395
[PPL test] : 1214.1605359662408
Training duration (min): 0.28

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', 'company', 'said', '<eos>'], 'seq_log_prob': -14.817289619705003, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', 'company', 'said', '<eos>'], 'seq_log_prob': -14.817289649824325, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'the', 'the', 'the', 'the'], 'seq_log_prob': -17.84147481239684, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', 'company', 'said', '<eos>'], 'seq_log_prob': -14.817289710054675, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', '<unk>', '<eos>'], 'seq_log_prob': -4.3586752540819536, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/reuters__interp_3-gram.pkl

================================================================================
Training gutenberg
================================================================================
gutenberg  read. Num words:
-> train: 68767 
-> dev: 14667 
-> test: 14861
vocab: 25458
Fitting training data...

================================================================================
In domain Perplexities
================================================================================
[PPL train]: 1039.588426726311
[PPL dev]  : 1042.1867921470396
[PPL test] : 1022.7171155687306
Training duration (min): 0.44

================================================================================
Generating samples
================================================================================
------------------------------------------------------------
{'sequence': ['<bos>', 'United', 'States', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -10.34809745954381, 'prefix': ['United', 'States', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'united', 'states', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -10.347758388863085, 'prefix': ['united', 'states', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'on', 'the', 'other', 'the', 'the', 'the', 'the'], 'seq_log_prob': -15.808742667562836, 'prefix': ['on', 'the', 'other'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'bank', 'of', 'the', 'LORD', '<eos>'], 'seq_log_prob': -10.34471381269013, 'prefix': ['bank', 'of'], 'max_new_tokens': 5}
------------------------------------------------------------
{'sequence': ['<bos>', 'and', 'the', 'lord', 'the', 'the', 'the', 'the'], 'seq_log_prob': -15.803752727272936, 'prefix': ['and', 'the', 'lord'], 'max_new_tokens': 5}
Persisting model at ../results/ngram/gutenberg__interp_3-gram.pkl
Done!
```
