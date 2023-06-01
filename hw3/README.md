# Open Domain Question Answering

In this assignment, you will be extending an existing implementation of a two-stage ODQA system. 
The two-stages consist of an information retrieval stage, often executed by a **retriever** model, and a reading stage, executed by a **reader** model.
The reading stage is also accompanied by an answer selection process, in which different candidate answers are considered for selecting the final answer that better addresses the user specified question.

Consider the following structure: 

1. [Installation and Setup](#installation-and-setup)
2. [Task 1: Improving the reader](#tasks)
2. [Task 2: Improving the retriever](#tasks)
2. [Task 3: Improving the answer selection strategy](#tasks)
3. [Code Structure](#repository-structure)



## Installation and Setup

The code in this repository was originally created in Python 3.9.
Please consider installing the following dependencies to run the code in this repository:

```
torch
rank_bm25
sentencepiece
transformers
faiss-cpu # alternatively, if you have GPU, faiss-gpu
sentence-transformers
tqdm
```

### Creating an environment with Anaconda

If you're considering installing the environment from scratch using the Anaconda dependency manager, here are the commands we followed. 

1. Create a Python3.9 environment named `cs272-hw3` and then activate it
```
conda create -n cs272-hw3 python=3.9
conda activate cs272-hw3 
```

2. Configure our conda installation to look up the packages on the channels `conda-forge` and `anaconda`.  This can be especially useful if you are installing multiple packages in individual commands.
```
conda config --env --add channels conda-forge
conda config --env --add channels anaconda
```

3. Install the basic Python data-processing and data visualization toolkit (based off of the packages `pandas`, `numpy`, `matplotlib`, `seaborn`). Also add `jupyter` for quick prototyping and `tqdm` for progressive bars.
```
conda install numpy pandas matplotlib seaborn jupyter tqdm
```

3. Install Pytorch=2.0.0 with cuda toolkit (since we have access to a gpu). Make sure the downloaded pytorch package is the cuda version (if you'd like to use the GPU). The name of the package should contain the pytorch version, your python version and the word cuda (e.g., here is an example of the name I get in a Linux machine `pytorch/linux-64::pytorch-2.0.0-py3.9_cuda11.7_cudnn8.5.0_0`).
```
conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Test that your implementation is cuda enable by executing the following in the command line. The command should execute without error and if you are planning to use a GPU it should print True in case your pytorch installation recognizes the GPU as a valid device.

```
python -c "import torch; print(torch.cuda.is_available()); torch.tensor([1]).to('cuda')"
```

4. Let us also install the fast indexing library `faiss-gpu` (if you don't have GPU, you should install `faiss-cpu` instead). 
```
conda install -c conda-forge faiss-gpu=1.7.4
```

5. Install huggingface related packages. Note that you should install transformers version greater than 4.26.
```
conda install protobuf=3.20.3 sentencepiece "transformers>=4.26.1" sentence-transformers=2.2.2
```

6. Install other useful packages for natural language processing
```
conda install nltk
```

7. Install the `rank_bm25` package, a Python implementation of several variants of BM25 ranking model. Since it is only available on pip, we will use pip command.
```
pip install rank_bm25
```

### (Optional) Setting up the Bing Search Retriever

In order to use Bing Web Search Retriever, you will have to sign up for the free access.
To obtain the subscription key head over to [Bing Web Search: Get Started](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview)
and follow their directions to obtain the free access. 
Note that the free access allows you to make 3 Transactions Per Second (TPS)
and **up to 1k calls per month free of charge**.
You might have to use your student email to obtain the student perks from Azure.

## Tasks: Extending and evaluating a two-part ODQA system

### Task 1. Implement `GenerativeQAReader` model at `reader.py`

A common approach to model readers in 2-part ODQA systems is to use span extraction models that extract the answer from a continuous piece of the supporting document.
While this works for simpler questions, it may not be the case for more complex questions that involve combining information from multiple parts of the supporting document.
In those cases, generative approaches can be more useful. In this homework, your first exercise will be to implement a T5-based generative model for addressing the reading problem in ODQA systems. Your model should receive a document and a question and output an answer that may or may not be verbatim from the supporting document.
We suggest that you implement your system in a way that can be described fully via a configuration file, as it will help you run experiments quickly.

After implementing this model, you should use the `run_eval.py` script to conduct analysis of the implemented reader system. To conduct the analysis using the golden documents, you should run the following command:

```
python -m run_eval --reader_gold_eval --reader_filepath <path_to_your_config_file>
```

For example, here is the command we used to obtain the result for the default reader (located at `./config/rd_default.json`). We execute the following command within the code directory (for simplicity).

```
python -m run_eval --reader_gold_eval --reader_filepath ../configs/rd_default.json
```

Executing the command above in the terminal yielded the following output
```
============================================ Conduct default evaluation ============================================
Number of contexts: 2582
Number of questions: 337
Number of answers: 337
============================================ Evaluating ODQA Pipeline ============================================
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 3208.22it/s]
Duration (min): 7.193485895792643e-05
Reader Exact Match: 0.59%
```


### Task 2. Implement the `SentenceEncRetriever` model at `retriever.py`

This repository contains complete implementation of different baselines, including using sparse representations of texts based on empirical counts and leveraging static word embeddings to create a denser representation. 
The former is mostly based on **tf-idf weighting** scheme, where instead of raw counts, we use term frequencies and inverse document frequencies to weight terms differently (e.g., not putting too much weight on stopwords, relying on rarer words). A slightly more powerful variant of td-idf weighting scheme is called **BM25**, which introduces a relative weighting parameter `k1` and a normalization by the document length, controlled by the `b` parameter.

On the other hand, `AvgWordEmbeddingRetriever` preemptively loads [`GloVe` embeddings]() and obtains a lower dimensional (more dense) representation by averaging all the word embeddings that comprise a piece of text.
To use this variant, consider downloading the embeddings from this [GoogleDrive folder](https://drive.google.com/drive/folders/1RxxhmaIoBI1rA6ly5E4tDlvOET7YRUWI?usp=sharing). There will be a .zip file that you should download and unzip. The resulting path should then be specified in the corresponding config files under the `embedding_path` config. Note that you can also download from the embeddings from the [original Stanford University Webpage](https://nlp.stanford.edu/projects/glove/) but may face some problems when loading the files for 100- and 200-dimensions (`100d` and `200d`).

**However**, neither of these approaches takes the ordering of the words in the piece of text into consideration, or synonymity. One idea to overcome both these issues is to use sentence encoders, where given a sentence, we obtain a single embedding representation for it. 
Your task will be to:
- use `sentence-transformers` to implement a `SentenceEncRetriever` class in `retriever.py`. We recommend implementing the model in a way that can be fully described in terms of config files.
- report the retriever's `recall@10` (that is the recall of the retrievers when retrieving 10 documents). This performance metric represents the fraction of times that a given model returns at least one of the correct documents amongst the k retrieved documents. 

To compute the evaluation metric, you can use the `run_eval.py` script, as follows:

```
python -m run_eval --retriever_filepath <path_to_your_config_file> --k 10
```

For example, here is the command we used to obtain the result for the bm25 retriever (located at `./config/rt_bm25.json`). We execute the following command within the `./code` directory (for simplicity).

```
python -m run_eval --retriever_filepath ../configs/rt_bm25.json
```

Executing the command above in the terminal yielded the following output:
```
======================================== Conduct default evaluation ========================================
Namespace(datapath='../data/bioasq_dev.json', retriever_filepath='../configs/rt_bm25.json', reader_filepath='../configs/rd_default.json', reader_gold_eval=False, k=10, batch_size=32)
Number of contexts: 2582
Number of questions: 337
Number of answers: 337
Fitting 2582 documents to retriever
Duration (min): 0.019374509652455647
======================================== Evaluating ODQA Pipeline ========================================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:01<00:00, 10.57it/s]
Duration (min): 0.01735107898712158
Retriever R@10: 90.21%
Reader Exact Match: 0.59%
```

Notice, that for BM25 the retriever recall@10 is 90.21% (see line `Retriever R@10: 90.21%`).


### Task 3. Implement a confidence-based answer selection strategy in `reader.py`

So far, we have been assuming that we have a perfect retriever and that, indeed, the most relevant document will be the first document.
Unfortunately, this may not always be the case.

Your task is to implement a confidence-based answer selection strategy. You should modify the method `Reader._select_answer` at `reader.py`.
Report the reader's performance on the dev set when using this answer selection strategy. 
You can update the answer selection strategy in your reader config files by updating the `"answer_selection": "first"` to be `"answer_selection": "confidence"` and then running the same commands you run in [Task 1](#task-1-implement-generativeqareader-model-at-readerpy).

### Running ODQA queries with `run_custom_query.py`

We also make available a python script to facilitate running your own experiments with custom queries (while using the same document collection). To do that, consider using the following command (note that to specify different queries in the same command, you should use the semicolon `;`.
):
```
python -m run_custom_query --reader_filepath <your reader config filepath> --retriever_filepath <your retriever config filepath> --k 10 --query "Is there evidence that tomato juice lowers cholesterol levels?;Which type of lung cancer is afatinib used for?;Which hormone abnormalities are characteristic to Pendred syndrome?"
```

Executing this command will execute the ODQA sytem end-to-end, using the specified retriever to retrieve `k` documents for each of the specified queries; and using the specified reader to obtain the final answer. The results are put in a file `results.jsonl` in the specified output_dir (defaults to `./results`).


## Repository Structure

Let us first describe the organization of the repository at a high-level:

- `code`: contains all the necessary source code files for this assignment;
- `configs`: contains the different reader and retriever configurations that you will be using to run your experiments;
- `data`: contains the data files `bioasq_dev.json` and `bioasq_test.json`.
- `results`: directory where by default all artifacts of code execution will be saved.

Let us now dive into what exactly is the organization of the `code` folder:

- `data`: utilities to load the data from the provided files and class definitions for `Answer` and `ODQADataset`.
- `evaluate`: utilities to conduct evaluation for ODQA systems. It contains the definition of recall@k (used to evaluate the retriever) and exact match (used to evaluate the reader);
- `reader`: definition of reader API and exposes a span extraction baseline. You will have to update this file to complete this assignment's tasks 1 and 3.
- `retriever`: definition of retriever API and exposes several baselines including the average word embedding, bm25, and bing API. You will have to update this file to complete this assignment's task 2.
- `run_custom_query`: python script that enables you to try custom queries against the biomedical pool of documents. You should use this to conduct your own analysis.
- `run_eval`: executes the evaluation of retriever and reader system. By default it will run the end2end evaluation.
- `utils`: utilities to dynamically load classes and embeddings based on config files.

As for the `configs` folder, the current files follow a simple convention: all the reader configurations are prefixed with `rd` (short for reader), whereas all the retriever configurations are prefixed with `rt`.


## Disclaimer
 
For the purpose of this homework3, we are reusing the **BioASQ Task B** data made publicly available by [dmis-lab/biobert](https://github.com/dmis-lab/biobert).