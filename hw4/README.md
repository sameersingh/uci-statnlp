# HW4: Summarization

You will need to download the `xsum_val_small.jsonl` file from the course website, and place it into the `data` folder inside `hw4` (if you put it elsewhere, change the location in the code). 
You will need Python3 and the Python packages ``jsonlines``, ``rouge-score``, ``torch``, ``transformers``, and  `tqdm`. 


## Files

There are a few files in this folder:

#### Files you should modify

* `decoders.py`: You will need to complete the functions `nucleus_sampling` and `beam_search_decoding`.
* `generate.py`: You will change the decoder being used to generate summaries in the `generate()` function in this file.

#### Files you need not modify

* `decoders_test.py`: Used to test your implementation of nucleus and beam search decoding. 
* `evaluate.py`: Used to evalute generated summaries.
* `models.py`: Wrapper around models to interface with the `Candidate` class in `decoder.py`

## Running Tests
The file `decoders_test.py` contains a test for nucleus sampling and for beam search decoding.
If you are able to run `python decoders_test.py` without any error, then you should be good to go!


One your tests have passed, you can continue to generating summaries.

## Generating Summaries
Summaries are generated using `generate.py`, which loads in a [BART Transformer model](https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration) that has [already been trained](https://huggingface.co/sshleifer/distilbart-xsum-1-1) on the summarization dataset [XSUM](https://huggingface.co/datasets/viewer/?dataset=xsum).
Now we want to use this model to generate summaries on a small amount of validation data.

Running `generate.py` as provided will use greedy decoding to generate summaries. 
To change the decoder, you will need to change the function call in the `generate()` function.

``` bash
python generate.py 
    --input_file data/xsum_val_small.jsonl 
    --output_file <output file for generated summaries> 
```

## Evaluating Summaries
Your prediction file is evaluated using the ROUGE-L metric.
To evaluate your model, run:

``` bash
python generate.py 
    --gold_file data/xsum_val_small.jsonl 
    --predictions_file <file from generate.py>
    --output_file <file for instance-wise ROUGE-L scores> (optional)
```

This will print out the ROUGE-L score on the validation set.
Setting the flag `--output_file` will write out the ROUGE-L scores per generated summary.