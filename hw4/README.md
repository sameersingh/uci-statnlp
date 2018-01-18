HW4: Neural Machine Translation
===


Assignment Description
---
In this programming assignment, you will design and train a neural machine translation system.
We have included a basic sequence-to-sequence model implemented in PyTorch for you to build upon.
Your task will be to build on this basic implementation in some way. Some ideas to try are (in order of increasing difficulty):

1. Changing the number of layers/hidden units/activation functions in the network architecture.
2. Making the encoder bidirectional.
3. Adding an attention mechanism to learn alignment. See [this paper](https://arxiv.org/abs/1409.0473) for more details.
4. Using beam-search in your decoder (with a beam-width larger than 1).

**Note: We highly reccomend getting an early start on this assignment!
Getting the proper hardware set up, and training neural networks can both be very frusterating and time consuming processes - especially if you have not done so in the past.
Please contact us early on if you are having difficulties.
Do not expect any sympathy if you start the project days before the deadline and encounter issues.**


Requirements
---
In addition, to an up-to-date installation of Python (e.g. 2.7 or 3.5+) you will also need to install the PyTorch neural network library.
Installation instructions can be found at <http://pytorch.org/>.
We also *strongly* reccommend that you use a NVIDIA CUDA enabled GPU to train your model (otherwise it is very unlikely you will be able to finish training before the deadline).
If you do not own a GPU do not worry, you can obtain *free* access to one through the [Google Cloud](https://cloud.google.com/) platform.
You may apply for credits [here]().
To help get aquainted with this system we reccommend checking out Stanford CS231n's [Google Cloud tutorial](http://cs231n.github.io/gce-tutorial/).


Files
---
* `model.py`: Contains the basic sequence-to-sequence model implementation.
* `train.py`: A script used to train the model.
* `translate.py`: A script used to generate translations.
* `TBD`: BLEU score evaluation script from moses-nmt.


Instructions
---


#### Training
To train the model, make sure that all of the lines in the `config.yaml` file point to the correct directories on your system.
Then run:

```bash
train.py --config config.yaml`
```


#### Generating translations
To generate translations run:

```bash
translate.py --input src.txt --output tgt.txt --checkpoint data/ckpt.pt
```


#### Evaluation

To evaluate the model run:

```bash
TBD (correct args)
```

