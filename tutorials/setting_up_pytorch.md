PyTorch Installation - Best Practices
===

This tutorial provides instructions and advice for how to setup a Python environment with the PyTorch module.


## **Step 1** Install Anaconda

We strongly recommend using the Anaconda Python distribution for your coursework.
To install Anaconda, follow the instructions for you operating system at: https://www.anaconda.com/distribution/.

## **Step 2** Create and activate a virtual environment

Create and activate virtual environment by entering the following into your terminal:
```{bash}
conda create -n venv
conda activate venv
```
After running this, the command line should now have the prefix `(venv)`.

Note: You may use a name other than `venv` in the lines above if you prefer - it is just the name you are giving to the virtual environment.
One Common convention is to give the environment the same name as the project you are using it for.

## **Step 3** Install PyTorch

Install the latest version PyTorch to your environment by running one of the following:

Linux and Windows
```{bash}
# CPU only
conda install pytorch torchvision cpuonly -c pytorch

# GPU
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

MacOS
```{bash}
# CPU / GPU
conda install pytorch torchvision -c pytorch
```
Note: According to PyTorch's website, MaxOS binaries don't support CUDA.
If you want to use GPU acceleration you will need to install CUDA yourself.
The installation files and instructions are available at:  https://developer.nvidia.com/cuda-downloads.


##  Frequently Asked  Questions

*FAQ: How is Anaconda different from Python?*

Anaconda is a package and environment manager for Python designed to facilitate doing data science and machine learning.
Installing Anaconda installs a copy of Python which is pre-configured with a lot of useful libraries (like Jupyter, NumPy, Scikit-Learn).
In addition, Anaconda also makes it really easy to install PyTorch using its package manager.
Unlike pip (the default pacakge manager for Python) Anaconda's package manager also takes care of installing external dependencies such as CUDA and CuDNN (at least on Linux and Windows) which are required for GPU computing (and can be tricky to manually install).


*FAQ: What is a virtual environment?*

Virtual environments ensure that project dependencies do not cause conflicts across projects.
To understand the problem virtual environments solve, consider the following scenario:

>  You've come up with the next amazing model.
>  You decide that you are going to write it using a package `foo`.
>  So you follow the installation instructions, your model works, you write up a paper describing your results, and send it off to a top-tier conference to be published.
> Life is good.
>
>  Then reviews come back.
>  Everyone agrees your results look great, but they won't accept your paper unless you include results for some super old baseline from 2018 for comparison.
>  Luckily, all of the code for the baseline is available online so you can just run it on your data and go on to getting your best paper award. Right?
>
> Not quite. When you try to run the code you get an error:
> ```
> NameError: 'foo.old_function()' is not defined
> ```
> After a quick search on StackExchange you learn that `old_function` was removed from the current version of `foo`.
> Okay!
> So to fix the issue you just need the old version of `foo`.
> This an easy enough problem to solve: the old version is available online.
> So you install it, run the baseline, update your paper, and the reviewers are satisfied.
> Life is good again.
>
> Now there's 15 milliseconds before your final draft is due - plenty of time to run some last minute experiments according to your advisor.
> No big deal.
> Your code was expertly crafted, you knew you would have to accomodate these kinds of requests, all you need to do is change one command line parameter.
> So you run `python accomodate_advisor.py --minutes_ago 10` and then the following pops up:
> ```
> NameError: `foo.new_function()` is not defined
> ```
> Oh no!
> Your code is incompatible with the old version of `foo` you installed to run the baseline.
> There's no time to update it.
> You are forced to omit the experiment from the paper.
>
> The next day your archnemesis who works for *Huge Company with 1 Million GPUs Inc.* posts a remarkably similar ArXiV preprint of their submission to Twitter and it is retweeted by everyone in the community.
> Unlike your submission, it includes the last minute experiment.
>
> When conference time comes around their work recieves the best paper award and gets a spotlight talk.
> Meanwhile, your work is relegated to the darkest most remote corner of the venue to be presented at a poster session scheduled at the same time as their talk.
> You come back to a life in shambles: your advisor shreds your thesis in front of the committee during your defense, at family dinners all your parents talk about is what a dissappointment you are, and your partner leaves you for your arch nemesis.
> Lonliness and defeat is all you'll ever know.

This could all be avoided by creating seperate virtual environments for your project and the baseline: you can install the new version of `foo` in your project's environment, the old version of `foo` in the baseline's environment, and there will never be any conflict since the environments are isolated.

