import glob
import os
import pickle
import streamlit as st

from generator import Sampler
from lm import LangModel, Unigram, Ngram


def load_lm():
    # get list of all directories with `.pkl` files
    save_dirs = sorted(set([os.path.dirname(d) for d in glob.glob('*/*.pkl')]))
    save_dir = st.sidebar.selectbox("Pick a save dir to load an LM", save_dirs)

    # get list of all `.pkl` files in directory and pick one
    pkl_files = sorted(glob.glob(os.path.join(save_dir, '*.pkl')))
    pkl_file = st.sidebar.selectbox("Pick which LM to load", pkl_files)
    with open(pkl_file, "rb") as file:
        lm = pickle.load(file)

    return lm


def generate(lm):
    st.subheader("Language Model Generation")
    prefix = st.text_input("Enter a prefix (default: no prefix)")
    temp = st.number_input("Enter a temperature", value=1.0)
    max_length = st.slider("Maximum length of generated sentences", 10, 50)
    num_samples = st.slider("Number of sentences to generate", 1, 10)
    sampler = Sampler(lm, temp)
    if st.button("Generate"):
        for i in range(num_samples):
            generated_text = " ".join(sampler.sample_sentence(prefix.split(), max_length))
            st.write(f"{i}) {generated_text}\n")


def score(lm):
    st.subheader("Language Model Scoring")
    text = st.text_input("Enter a piece of text to score").split()
    if st.button("Score"):
        numOOV = lm.get_num_oov([text])
        logprob_score = lm.logprob_sentence(text, numOOV)
        perplexity = lm.perplexity([text])
        st.write(f"Number of OOV tokens: {numOOV}")
        st.write(f"Log probability: {logprob_score:.2f}")
        st.write(f"Perplexity: {perplexity:.2f}")


def main():
    lm = load_lm()
    generate(lm)
    score(lm)


if __name__ == "__main__":
    main()
