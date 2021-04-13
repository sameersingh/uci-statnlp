import pickle
import streamlit as st

from generator import Sampler
from lm import LangModel, Unigram, Ngram


def load_lms():
    with open("saved_lms.pkl", "rb") as file:
        lm_dict = pickle.load(file)
    return lm_dict


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
    text = st.text_input("Enter a piece of text to score")
    if st.button("Score"):
        if text == "":
            st.write("Please entire a piece of text")
        else:
            text = text.split()
            numOOV = lm.get_num_oov([text])
            logprob_score = lm.logprob_sentence(text, numOOV)
            perplexity = lm.perplexity([text])
            st.write(f"Number of OOV tokens: {numOOV}")
            st.write(f"Log probability: {logprob_score:.2f}")
            st.write(f"Perplexity: {perplexity:.2f}")


def main():
    lm_dict = load_lms()
    key = st.sidebar.selectbox("Pick language model to use:", list(lm_dict.keys()))
    lm = lm_dict[key]

    generate(lm)
    score(lm)


if __name__ == "__main__":
    main()
