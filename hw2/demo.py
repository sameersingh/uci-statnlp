import pickle
import streamlit as st

from generator import Sampler
from lm import LangModel, Unigram


def load_lms():
    with open("saved_lms.pkl", "rb") as file:
        lm_dict = pickle.load(file)
    return lm_dict


def main():
    st.set_page_config(layout="wide")
    lm_dict = load_lms()
    key = st.sidebar.selectbox("Pick language model to use:", list(lm_dict.keys()))
    lm = lm_dict[key]
    left_col, right_col = st.beta_columns(2)

    # code for language model generation
    left_col.subheader("Language Model Generation")
    sampler = Sampler(lm)
    prefix = left_col.text_input("Enter a prefix if you want")
    max_length = left_col.slider("Maximum length of generated sentences", 10, 50)
    num_samples = left_col.slider("Number of sentences to generate", 1, 10)
    if st.button("Generate"):
        for i in range(num_samples):
            generated_text = " ".join(sampler.sample_sentence(prefix.split(), max_length))
            left_col.write(f"{i}) {generated_text}\n")

    # code for language model scoring
    right_col.subheader("Language Model Scoring")
    text = right_col.text_input("Enter a piece of text to score")
    if right_col.button("Score"):
        if text == "":
            right_col.write("Please entire a piece of text")
        else:
            text = text.split()
            numOOV = lm.get_num_oov([text])
            logprob_score = lm.logprob_sentence(text, numOOV)
            norm_logprob_score = logprob_score/len(text)
            right_col.write(f"Number of OOV tokens: {numOOV}")
            right_col.write(f"Log probability score: {logprob_score:.2f}")
            right_col.write(f"Normalized log probability score: {norm_logprob_score:.2f}")


if __name__ == "__main__":
    main()
