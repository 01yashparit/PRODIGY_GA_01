import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2-finetuned", tokenizer="gpt2-finetuned")

st.title("GPT-2 Fine-tuned Text Generator")

prompt = st.text_area("Enter your prompt:", "")

if st.button("Generate"):
    generator = load_generator()
    with st.spinner("Generating..."):
        results = generator(
            prompt,
            max_length=50,
            num_return_sequences=3,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
    for i, res in enumerate(results):
        st.markdown(f"**Generated Text {i+1}:**")
        st.write(res['generated_text'])
        st.markdown("---")