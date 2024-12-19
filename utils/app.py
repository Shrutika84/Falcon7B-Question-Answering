import streamlit as st
from utils.model_utils import load_model, get_answer_mistral
from utils.retriever import load_json_data, embed_documents, retrieve_and_answer

# Initialize components
st.title("Custom NLP QA System")
st.write("Ask a question and get an answer based on the pre-loaded documents!")

# Load data and embeddings
data_path = "data/processed/lectureNotes.json"
doc_names, texts, embeddings = load_json_data(data_path)
embedder = embed_documents(texts)

# Load Mistral-7B model
model, tokenizer = load_model()

# User query
query = st.text_input("Enter your question:")

if st.button("Submit"):
    if query:
        with st.spinner("Fetching answer..."):
            answers = retrieve_and_answer(query, doc_names, texts, embeddings, embedder, model, tokenizer)
            for i, answer in enumerate(answers, 1):
                st.write(f"**Document {i}: {answer['doc']}**")
                st.write(f"**Answer {i}: {answer['answer']}**")
                st.write("---")
    else:
        st.error("Please enter a question!")
