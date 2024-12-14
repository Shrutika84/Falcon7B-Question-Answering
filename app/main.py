import streamlit as st
from retriever import Retriever
from summarizer import Summarizer
from qa_model import QAWithT5
from utils import process_query

# Initialize components
retriever = Retriever()
retriever.index_documents("data/processed/lecture_notes.json")

summarizer = Summarizer()
qa_model = QAWithT5()

# Streamlit app UI
st.title("RAG-based NLP Tutor")
st.write("Ask questions about the lecture materials!")

# User input
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Processing..."):
        # Process the query and debug the outputs
        relevant_context, summarized_content, answer = process_query(retriever, summarizer, qa_model, query)

        st.write("**Relevant Context Extracted:**")
        st.write(relevant_context)

        st.write("**Summarized Content:**")
        st.write(summarized_content)

        st.write("**Answer:**")
        st.write(answer)
