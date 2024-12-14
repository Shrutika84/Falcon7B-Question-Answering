import streamlit as st
from utils.dpr_utils import retrieve_top_k
from utils.falcon_utils import generate_answer, load_models
import json


# Load preprocessed JSON data
@st.cache_data
def load_json_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


# Set paths
JSON_PATH = "data/lecture_notes.json"

# Load preprocessed data
st.write("Loading data...")
raw_data = load_json_data(JSON_PATH)
documents = list(raw_data.values())

# Load models
st.write("Loading models...")
falcon_model, falcon_tokenizer, dpr_encoder, dpr_tokenizer, device = load_models()

# Encode documents for retrieval
st.write("Encoding documents...")
from utils.dpr_utils import encode_documents

doc_embeddings = encode_documents(documents, dpr_tokenizer, dpr_encoder, device)

# Streamlit UI
st.title("Falcon 7B Question Answering System")

st.markdown(
    """
    **Instructions**: Enter your question in the text box below to get an answer based on the preprocessed documents.
    """
)

# User Input
question = st.text_input("Ask your question:")

if st.button("Submit") and question:
    with st.spinner("Processing your question..."):
        # Retrieve top-k relevant context
        context = retrieve_top_k(question, documents, doc_embeddings, dpr_tokenizer, dpr_encoder, device)

        # Generate answer
        answer = generate_answer(context, question, falcon_model, falcon_tokenizer, device)

        # Display results
        st.subheader("Retrieved Context:")
        st.write(context)

        st.subheader("Generated Answer:")
        st.write(answer)
