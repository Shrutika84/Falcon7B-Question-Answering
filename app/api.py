from flask import Flask, request, jsonify
from retriever import Retriever
from summarizer import Summarizer
from qa_model import QAWithT5
from utils import process_query
import json

# Initialize Flask app
app = Flask(__name__)

# Initialize RAG components
retriever = Retriever()
retriever.index_documents(r"C:\Users\shrut\Documents\Fall24\Advance NLP\rag_tutor\data\processed\lecture_notes.json")


summarizer = Summarizer()
qa_model = QAWithT5()


@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        # Get user query from the request
        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"error": "Query not provided"}), 400

        # Process the query
        relevant_context, summarized_content, answer = process_query(retriever, summarizer, qa_model, query)

        # Prepare the response
        response = {
            "query": query,
            "relevant_context": relevant_context,
            "summarized_content": summarized_content,
            "answer": answer,
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
