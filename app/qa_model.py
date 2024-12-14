from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QAWithT5:
    def __init__(self, model_dir="models/google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    def answer_question(self, context, question):
        input_text = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Answer in a detailed and structured manner, including examples if applicable:"
        )
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=300,  # Allow for longer responses
            num_beams=5,  # Use beam search for better responses
            length_penalty=1.2,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
