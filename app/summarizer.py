from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self, model_dir="models/t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    def summarize(self, text, max_length=300):  # Increase summary length
        inputs = self.tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=150,  # Minimum length for more detailed summaries
            num_beams=5,
            length_penalty=2.0,  # Encourage longer summaries
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

