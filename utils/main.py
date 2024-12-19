import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = "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"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Load model and tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=HF_TOKEN
    )
    return model, tokenizer

# Generate answer
def generate_answer(question, context, model, tokenizer, max_new_tokens=150):
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()
