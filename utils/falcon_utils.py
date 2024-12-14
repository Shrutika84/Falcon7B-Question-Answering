import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DPRContextEncoder, DPRContextEncoderTokenizer


def load_models():
    """Load Falcon and DPR models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Falcon model
    falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    falcon_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct").to(device)

    # Load DPR model
    dpr_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
    dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    return falcon_model, falcon_tokenizer, dpr_encoder, dpr_tokenizer, device


def generate_answer(context, question, falcon_model, falcon_tokenizer, device, max_new_tokens=106, num_beams=7,
                    temperature=0.5, top_p=0.95):
    """Generate a concise and complete answer using Falcon 7B."""
    input_text = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Provide a single, complete, and precise answer in one or two sentences. Ensure the response ends with a full stop:"
    )
    inputs = falcon_tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=1024, padding="max_length"
    ).to(device)

    output_ids = falcon_model.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens,
        num_beams=num_beams, temperature=temperature, top_p=top_p, early_stopping=True
    )
    return falcon_tokenizer.decode(output_ids[0], skip_special_tokens=True)
