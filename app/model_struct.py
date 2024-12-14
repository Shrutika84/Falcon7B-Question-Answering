from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Define the directory to save the model
model_name = "google/flan-t5-small"
save_dir = "models/google/flan-t5-small"


# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Download and save the tokenizer and model
print(f"Downloading and saving model '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Model '{model_name}' has been saved to '{save_dir}'.")
