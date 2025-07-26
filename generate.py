# generate.py

from transformers import pipeline
import os # It's good practice to use the os module for paths

def generate_text_from_model():
    """
    Loads the fine-tuned model and generates text based on a prompt.
    """
    model_path = "gpt2-finetuned"
    
    # Check if the directory exists before trying to load it
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found at '{model_path}'.")
        print("Please run main.py to fine-tune and save the model first.")
        return

    print(f"Loading model from: {os.path.abspath(model_path)}")
    text_generator = pipeline("text-generation", model=model_path, tokenizer=model_path)

    # --- Take input for prompt ---
    prompt = input("Enter your prompt: ")

    generated_texts = text_generator(
        prompt,
        max_length=50,
        num_return_sequences=3,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    print(f"\nPrompt: '{prompt}'\n")
    for i, text in enumerate(generated_texts):
        print(f"--- Generated Text {i+1} ---")
        print(text['generated_text'])
        print("-" * 25)

if __name__ == "__main__":
    generate_text_from_model()