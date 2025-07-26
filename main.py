import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset  # <-- Add this import

def fine_tune_gpt2():
    """
    This function fine-tunes the GPT-2 model on a custom dataset.
    """
    # --- 1. Load Pre-trained Model and Tokenizer ---
    print("Loading pre-trained GPT-2 model and tokenizer...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # GPT-2 does not have a pad token by default, so we add one.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # --- 2. Load and Prepare Your Dataset ---
    train_file = "train.txt"
    print(f"Loading and tokenizing dataset from {train_file}...")

    # Load dataset using datasets library
    datasets = load_dataset("text", data_files={"train": train_file})

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We are doing Causal Language Modeling, not Masked Language Modeling
    )

    # --- 3. Set Up the Trainer ---
    print("Setting up training arguments...")
    output_dir = "./gpt2-finetuned"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,           # More epochs can lead to better results but also overfitting
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )

    # --- 4. Fine-Tune the Model ---
    print("Starting fine-tuning...")
    trainer.train()

    # --- 5. Save the Final Model ---
    print("Saving the fine-tuned model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("Fine-tuning complete!")

if __name__ == "__main__":
    fine_tune_gpt2()