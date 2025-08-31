import argparse

import torch
from transformers import TrainerCallback
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

from finetune_llms.custom_dataset import load_custom_dataset
from finetune_llms.custom_evaluator import evaluate_model


class CustomEvalCallback(TrainerCallback):
    """Custom callback for running trademark evaluation during training."""

    def __init__(self, model, tokenizer, eval_steps=500):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        """Run custom evaluation at specified intervals."""
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\nRunning custom evaluation at step {state.global_step}...")
            # Switch to inference mode temporarily
            was_training = self.model.training
            FastLanguageModel.for_inference(self.model)

            # Run evaluation
            evaluate_model(self.model, self.tokenizer, log_to_wandb=True)

            # Switch back to training mode if needed
            if was_training:
                self.model.train()


def main(data_path: str = "data", json_key: str = "content", max_length: int = 2048):
    """Main training function."""
    # Configuration
    max_seq_length = max_length
    dtype = None  # None for auto detection
    load_in_4bit = False

    print("Loading Mistral 7B model...")
    # Load pretrained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-v0.3",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token="hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    print("Setting up LoRA adapters...")
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=True,  # Rank stabilized LoRA
        loftq_config=None,  # LoftQ
    )

    print("Loading and preparing custom dataset...")
    # Load custom dataset with train/validation split
    datasets = load_custom_dataset(
        data_path=data_path, json_key=json_key, max_length=max_length
    )

    train_dataset = datasets["train"]

    # Limit dataset sizes for testing
    # train_dataset = train_dataset.select(range(min(2500, len(train_dataset))))

    print(f"Train dataset size: {len(train_dataset)}")

    def formatting_prompts_func(examples):
        texts = []
        for text in examples["text"]:
            # Add EOS token to prevent infinite generation
            formatted_text = text + tokenizer.eos_token
            texts.append(formatted_text)
        return {"text": texts}

    # Apply formatting to dataset
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    # Create custom evaluation callback
    custom_eval_callback = CustomEvalCallback(model, tokenizer, eval_steps=50)

    print("Starting training...")
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        callbacks=[custom_eval_callback],
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            # Use warmup_ratio instead of warmup_steps
            warmup_ratio=0.1,
            num_train_epochs=1,
            # Select a learning rate
            learning_rate=5e-5,
            embedding_learning_rate=5e-6,
            logging_steps=10,
            # Select an optimizer
            optim="adamw_8bit",
            # Select a weight decay
            weight_decay=0.00,
            # Select a learning rate scheduler
            lr_scheduler_type="cosine",
            # Miscellaneous settings
            seed=3407,
            output_dir="outputs",
            report_to="wandb",
            save_strategy="steps",
            save_steps=500,
        ),
    )

    trainer_stats = trainer.train()

    print("Training completed!")
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(
        f"Samples per second: {trainer_stats.metrics['train_samples_per_second']:.2f}"
    )

    print("\nRunning final evaluation...")
    FastLanguageModel.for_inference(model)
    evaluate_model(model, tokenizer, log_to_wandb=True)

    print("\nTesting inference...")
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    inputs = tokenizer(
        [
            'A mark is generic if it is the common name for the product. A mark is descriptive if it describes a purpose, nature, or attribute of the product. A mark is suggestive if it suggests or implies a quality or characteristic of the product. A mark is arbitrary if it is a real English word that has no relation to the product. A mark is fanciful if it is an invented word.\n\nQ: The mark "Ivory" for a product made of elephant tusks. What is the type of mark?\nA: generic\n\nQ: The mark "Tasty" for bread. What is the type of mark?\nA: descriptive\n\nQ: The mark "Caress" for body soap. What is the type of mark?\nA: suggestive\n\nQ: The mark "Virgin" for wireless communications. What is the type of mark?\nA: arbitrary\n\nQ: The mark "Aswelly" for a taxi service. What is the type of mark?\nA: fanciful\n\nQ: The mark "Mask" for cloth that you wear on your face to filter air. What is the type of mark?\nA:'
        ],
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
        generated_text = tokenizer.batch_decode(outputs)[0]
        print("\nGenerated story:")
        print(generated_text)

    print("\nSaving model to disk...")
    model.save_pretrained("finetune-cpt-test")
    tokenizer.save_pretrained("finetune-cpt-test")
    model.save_pretrained_gguf(
        "finetune-cpt-test-gguf",
        tokenizer,
        quantization_type="Q8_0",
    )

    print("\nSaving model to HuggingFace Hub...")
    model_name = "pbouda/finetune-cpt-test"
    model.push_to_hub(model_name, token=True)
    tokenizer.push_to_hub(model_name, token=True)

    gguf_model_name = "pbouda/finetune-cpt-test-gguf"
    model.push_to_hub_gguf(
        gguf_model_name,
        tokenizer,
        quantization_type="Q8_0",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral 7B with custom dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to directory containing JSON files",
    )
    parser.add_argument(
        "--json_key",
        type=str,
        default="content",
        help="Key in JSON files containing text content",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum length of text chunks"
    )

    args = parser.parse_args()

    main(data_path=args.data_path, json_key=args.json_key, max_length=args.max_length)
