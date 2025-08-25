from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth import UnslothTrainer, UnslothTrainingArguments

import torch

def main():
    """Main training function."""
    # Configuration
    max_seq_length = 2048
    dtype = None  # None for auto detection
    load_in_4bit = True
    
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=True,   # Rank stabilized LoRA
        loftq_config=None, # LoftQ
    )
    
    print("Loading and preparing dataset...")
    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train[:2500]")
    
    def formatting_prompts_func(examples):
        texts = []
        for story in examples["text"]:
            # Add EOS token to prevent infinite generation
            text = story + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}
    
    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    print("Starting training...")
    # Training configuration
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            
            # Use warmup_ratio instead of warmup_steps
            warmup_ratio=0.1,
            num_train_epochs=1,
            
            # Select a learning rate
            learning_rate=5e-5,
            
            # Select an optimizer
            optim="adamw_8bit",
            
            # Select a weight decay
            weight_decay=0.01,
            
            # Select a learning rate scheduler
            lr_scheduler_type="linear",
            
            # Enable automatic mixed precision
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            
            # Miscellaneous settings
            seed=3407,
            output_dir="outputs",
            save_strategy="no",
        ),
    )
    
    # Start training
    trainer_stats = trainer.train()
    
    print("Training completed!")
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Samples per second: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    
    print("\nTesting inference...")
    # Test inference
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    inputs = tokenizer(
        [
            "Once upon a time, in a galaxy, far far away,"
        ], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
        generated_text = tokenizer.batch_decode(outputs)[0]
        print("\nGenerated story:")
        print(generated_text)
    
    print("\nSaving model...")
    # Save model in native format
    model.save_pretrained("mistral_7b_finetuned")
    tokenizer.save_pretrained("mistral_7b_finetuned")
    
    # Save model in GGUF format for efficient inference
    model.save_pretrained_gguf("mistral_7b_finetuned", tokenizer, quantization_method="q4_k_m")
    
    print("Model saved successfully!")

if __name__ == "__main__":
    main()