import argparse

import torch
import wandb
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from finetune_llms.custom_dataset import load_annotation_dataset


def main(
    annotations_path: str,
    data_path: str,
    max_length: int,
    model_name: str,
    output_model_name: str,
):
    """Main PEFT training function."""
    # Configuration
    output_dir = f"outputs/{output_model_name}"
    max_seq_length = max_length
    dtype = None  # None for auto detection
    load_in_4bit = False  # Use 4bit for PEFT training

    print(f"Loading {model_name} model...")
    # Load pretrained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
    # TODO: Make this depend on the model
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    print("Setting up LoRA adapters...")
    # Add LoRA adapters - smaller rank for PEFT training
    model = FastLanguageModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # SHould leave on always!
        r=16,
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    print("Loading and preparing annotation dataset...")
    # Load annotation dataset
    datasets = load_annotation_dataset(
        annotations_path=annotations_path, data_path=data_path, max_length=max_length
    )

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    def formatting_prompts_func(examples):
        texts = []
        for conversations in examples["conversations"]:
            # Apply chat template to conversations
            formatted_text = tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            # Add EOS token to prevent infinite generation
            formatted_text = formatted_text + tokenizer.eos_token
            texts.append(formatted_text)
        return {"text": texts}

    # Apply formatting to datasets
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    print(train_dataset[100]["text"])

    print("Starting PEFT training...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=30,
            learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
            logging_steps=10,
            eval_steps=20,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="wandb",  # Use this for WandB etc
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    # trainer = UnslothTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     dataset_text_field="text",
    #     max_seq_length=max_seq_length,
    #     dataset_num_proc=2,
    #     args=UnslothTrainingArguments(
    #         per_device_train_batch_size=4,  # Larger batch size for PEFT
    #         per_device_eval_batch_size=8,
    #         gradient_accumulation_steps=4,  # Smaller accumulation
    #         warmup_ratio=0.1,
    #         num_train_epochs=3,  # More epochs for instruction tuning
    #         learning_rate=2e-4,  # Higher learning rate for PEFT
    #         embedding_learning_rate=1e-5,
    #         logging_steps=5,
    #         eval_steps=50,
    #         evaluation_strategy="steps",
    #         optim="adamw_8bit",
    #         weight_decay=0.01,
    #         lr_scheduler_type="cosine",
    #         seed=3407,
    #         output_dir=output_dir,
    #         report_to="wandb",
    #         save_strategy="steps",
    #         save_steps=100,
    #         save_total_limit=3,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="eval_loss",
    #         greater_is_better=False,
    #         # PEFT specific settings
    #         dataloader_pin_memory=False,
    #         remove_unused_columns=False,
    #     ),
    # )

    print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
    print(
        tokenizer.decode(
            [
                tokenizer.pad_token_id if x == -100 else x
                for x in trainer.train_dataset[100]["labels"]
            ]
        ).replace(tokenizer.pad_token, " ")
    )

    trainer_stats = trainer.train()

    print("Training completed!")
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(
        f"Samples per second: {trainer_stats.metrics['train_samples_per_second']:.2f}"
    )

    print("\nRunning final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"Final evaluation metrics: {final_metrics}")

    print("\nTesting inference...")
    FastLanguageModel.for_inference(model)
    run_inference_test(model, tokenizer, eval_dataset)

    print("\nSaving model to disk...")
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    print("\nSaving model to HuggingFace Hub...")
    # Use the output model name for HuggingFace Hub
    model_name_hub = f"pbouda/{output_model_name}"
    model.push_to_hub(model_name_hub, token=True)
    tokenizer.push_to_hub(model_name_hub, token=True)

    print("PEFT training completed successfully!")


def run_inference_test(model, tokenizer, val_dataset):
    """Run inference test using first 20 examples from validation dataset."""
    print("\nRunning inference test with validation data...")

    # Use first 20 examples from validation dataset
    num_examples = min(20, len(val_dataset))
    eval_examples = [val_dataset[i] for i in range(num_examples)]
    print(f"Using {num_examples} examples from validation dataset")

    results = []

    for i, example in enumerate(eval_examples):
        # Extract conversations from the dataset example
        conversations = example["conversations"]
        # Get the user prompt from the first conversation
        user_message = conversations[0]["content"]
        # Get the expected response from the assistant message
        expected_response = conversations[1]["content"]

        # Format user message as conversation and apply chat template
        messages = [{"role": "user", "content": user_message}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Tokenize and generate
        inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
            generated_text = tokenizer.batch_decode(outputs)[0]

            # Extract only the generated part (after the input)
            generated_response = generated_text[len(input_text) :].strip()

        result = {
            "example_id": i,
            "prompt": user_message,
            "expected_answer": expected_response,
            "generated_response": generated_response,
            "full_output": generated_text,
        }
        results.append(result)

        # Print only the first example
        if i == 0:
            print("\nFirst example result:")
            print(f"Prompt: {user_message[:200]}...")
            print(f"Expected: {expected_response[:200]}...")
            print(f"Generated: {generated_response[:200]}...")

    # Log results to wandb
    if wandb.run is not None:
        # Log summary metrics
        wandb.log({"inference_examples_count": len(results)})

        # Log detailed results as a table
        table_data = []
        for result in results:
            table_data.append(
                [
                    result["example_id"],
                    result["prompt"][:100],  # Truncate for table display
                    result["expected_answer"][:100],
                    result["generated_response"][:100],
                ]
            )

        table = wandb.Table(
            columns=["Example ID", "Prompt", "Expected Answer", "Generated Response"],
            data=table_data,
        )
        wandb.log({"inference_results": table})
        print(f"\nLogged {len(results)} inference results to wandb")
    else:
        print("\nWandb not initialized - skipping logging")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune model with PEFT using annotation dataset"
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="data/eng/subset/annotations/qa",
        help="Path to directory containing annotation JSON files",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/eng/subset",
        help="Path to directory containing source JSON files",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum length of input/output pairs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-3-12b-it",
        help="Base model to use for training",
    )
    parser.add_argument(
        "--output_model_name",
        type=str,
        default="peft-unsloth-testing",
        help="Output model name (outputs/ will be prefixed automatically)",
    )

    args = parser.parse_args()

    main(
        annotations_path=args.annotations_path,
        data_path=args.data_path,
        max_length=args.max_length,
        model_name=args.model_name,
        output_model_name=args.output_model_name,
    )
