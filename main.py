import logging
import os
import traceback

from unsloth import FastLanguageModel  # ruff: isort: skip
from unsloth.chat_templates import get_chat_template, train_on_responses_only  # ruff: isort: skip

import boto3
import requests
import runpod

# import torch
# import wandb
from trl import SFTConfig, SFTTrainer

from finetune_llms.custom_dataset import load_training_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AI_PLATFORM_API_BASE_URL = os.getenv("AI_PLATFORM_API_BASE_URL", "")
AI_PLATFORM_API_KEY = os.getenv("AI_PLATFORM_API_KEY", "")
JOBS_PATH = "jobs/finetunes/"
JOBS_DONE_PATH = "jobs_done/fintunes/"
JOBS_FAILED_PATH = "jobs_failed/finetunes/"
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://s3.peterbouda.eu:3900")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "garage")


def update_finetune_status_api(finetune_id: str, status: str) -> bool:
    """Update finetune status via API.

    Args:
        finetune_id: The ID of the finetune
        status: The status to set (e.g., "RUNNING", "COMPLETED", "FAILED")

    Returns:
        True if successful, False otherwise
    """
    url = (
        f"{AI_PLATFORM_API_BASE_URL}/api/external/finetunes/{finetune_id}/update-status"
    )
    headers = {
        "X-API-Key": AI_PLATFORM_API_KEY,
        "Content-Type": "application/json",
    }
    data = {"status": status}

    try:
        response = requests.put(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        logger.info(f"Updated finetune {finetune_id} status to {status}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to update finetune status: {e}")
        return False


def run_training(
    s3_bucket: str,
    training_dataset_s3_path: str,
    documents_s3_path: str,
    base_model_name: str,
    model_name: str,
    finetune_id: str,
):
    """Main PEFT training function."""
    # Configuration
    output_dir = f"outputs/{model_name}"
    max_seq_length = 4096  # Context size
    dtype = None  # None for auto detection
    load_in_4bit = False  # Use 4bit for PEFT training

    s3_client = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        region_name=AWS_DEFAULT_REGION,
    )

    update_finetune_status_api(finetune_id, "RUNNING")

    try:
        logger.info(f"Loading {base_model_name} model...")
        # Load pretrained model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
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

        logger.info("Setting up LoRA adapters...")
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

        logger.info("Loading and preparing annotation dataset...")
        # Load annotation dataset
        datasets = load_training_dataset(
            s3_bucket=s3_bucket,
            training_dataset_s3_path=training_dataset_s3_path,
            documents_s3_path=documents_s3_path,
            s3_client=s3_client,
        )

        train_dataset = datasets["train"]
        eval_dataset = datasets["validation"]

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(eval_dataset)}")

        # Calculate save_steps based on training data size
        # Total steps = (dataset_size * num_epochs) / (batch_size * gradient_accumulation)
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 4
        num_train_epochs = 1

        total_steps = (len(train_dataset) * num_train_epochs) // (
            per_device_train_batch_size * gradient_accumulation_steps
        )

        # Calculate save_steps to save at 1/3 and 2/3 of training
        # We'll use the smaller of the two intervals (1/3) as save_steps
        # save_steps = max(10, min(500, total_steps // 3))

        logger.info(f"Total training steps: {total_steps}")

        def formatting_prompts_func(examples):
            texts = []
            for conversations in examples["conversations"]:
                # Apply chat template to conversations
                formatted_text = tokenizer.apply_chat_template(
                    conversations, tokenize=False, add_generation_prompt=False
                ).removeprefix("<bos>")
                texts.append(formatted_text)
            return {"text": texts}

        # Apply formatting to datasets
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

        logger.info("Starting PEFT training...")
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,  # Use GA
                warmup_steps=5,
                num_train_epochs=num_train_epochs,  # Set this for 1 full training run.
                # max_steps=30,
                learning_rate=2e-5,  # Reduce to 2e-5 for long training runs
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                report_to=None,  # "wandb",  # Use this for WandB etc
                output_dir="outputs",
                # save_strategy="steps",
                # save_steps=save_steps,
            ),
        )

        # TOOD: Make this depend on the model
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        trainer_stats = trainer.train()

        logger.info("Training completed!")
        logger.info(f"Trainer statistics: {trainer_stats}")

        final_metrics = trainer.evaluate()
        logger.info(f"Final evaluation metrics: {final_metrics}")

        # logger.info("\nSaving model to disk...")
        # model.save_pretrained(f"{output_dir}/final_model")
        # tokenizer.save_pretrained(f"{output_dir}/final_model")

        logger.info("Saving model to HuggingFace Hub...")
        # Use the output model name for HuggingFace Hub
        model_name_hub = f"pbouda/{model_name}"
        model.push_to_hub_merged(model_name_hub, tokenizer)
        # TODO: gguf quantized model saving does not work currently, they say they fixed it
        # but I get an error: https://github.com/unslothai/unsloth/issues/2581
        # Maybe better to convert manually with llama.cpp:
        # https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune#fine-tuning-gemma-3-in-unsloth
        # model.push_to_hub_gguf(
        #     f"{output_dir}/final_model",
        #     tokenizer,
        #     quantization_type="Q8_0",
        #     repo_id=f"{model_name_hub}-gguf",
        # )

        logger.info("Testing inference...")
        FastLanguageModel.for_inference(model)
        # run_inference_test(model, tokenizer, eval_dataset)

        # Move finished job file to `jobs_done/`
        job_filename = training_dataset_s3_path.split("/")[-1]
        jobs_done_key = f"{JOBS_DONE_PATH}{job_filename}"

        # Copy the job file to jobs_done/
        s3_client.copy_object(
            Bucket=s3_bucket,
            CopySource={"Bucket": s3_bucket, "Key": training_dataset_s3_path},
            Key=jobs_done_key,
        )

        # Delete the original job file
        s3_client.delete_object(Bucket=s3_bucket, Key=training_dataset_s3_path)
        logger.info(
            f"Moved completed job file from {training_dataset_s3_path} to {jobs_done_key}"
        )

        logger.info("PEFT training completed successfully!")
        update_finetune_status_api(finetune_id, "DONE")

    except Exception as e:
        logger.error("Job failed!!!!!!!!!!!!!!!!!!!")
        logger.error(traceback.format_exc())

        # Move failed job file to `jobs_failed/`
        job_filename = training_dataset_s3_path.split("/")[-1]
        jobs_failed_key = f"{JOBS_FAILED_PATH}{job_filename}"

        # Copy the job file to jobs_failed/
        s3_client.copy_object(
            Bucket=s3_bucket,
            CopySource={"Bucket": s3_bucket, "Key": training_dataset_s3_path},
            Key=jobs_failed_key,
        )

        # Delete the original job file
        s3_client.delete_object(Bucket=s3_bucket, Key=training_dataset_s3_path)
        logger.info(
            f"Moved completed job file from {training_dataset_s3_path} to {jobs_failed_key}"
        )

        update_finetune_status_api(finetune_id, "FAILED")
        raise e


# def run_inference_test(model, tokenizer, val_dataset):
#     """Run inference test using first 20 examples from validation dataset."""
#     print("\nRunning inference test with validation data...")

#     # Use first 20 examples from validation dataset
#     num_examples = min(20, len(val_dataset))
#     eval_examples = [val_dataset[i] for i in range(num_examples)]
#     print(f"Using {num_examples} examples from validation dataset")

#     results = []

#     for i, example in enumerate(eval_examples):
#         # Extract conversations from the dataset example
#         conversations = example["conversations"]
#         # Get the user prompt from the first conversation
#         user_message = conversations[0]["content"]
#         # Get the expected response from the assistant message
#         expected_response = conversations[1]["content"]

#         # Format user message as conversation and apply chat template
#         messages = [{"role": "user", "content": user_message}]
#         input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

#         # Tokenize and generate
#         inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

#         with torch.no_grad():
#             outputs = model.generate(**inputs, use_cache=True)
#             generated_text = tokenizer.batch_decode(outputs)[0]

#             # Extract only the generated part (after the input)
#             generated_response = generated_text[len(input_text) :].strip()

#         result = {
#             "example_id": i,
#             "prompt": user_message,
#             "expected_answer": expected_response,
#             "generated_response": generated_response,
#             "full_output": generated_text,
#         }
#         results.append(result)

#         # Print only the first example
#         if i == 0:
#             print("\nFirst example result:")
#             print(f"Prompt: {user_message}...")
#             print(f"Expected: {expected_response}...")
#             print(f"Generated: {generated_response}...")

#     # Log results to wandb
#     if wandb.run is not None:
#         # Log summary metrics
#         wandb.log({"inference_examples_count": len(results)})

#         # Log detailed results as a table
#         table_data = []
#         for result in results:
#             table_data.append(
#                 [
#                     result["example_id"],
#                     result["prompt"][:100],  # Truncate for table display
#                     result["expected_answer"][:100],
#                     result["generated_response"][:100],
#                 ]
#             )

#         table = wandb.Table(
#             columns=["Example ID", "Prompt", "Expected Answer", "Generated Response"],
#             data=table_data,
#         )
#         wandb.log({"inference_results": table})
#         print(f"\nLogged {len(results)} inference results to wandb")
#     else:
#         print("\nWandb not initialized - skipping logging")

#     return results


def handler(job):
    job_input = job["input"]
    try:
        run_training(
            s3_bucket=job_input["s3_bucket"],
            training_dataset_s3_path=job_input["training_dataset_s3_path"],
            documents_s3_path=job_input["documents_s3_path"],
            base_model_name=job_input["base_model_name"],
            model_name=job_input["model_name"],
            finetune_id=job_input["finetune_id"],
        )
        return "Training done."
    except Exception as exception:
        return {"error": str(exception)}


runpod.serverless.start({"handler": handler})
