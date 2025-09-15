import logging

# from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer for the given model name.
    Uses the same approach as belebele-batched.py for consistent generation behavior.
    Supports LoRA models via PEFT library.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set left padding for decoder-only models like OLMo
    if (
        "olmo" in model_name.lower()
        or "smollm3" in model_name.lower()
        or "qwen3" in model_name.lower()
    ):
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# def load_peft_model_from_huggingface(
#     model_name: str,
#     max_seq_length: int = 2048,
#     dtype: Optional[torch.dtype] = None,
#     load_in_4bit: bool = False,
#     token: Optional[str] = None,
# ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
#     """
#     Load a fine-tuned PEFT model from HuggingFace Hub.

#     Args:
#         model_name: HuggingFace model name (e.g., "pbouda/finetune-cpt-test")
#         max_seq_length: Maximum sequence length (default: 2048)
#         dtype: Model dtype, None for auto-detection (default: None)
#         load_in_4bit: Whether to load in 4-bit quantization (default: True)
#         token: HuggingFace token for private models (default: None)

#     Returns:
#         Tuple of (model, tokenizer) ready for inference

#     Example:
#         >>> from src.utils import load_peft_model_from_huggingface
#         >>> model, tokenizer = load_peft_model_from_huggingface(
#         ...     "pbouda/finetune-cpt-test"
#         ... )
#         >>>
#         >>> # Enable fast inference
#         >>> FastLanguageModel.for_inference(model)
#         >>>
#         >>> # Generate text
#         >>> inputs = tokenizer(["Your prompt here"], return_tensors="pt").to("cuda")
#         >>> outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
#         >>> generated_text = tokenizer.batch_decode(outputs)[0]
#         >>> print(generated_text)
#     """
#     print(f"Loading fine-tuned model from HuggingFace Hub: {model_name}")

#     # Load the fine-tuned model and tokenizer
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=model_name,
#         max_seq_length=max_seq_length,
#         dtype=dtype,
#         load_in_4bit=load_in_4bit,
#         token=token,
#         # Important: Set this to False for loading fine-tuned models
#         device_map="auto",
#     )

#     print("Model and tokenizer loaded successfully!")
#     print(f"Model device: {model.device}")
#     print(f"Model dtype: {model.dtype}")

#     return model, tokenizer


# def setup_model_for_inference(model: PreTrainedModel) -> PreTrainedModel:
#     """
#     Prepare model for fast inference.

#     Args:
#         model: The loaded model

#     Returns:
#         Model optimized for inference
#     """
#     FastLanguageModel.for_inference(model)
#     return model


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_cache: bool = True,
) -> str:
    """
    Generate text using the fine-tuned model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        use_cache: Whether to use key-value cache

    Returns:
        Generated text
    """
    inputs = tokenizer([prompt], return_tensors="pt")

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text


def chat_responses(model, tokenizer, conversations, tools=None):
    """
    Generate chat responses for a batch of conversations using chat templates.
    Uses the same approach as generate_batch_responses() but with chat templates.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        conversations: List of conversation histories, where each conversation is a list of messages
                      Each message is a dict with 'role' and 'content' keys
                      Example: [
                          [{"role": "user", "content": "Hello"}],
                          [{"role": "user", "content": "What's the weather?"},
                           {"role": "assistant", "content": "I'll check that for you."}]
                      ]
        batch_size: Batch size for generation
        tools: Optional list of tool definitions for tool calling. Each tool should be a dict with:
               - name: Tool name (string)
               - description: Tool description (string)
               - parameters: Tool parameter schema (dict)
               Example: [{"name": "weather", "description": "Get weather", "parameters": {...}}]

    Returns:
        List of response strings (only the generated assistant response)
    """
    all_responses = []

    # Check if tokenizer has chat template
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer {tokenizer.__class__.__name__} does not have a chat template. "
            "Use generate_batch_responses() for models without chat templates."
        )

    for conversation in conversations:
        tokenized_prompt = tokenizer.apply_chat_template(
            conversation,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            output_ids = model.generate(
                **tokenized_prompt.to(model.device),
                max_new_tokens=1000,
                cache_implementation="offloaded",
            )

        # Decode responses using same approach as generate_batch_responses
        # responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_responses.append(
            tokenizer.decode(output_ids[0][len(tokenized_prompt["input_ids"][0]) :])
        )

    return all_responses
