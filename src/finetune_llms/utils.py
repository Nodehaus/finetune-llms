"""
Utility functions for loading and working with fine-tuned models.
"""

from typing import Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from unsloth import FastLanguageModel


def load_peft_model_from_huggingface(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
    token: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a fine-tuned PEFT model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name (e.g., "pbouda/finetune-cpt-test")
        max_seq_length: Maximum sequence length (default: 2048)
        dtype: Model dtype, None for auto-detection (default: None)
        load_in_4bit: Whether to load in 4-bit quantization (default: True)
        token: HuggingFace token for private models (default: None)

    Returns:
        Tuple of (model, tokenizer) ready for inference

    Example:
        >>> from src.utils import load_peft_model_from_huggingface
        >>> model, tokenizer = load_peft_model_from_huggingface("pbouda/finetune-cpt-test")
        >>>
        >>> # Enable fast inference
        >>> FastLanguageModel.for_inference(model)
        >>>
        >>> # Generate text
        >>> inputs = tokenizer(["Your prompt here"], return_tensors="pt").to("cuda")
        >>> outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
        >>> generated_text = tokenizer.batch_decode(outputs)[0]
        >>> print(generated_text)
    """
    print(f"Loading fine-tuned model from HuggingFace Hub: {model_name}")

    # Load the fine-tuned model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=token,
        # Important: Set this to False for loading fine-tuned models
        device_map="auto",
    )

    print("Model and tokenizer loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer


def setup_model_for_inference(model: PreTrainedModel) -> PreTrainedModel:
    """
    Prepare model for fast inference.

    Args:
        model: The loaded model

    Returns:
        Model optimized for inference
    """
    FastLanguageModel.for_inference(model)
    return model


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
        outputs = model.generate(
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
