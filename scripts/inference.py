from finetune_llms.utils import (
    generate_text,
    load_peft_model_from_huggingface,
    setup_model_for_inference,
)

if __name__ == "__main__":
    # Example of how to use the utility functions

    # Load the fine-tuned model
    model, tokenizer = load_peft_model_from_huggingface("pbouda/finetune-cpt-test")

    # Set up for inference
    model = setup_model_for_inference(model)

    # Test prompts
    test_prompts = [
        'A mark is generic if it is the common name for the product. A mark is descriptive if it describes a purpose, nature, or attribute of the product. A mark is suggestive if it suggests or implies a quality or characteristic of the product. A mark is arbitrary if it is a real English word that has no relation to the product. A mark is fanciful if it is an invented word.\n\nQ: The mark "Ivory" for a product made of elephant tusks. What is the type of mark?\nA: generic\n\nQ: The mark "Tasty" for bread. What is the type of mark?\nA: descriptive\n\nQ: The mark "Caress" for body soap. What is the type of mark?\nA: suggestive\n\nQ: The mark "Virgin" for wireless communications. What is the type of mark?\nA: arbitrary\n\nQ: The mark "Aswelly" for a taxi service. What is the type of mark?\nA: fanciful\n\nQ: The mark "Mask" for cloth that you wear on your face to filter air. What is the type of mark?\nA:'
    ]

    print("\n" + "=" * 50)
    print("Testing fine-tuned model:")
    print("=" * 50)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt}")
        print("-" * 40)

        generated = generate_text(
            model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=100
        )

        # Extract only the generated part (remove the original prompt)
        generated_only = generated[len(prompt) :].strip()
        print(f"Generated: {generated_only}")
        print("-" * 40)
