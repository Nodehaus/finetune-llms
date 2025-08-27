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
        "Once upon a time, in a galaxy far, far away,",
        "The future of artificial intelligence is",
        "In a world where technology has advanced beyond imagination,",
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
