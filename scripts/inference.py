import json

from finetune_llms.model_utils import chat_responses, load_model_and_tokenizer

if __name__ == "__main__":
    # Example of how to use the utility functions

    # Load the fine-tuned model
    model, tokenizer = load_model_and_tokenizer("pbouda/peft-unsloth-testing")

    # Test prompts
    conversations = [
        [{"role": "user", "content": "What is a 'data breach'?"}],
    ]

    generated = chat_responses(model, tokenizer, conversations)
    for conversation in generated:
        print(json.dumps(conversation, indent=2))
        print("=====")
