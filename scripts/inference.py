import argparse
import json

from finetune_llms.model_utils import chat_responses, load_model_and_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a model")
    parser.add_argument(
        "--model",
        type=str,
        default="pbouda/peft-unsloth-testing",
        help="Model name or path (default: pbouda/peft-unsloth-testing)"
    )
    args = parser.parse_args()

    # Load the model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Test prompts
    conversations = [
        [{"role": "user", "content": "What is a 'data breach'?"}],
    ]

    generated = chat_responses(model, tokenizer, conversations)
    for conversation in generated:
        print(json.dumps(conversation, indent=2))
        print("=====")
