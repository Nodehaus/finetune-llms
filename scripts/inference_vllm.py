import json
import os
from typing import Dict, List

import requests


def chat_completions_vllm(
    base_url: str,
    conversations: List[List[Dict[str, str]]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> List[List[Dict[str, str]]]:
    """
    Generate chat completions using a remote vLLM instance.

    Args:
        base_url: The base URL of the vLLM API endpoint
        conversations: List of conversation histories
        model: Model name (for vLLM API compatibility)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        List of conversations with generated assistant responses
    """
    results = []
    runpod_api_key = os.getenv("RUNPOD_API_KEY")

    for conversation in conversations:
        payload = {
            "messages": conversation,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(
            base_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {runpod_api_key}",
            },
            timeout=300,
        )
        response.raise_for_status()

        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]

        # Add the assistant response to the conversation
        updated_conversation = conversation + [
            {"role": "assistant", "content": assistant_message}
        ]
        results.append(updated_conversation)

    return results


if __name__ == "__main__":
    vllm_base_url = "https://api.runpod.ai/v2/nq2dji34jj9404/openai/v1/chat/completions"

    conversations = [
        [
            {
                "role": "user",
                "content": "What information must be included in the notification to a subscriber or individual affected by a data breach?",
            }
        ],
    ]

    generated = chat_completions_vllm(
        base_url=vllm_base_url, conversations=conversations
    )

    for conversation in generated:
        print(json.dumps(conversation, indent=2))
        print("=====")
