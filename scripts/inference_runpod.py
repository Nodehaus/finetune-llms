import os

from dotenv import load_dotenv

from finetune_llms.clients.runpod_client import RunpodClient

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    """Run inference using RunPod client."""
    # Get configuration from environment variables
    runpod_api_key = os.getenv("RUNPOD_API_KEY", "")
    pod_id = os.getenv("RUNPOD_POD_ID", "xsvgocj2jv1i2i")

    # Initialize RunPod client
    client = RunpodClient(
        url="https://api.runpod.ai",
        pod_id=pod_id,
        api_key=runpod_api_key,
    )

    # Generate text from a simple prompt
    prompt = "What's the capital of Germany?"  # "What information must be included in the notification to a subscriber or individual affected by a data breach?"
    print(f"\nPrompt: {prompt}")
    print("Generating response...")

    response = client.generate(prompt)
    print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
