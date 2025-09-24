import logging

from .base_client import BaseClient

logger = logging.getLogger(__name__)


class RunpodClient(BaseClient):
    """Client for interacting with RunPod IO API."""

    def __init__(
        self,
        url: str = "https://api.runpod.ai",
        pod_id: str = "",
        api_key: str = "",
    ):
        """Initialize RunPod client.

        Args:
            url: The RunPod endpoint URL
            pod_id: The Pod ID on RunPod
            api_key: Optional API key for Authorization header
        """
        super().__init__(url)
        if not pod_id:
            raise ValueError("You must specify a Pod ID for runpod.")
        self.pod_id = pod_id
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1000,
    ) -> str:
        """Generate text completion using RunPod API.

        Args:
            prompt: The input prompt for text generation

        Returns:
            Generated text response

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        payload = {
            "input": {
                "openai_route": "/v1/completions",
                "openai_input": {
                    "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                },
            }
        }

        response = self.make_request(
            f"v2/{self.pod_id}/runsync", payload, api_key=self.api_key
        )
        try:
            response_text = response["output"][0]["choices"][0]["text"].strip()
        except KeyError:
            raise ValueError(
                f"Did not receive valid response from Runpod, response was: {response}"
            )
        return response_text

    def test_connection(self) -> bool:
        """Test if RunPod endpoint is available.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Send a simple test prompt to check connectivity
            self.generate("Test")
            logger.info("RunPod connection successful.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RunPod: {e}")
            return False
