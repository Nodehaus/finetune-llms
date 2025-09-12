import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"
    ):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()

    def make_request(
        self, endpoint: str, data: Dict[str, Any] | None = None, method: str = "POST"
    ) -> Dict[str, Any]:
        """Make a request to Ollama API."""
        url = f"{self.base_url}/{endpoint}"
        if method.upper() == "GET":
            response = self.session.get(url, timeout=300)
        else:
            response = self.session.post(url, json=data, timeout=300)
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Generate text using the Ollama model."""
        return self.make_request(
            "api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                },
            },
        )

    def test_connection(self) -> bool:
        """Test if Ollama is available and the model is loaded."""
        try:
            response = self.make_request("api/tags", method="GET")
            models = [model["name"] for model in response.get("models", [])]

            if self.model in models:
                logger.info(
                    f"Ollama connection successful. Model {self.model} is available."
                )
                return True
            else:
                logger.warning(
                    f"Model {self.model} not found. Available models: {models}"
                )
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
