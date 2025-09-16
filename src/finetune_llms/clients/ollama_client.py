import logging

from .base_client import BaseClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseClient):
    """Client for interacting with Ollama API."""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "gemma3:1b"
    ):
        super().__init__(base_url)
        self.base_url = base_url
        self.model = model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1000,
        stream: bool = False,
    ) -> str:
        """Generate text using the Ollama model."""
        response = self.make_request(
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
        response_text = response.get("response")
        if not response_text:
            raise ValueError("Did not receive valid response from Ollama.")
        return response_text
