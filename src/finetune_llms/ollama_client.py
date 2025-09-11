import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


@dataclass
class Obligation:
    """Represents a legal obligation."""

    type: str  # "requirement" or "prohibition"
    description: str
    scope_subject: str
    scope_affected_parties: str
    context: str


class OllamaClient:
    """Client for interacting with Ollama API for legal text annotation."""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"
    ):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()

    def _make_request(
        self, endpoint: str, data: Dict[str, Any] = None, method: str = "POST"
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

    def generate_obligation_annotations(
        self, text: str, document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate obligation annotations for a given legal text.

        Args:
            text: The legal document text to analyze
            document_id: Identifier for the document

        Returns:
            List of obligation annotations
        """
        # Split text into chunks to avoid token limits
        chunks = self._split_text_into_chunks(text, max_chunk_size=4000)
        all_obligations = []

        for i, chunk in enumerate(chunks):
            logger.info(
                f"Processing chunk {i + 1}/{len(chunks)} for document {document_id}"
            )

            prompt = self._create_obligation_prompt(chunk)

            try:
                response = self._make_request(
                    "api/generate",
                    {
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent output
                            "top_p": 0.9,
                        },
                    },
                )

                obligations = self._parse_obligations_response(
                    response.get("response", "")
                )

                # Add document context to each obligation
                for obligation in obligations:
                    obligation["document_id"] = document_id
                    obligation["chunk_id"] = i
                    obligation["text_excerpt"] = (
                        chunk[:500] + "..." if len(chunk) > 500 else chunk
                    )

                all_obligations.extend(obligations)

            except Exception as e:
                logger.error(
                    f"Failed to process chunk {i} for document {document_id}: {e}"
                )
                continue

        return all_obligations

    def _create_obligation_prompt(self, text: str) -> str:
        """Create prompt for obligation extraction."""
        return f"""You are a legal AI assistant specializing in EU law. Extract legal obligations from the provided EUR-Lex text.

Focus on requirements (what must be done) and prohibitions (what must not be done).

Text: {text}

Return your response as valid JSON with the following structure for each obligation found:
{{
  "obligations": [
    {{
      "type": "requirement|prohibition",
      "description": "Clear summary of what must/must not be done",
      "scope_subject": "Who is obligated, who must comply with the rules",
      "scope_affected_parties": "Those who need to be aware because the obligation impacts them, even if they're not directly obligated",
      "context": "When it applies (simplified conditions)"
    }}
  ]
}}

Only return valid JSON. If no clear obligations are found, return {{"obligations": []}}.

JSON:"""

    def _parse_obligations_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the model response and extract obligations."""
        # Try to find JSON in the response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")

        json_str = response[json_start:json_end]
        parsed = json.loads(json_str)

        return parsed.get("obligations", [])

    def _split_text_into_chunks(
        self, text: str, max_chunk_size: int = 4000
    ) -> List[str]:
        """Split text into manageable chunks for processing."""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        sentences = text.split(". ")
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def test_connection(self) -> bool:
        """Test if Ollama is available and the model is loaded."""
        try:
            response = self._make_request("api/tags", method="GET")
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
