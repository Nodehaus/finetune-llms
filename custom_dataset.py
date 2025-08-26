"""
Custom dataset implementation for loading JSON files with text content.
Compatible with HuggingFace datasets library.
Uses generators to avoid loading all data into memory.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Union

import datasets


class CustomTextDataset:
    """
    Dataset class for loading JSON files and processing text content.

    Each JSON file should contain a dictionary with a specified key (default: 'content')
    that contains text with paragraphs separated by '\n\n'.

    Uses generators to process data lazily without loading everything into memory.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        json_key: str = "content",
        max_length: int = 2048,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to directory containing JSON files
            json_key: Key in JSON dict containing the text content (default: 'content')
            max_length: Maximum length of returned text chunks
        """
        self.data_path = Path(data_path)
        self.json_key = json_key
        self.max_length = max_length
        self._json_files = self._get_json_files()

    def _get_json_files(self) -> List[Path]:
        """Get list of JSON files to process."""
        if not self.data_path.is_dir():
            raise ValueError(f"Invalid data path: {self.data_path}")

        json_files = list(self.data_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {self.data_path}")

        return json_files

    def _process_json_file(self, json_file: Path) -> Iterator[str]:
        """Process a single JSON file and yield text chunks."""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if self.json_key not in data:
                print(f"Warning: Key '{self.json_key}' not found in {json_file}")
                return

            content = data[self.json_key]
            if not isinstance(content, str):
                print(f"Warning: Content in {json_file} is not a string")
                return

            # Generate chunks from content
            yield from self._create_chunks(content)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {json_file}: {e}")
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    def _create_chunks(self, text: str) -> Iterator[str]:
        """
        Split text into paragraphs and create chunks within max_length.

        Args:
            text: Input text with paragraphs separated by '\n\n'

        Yields:
            Text chunks, each <= max_length
        """
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If single paragraph is longer than max_length, split it
            if len(paragraph) > self.max_length:
                # Yield current chunk if it exists
                if current_chunk:
                    yield current_chunk.strip()
                    current_chunk = ""

                # Split long paragraph into smaller chunks
                yield from self._split_long_paragraph(paragraph)
                continue

            # Check if adding this paragraph would exceed max_length
            potential_chunk = (
                current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            )

            if len(potential_chunk) <= self.max_length:
                current_chunk = potential_chunk
            else:
                # Yield current chunk and start new one
                if current_chunk:
                    yield current_chunk
                current_chunk = paragraph

        # Yield final chunk if it exists
        if current_chunk:
            yield current_chunk

    def _split_long_paragraph(self, paragraph: str) -> Iterator[str]:
        """Split a long paragraph into word-level chunks."""
        words = paragraph.split()
        current_chunk = ""

        for word in words:
            potential_chunk = current_chunk + " " + word if current_chunk else word

            if len(potential_chunk) <= self.max_length:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    yield current_chunk
                current_chunk = word

        if current_chunk:
            yield current_chunk

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Iterate over text chunks from all JSON files."""
        for json_file in self._json_files:
            for chunk in self._process_json_file(json_file):
                yield {"text": chunk}


def load_custom_dataset(
    data_path: Union[str, Path], json_key: str = "content", max_length: int = 2048
) -> datasets.DatasetDict:
    """
    Load custom dataset and convert to HuggingFace Dataset format with 0.1 validation split.

    Args:
        data_path: Path to directory containing JSON files
        json_key: Key in JSON dict containing the text content (default: 'content')
        max_length: Maximum length of returned text chunks

    Returns:
        datasets.DatasetDict with 'train' and 'validation' splits (90%/10%)
    """

    def text_generator():
        custom_dataset = CustomTextDataset(
            data_path=data_path, json_key=json_key, max_length=max_length
        )
        for item in custom_dataset:
            yield item

    dataset = datasets.Dataset.from_generator(
        text_generator, features=datasets.Features({"text": datasets.Value("string")})
    )

    # Create 0.1 validation split
    split = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)

    return datasets.DatasetDict({"train": split["train"], "validation": split["test"]})
