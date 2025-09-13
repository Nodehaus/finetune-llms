"""
Custom dataset implementation for loading JSON files with text content.
Compatible with HuggingFace datasets library.
Uses generators to avoid loading all data into memory.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

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
    Load custom dataset and convert to HuggingFace Dataset format with validation split.

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

    # Convert generator to list to ensure we get a regular Dataset, not IterableDataset
    data_list = list(text_generator())
    dataset: datasets.Dataset = datasets.Dataset.from_list(data_list)

    dataset_dict = dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)  # type: ignore

    return datasets.DatasetDict(
        {"train": dataset_dict["train"], "validation": dataset_dict["test"]}
    )


def load_evaluation_data():
    """Load evaluation data from the JSON file."""
    current_dir = Path(__file__).parent
    json_path = current_dir / "eval_data" / "legalbench-abercrombie-100samples.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    return [
        {"prompt": item["prompt"], "answer": item["correct_answer"]} for item in data
    ]


class AnnotationDataset:
    """
    Dataset class for loading annotation files and creating input/output pairs.

    Loads annotation files from the annotations folder and finds corresponding
    source text from the original data files.
    """

    def __init__(
        self,
        annotations_path: Union[str, Path],
        data_path: Union[str, Path],
        max_length: int = 2048,
    ):
        """
        Initialize the annotation dataset.

        Args:
            annotations_path: Path to directory containing annotation JSON files
            data_path: Path to directory containing original source JSON files
            max_length: Maximum length of formatted input/output pairs
        """
        self.annotations_path = Path(annotations_path)
        self.data_path = Path(data_path)
        self.max_length = max_length
        self._annotation_files = self._get_annotation_files()

    def _get_annotation_files(self) -> List[Path]:
        """Get list of annotation files to process."""
        if not self.annotations_path.is_dir():
            raise ValueError(f"Invalid annotations path: {self.annotations_path}")

        annotation_files = list(self.annotations_path.glob("*_annotations.json"))
        if not annotation_files:
            raise ValueError(f"No annotation files found in {self.annotations_path}")

        return annotation_files

    def _get_source_file_path(self, annotation_file: Path) -> Path:
        """Get the corresponding source file path for an annotation file."""
        # Extract base name:
        # 02002L0058-20091219_eng_qa_annotations.json -> 02002L0058-20091219_eng
        base_name = annotation_file.name.split("_annotations")[0]
        source_file = self.data_path / f"{base_name}.json"

        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        return source_file

    def _load_source_content(self, source_file: Path) -> str:
        """Load the content field from a source JSON file."""
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "content" not in data:
                raise ValueError(f"'content' field not found in {source_file}")

            return data["content"]

        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file {source_file}: {e}")

    def _format_input_output(
        self, input_text: str, output_text: str, input_field: str, output_field: str
    ) -> Dict[str, Any]:
        """Format input and output into a conversation dictionary."""
        # Create conversation format
        if input_field == "question" and output_field == "answer":
            # Q&A format - use the input text directly as the question
            user_content = input_text
        else:
            # General format for other annotation types
            user_content = input_text

        # Create conversation dictionary
        conversation = {
            "conversations": [
                {"content": user_content, "role": "user"},
                {"content": output_text, "role": "assistant"},
            ]
        }

        return conversation

    def _process_annotation_file(
        self, annotation_file: Path
    ) -> Iterator[Dict[str, str]]:
        """Process a single annotation file and yield training examples."""
        # Load annotation file
        with open(annotation_file, "r", encoding="utf-8") as f:
            annotation_data = json.load(f)

        # Get field information
        input_field = annotation_data.get("input_field", "source_text")
        output_field = annotation_data.get("output_field", "annotation")
        annotations = annotation_data.get("annotations", [])

        # Only load source content if needed
        source_content = None
        if input_field == "source_text":
            source_file = self._get_source_file_path(annotation_file)
            source_content = self._load_source_content(source_file)

        for annotation in annotations:
            # Get input text based on input_field
            if input_field == "source_text":
                # Use the chunk of source text based on character offsets
                if source_content is None:
                    raise ValueError("Source content is None.")
                start = annotation.get("within_start", 0)
                end = annotation.get("within_end", len(source_content))
                input_text = source_content[start:end].strip()
            else:
                # Generic handling: get the field directly from annotation
                input_text = annotation.get(input_field)

            # Get output text based on output_field
            if output_field == "annotation":
                output_text = json.dumps(annotation)
            else:
                # Generic handling: get the field directly from annotation
                output_text = annotation.get(output_field)

            # Skip empty pairs
            if not input_text.strip() or not output_text.strip():
                raise ValueError(
                    f"Input or output is empty:\n  Input: {input_text}\n  "
                    f"Output: {output_text}"
                )

            # Format and yield training example
            conversation_data = self._format_input_output(
                input_text, output_text, input_field, output_field
            )

            yield conversation_data

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over training examples from all annotation files."""
        for annotation_file in self._annotation_files:
            yield from self._process_annotation_file(annotation_file)


def load_annotation_dataset(
    annotations_path: Union[str, Path],
    data_path: Union[str, Path],
    max_length: int = 2048,
) -> datasets.DatasetDict:
    """
    Load annotation dataset and convert to HuggingFace Dataset format.

    Args:
        annotations_path: Path to directory containing annotation JSON files
        data_path: Path to directory containing original source JSON files
        max_length: Maximum length of formatted input/output pairs

    Returns:
        datasets.DatasetDict with 'train' and 'validation' splits (95%/5%)
    """

    def annotation_generator():
        annotation_dataset = AnnotationDataset(
            annotations_path=annotations_path,
            data_path=data_path,
            max_length=max_length,
        )
        for item in annotation_dataset:
            yield item

    # Convert generator to list to ensure we get a regular Dataset, not IterableDataset
    data_list = list(annotation_generator())
    dataset: datasets.Dataset = datasets.Dataset.from_list(data_list)

    # Create 0.05 validation split (smaller since we have fewer annotation examples)
    dataset_dict = dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)  # type: ignore

    return datasets.DatasetDict(
        {"train": dataset_dict["train"], "validation": dataset_dict["test"]}
    )
