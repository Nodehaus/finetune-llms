import json
import os
import shutil
import tempfile
from typing import Any, Dict, Iterator, Set

import boto3
import datasets


class AnnotationDataset:
    """
    Dataset class for loading annotation files and creating input/output pairs.

    Loads annotation files from S3 and finds corresponding source text
    from the original data files.
    """

    def __init__(
        self,
        s3_bucket: str,
        training_dataset_s3_path: str,
        documents_s3_path: str,
    ):
        """
        Initialize the annotation dataset.

        Args:
            s3_bucket: S3 bucket name
            training_dataset_s3_path: S3 path to the training dataset JSON file
            documents_s3_path: S3 path prefix for document files
        """
        self.s3_bucket = s3_bucket
        self.training_dataset_s3_path = training_dataset_s3_path
        self.documents_s3_path = documents_s3_path
        self.s3_client = boto3.client("s3")
        self.temp_dir = tempfile.mkdtemp()
        self.source_documents: Dict[str, str] = {}

    def _load_s3_json(self, s3_key: str) -> Dict[str, Any]:
        """Load JSON data from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"Error loading JSON from S3 key {s3_key}: {e}")

    def _get_source_file_key(self, filename: str) -> str:
        """Get the S3 key for a source document file."""
        return f"{self.documents_s3_path.rstrip('/')}/{filename}"

    def _download_source_document(self, filename: str) -> str:
        """Download a source document from S3 to temp directory."""
        if filename in self.source_documents:
            return self.source_documents[filename]

        source_key = self._get_source_file_key(filename)
        local_path = os.path.join(self.temp_dir, filename)

        try:
            self.s3_client.download_file(self.s3_bucket, source_key, local_path)
            self.source_documents[filename] = local_path
            return local_path
        except Exception as e:
            raise ValueError(f"Error downloading source document {source_key}: {e}")

    def _load_source_content(self, filename: str) -> str:
        """Load the content field from a local downloaded source JSON file."""
        local_path = self.source_documents[filename]

        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "content" not in data:
            raise ValueError(f"'content' field not found in {filename}")

        return data["content"]

    def _get_unique_source_documents(self, annotations: list) -> Set[str]:
        """Get unique source document filenames from annotations."""
        source_documents = set()
        for annotation in annotations:
            source_document = annotation.get("source_document")
            if source_document:
                source_documents.add(source_document)
        return source_documents

    def _download_all_source_documents(self, source_documents: Set[str]) -> None:
        """Download all unique source documents to temp directory."""
        for filename in source_documents:
            self._download_source_document(filename)

    def _format_input_output(self, input_text: str, output_text: str) -> Dict[str, Any]:
        """Format input and output into a conversation dictionary."""
        conversation = {
            "conversations": [
                {"content": input_text, "role": "user"},
                {"content": output_text, "role": "assistant"},
            ]
        }

        return conversation

    def _process_training_dataset(self) -> Iterator[Dict[str, Any]]:
        """Process the training dataset and yield training examples."""
        # Load training dataset from S3
        annotation_data = self._load_s3_json(self.training_dataset_s3_path)

        # Get field information
        input_field = annotation_data.get("input_field")
        output_field = annotation_data.get("output_field")
        annotations = annotation_data.get("annotations")

        if not input_field or not output_field:
            raise ValueError("input_field and output_field are required")

        if not annotations:
            raise ValueError("No annotations found in training dataset")

        # Get unique source documents and download them if needed
        if input_field == "source_text":
            unique_source_docs = self._get_unique_source_documents(annotations)
            if not unique_source_docs:
                raise ValueError("No source documents found in annotations")
            self._download_all_source_documents(unique_source_docs)

        for annotation in annotations:
            # Get input text based on input_field
            if input_field == "source_text":
                # Get source document for this annotation
                source_document = annotation.get("source_document")

                # Load source content from local file (no caching)
                source_content = self._load_source_content(source_document)
                start = annotation.get("within_start", 0)
                end = annotation.get("within_end", len(source_content))
                input_text = source_content[start:end].strip()
            else:
                # Generic handling: get the field directly from annotation
                raw_input = annotation.get(input_field)
                input_text = str(raw_input).strip()

            # Get output text based on output_field
            if output_field == "annotation":
                output_text = json.dumps(annotation)
            else:
                # Generic handling: get the field directly from annotation
                raw_output = annotation.get(output_field)
                output_text = str(raw_output).strip()

            # Skip empty pairs
            if not input_text or not output_text:
                raise ValueError(
                    f"Input or output is empty:\n  Input: {input_text}\n  "
                    f"Output: {output_text}"
                )

            # Format and yield training example
            conversation_data = self._format_input_output(input_text, output_text)

            yield conversation_data

    def cleanup(self) -> None:
        """Clean up temporary directory and downloaded files."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over training examples from the training dataset."""
        try:
            yield from self._process_training_dataset()
        finally:
            self.cleanup()

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.cleanup()


def load_annotation_dataset(
    s3_bucket: str,
    training_dataset_s3_path: str,
    documents_s3_path: str,
) -> datasets.DatasetDict:
    """
    Load annotation dataset from S3 and convert to HuggingFace Dataset format.

    Args:
        s3_bucket: S3 bucket name
        training_dataset_s3_path: S3 path to the training dataset JSON file
        documents_s3_path: S3 path prefix for document files

    Returns:
        datasets.DatasetDict with 'train' and 'validation' splits (95%/5%)
    """

    def annotation_generator():
        annotation_dataset = AnnotationDataset(
            s3_bucket=s3_bucket,
            training_dataset_s3_path=training_dataset_s3_path,
            documents_s3_path=documents_s3_path,
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
