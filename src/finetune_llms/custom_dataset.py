import json
import os
import shutil
import tempfile
from typing import Any, Dict, Iterator, Set

import datasets


class TrainingDataset:
    """
    Dataset class for loading a training dataset file and creating input/output pairs.

    Loads training dataset files from S3 and finds corresponding source text
    from the original data files.
    """

    def __init__(
        self,
        s3_bucket: str,
        training_dataset_s3_path: str,
        documents_s3_path: str,
        s3_client,
    ):
        """
        Initialize the training dataset.

        Args:
            s3_bucket: S3 bucket name
            training_dataset_s3_path: S3 path to the training dataset JSON file
            documents_s3_path: S3 path prefix for document files
        """
        self.s3_bucket = s3_bucket
        self.training_dataset_s3_path = training_dataset_s3_path
        self.documents_s3_path = documents_s3_path
        self.s3_client = s3_client
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
        training_data = annotation_data.get("training_dataset")

        if not input_field or not output_field:
            raise ValueError("input_field and output_field are required")

        if not training_data:
            raise ValueError("No data found in training dataset")

        # Get unique source documents and download them if needed
        if input_field == "source_text":
            unique_source_docs = self._get_unique_source_documents(training_data)
            if not unique_source_docs:
                raise ValueError("No source documents found in training data items")
            self._download_all_source_documents(unique_source_docs)

        for data_item in training_data:
            # Get input text based on input_field
            if input_field == "source_text":
                # Get source document for this data item
                source_document = data_item.get("source_document")

                # Load source content from local file (no caching)
                source_content = self._load_source_content(source_document)
                start = data_item.get("source_document_start")
                end = data_item.get("source_document_end")
                input_text = source_content[start:end].strip()
            else:
                # Generic handling: get the field directly from data item
                raw_input = data_item.get(input_field)
                input_text = str(raw_input).strip()

            # Get output text based on output_field
            if output_field == "data_item":
                if "source_document" in data_item:
                    del data_item["source_document"]
                if "source_document_start" in data_item:
                    del data_item["source_document_start"]
                if "source_document_end" in data_item:
                    del data_item["source_document_end"]
                output_text = json.dumps(data_item)
            else:
                # Generic handling: get the field directly from annotation
                raw_output = data_item.get(output_field)
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


def load_training_dataset(
    s3_bucket: str, training_dataset_s3_path: str, documents_s3_path: str, s3_client
) -> datasets.DatasetDict:
    """
    Load trainng dataset from S3 and convert to HuggingFace Dataset format.

    Args:
        s3_bucket: S3 bucket name
        training_dataset_s3_path: S3 path to the training dataset JSON file
        documents_s3_path: S3 path prefix for document files

    Returns:
        datasets.DatasetDict with 'train' and 'validation' splits (95%/5%)
    """

    def training_dataset_generator():
        training_dataset = TrainingDataset(
            s3_bucket=s3_bucket,
            training_dataset_s3_path=training_dataset_s3_path,
            documents_s3_path=documents_s3_path,
            s3_client=s3_client,
        )
        for item in training_dataset:
            yield item

    # Convert generator to list to ensure we get a regular Dataset, not IterableDataset
    data_list = list(training_dataset_generator())
    dataset: datasets.Dataset = datasets.Dataset.from_list(data_list)

    # Create 0.05 validation split (smaller since we have fewer examples)
    dataset_dict = dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)  # type: ignore

    return datasets.DatasetDict(
        {"train": dataset_dict["train"], "validation": dataset_dict["test"]}
    )
