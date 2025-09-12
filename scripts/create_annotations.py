import json
from pathlib import Path

from finetune_llms.annotation import (
    ObligationAnnotationGenerator,
    QuestionAnswerGenerator,
)
from finetune_llms.ollama_client import OllamaClient

target_folder = Path("../data/eng/subset")
ollama_client = OllamaClient(base_url="http://localhost:11434", model="gemma3:12b")

annotations_folder = target_folder / "annotations"
annotations_folder.mkdir(exist_ok=True)
input_files = target_folder.glob("*_eng.json")

annotations_to_create = {
    "qa_annotations": QuestionAnswerGenerator,
    "obligation_annotations": ObligationAnnotationGenerator,
}

for annotations_name, annotations_class in annotations_to_create.items():
    for file_path in input_files:
        print(f"Generating {annotations_name} for {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        content = doc_data.get("content")
        annotations = annotations_class.generate_annotations(
            ollama_client, content, doc_data.get("id")
        )

        annotations_filename = file_path.name.replace(
            "_eng.json", f"eng_{annotations_name}.json"
        )
        annotations_path = annotations_folder / annotations_filename

        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)
