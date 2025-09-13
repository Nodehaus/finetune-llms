import json
import time
from pathlib import Path

from finetune_llms.annotation import (
    ObligationAnnotationGenerator,
    QuestionAnswerGenerator,
)
from finetune_llms.ollama_client import OllamaClient
from finetune_llms.utils import get_gpu_info

MODEL = "gemma3:12b"

target_folder = Path("data/eng/subset")
ollama_client = OllamaClient(base_url="http://localhost:11434", model=MODEL)

annotations_folder = target_folder / "annotations"
annotations_folder.mkdir(exist_ok=True)
input_files = target_folder.glob("*_eng.json")

annotations_to_create = {
    "qa": QuestionAnswerGenerator,
    "obligations": ObligationAnnotationGenerator,
}

gpu_info = get_gpu_info()

for file_path in input_files:
    for annotations_id, annotations_class in annotations_to_create.items():
        print(f"Generating {annotations_id} for {file_path}")

        annotations_folder_with_id = annotations_folder / annotations_id
        annotations_folder_with_id.mkdir(exist_ok=True)
        annotations_path = annotations_folder_with_id / file_path.name
        if annotations_path.exists():
            print("  Annotation file exists, skipping...")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        content = doc_data.get("content")
        start_time = time.time()
        annotations = annotations_class.generate_annotations(
            ollama_client, content, doc_data.get("id")
        )
        end_time = time.time()
        inference_time = end_time - start_time
        count_annotations = len(annotations)
        avg_inference_time_per_annotation = inference_time / count_annotations

        data = {
            "model_name": MODEL,
            "model_runner": "ollama",
            "gpu_info": gpu_info,
            "input_field": annotations_class.input_field,
            "output_field": annotations_class.output_field,
            "count_annotations": len(annotations),
            "total_generation_time": round(inference_time, 3),
            "avg_generation_time_per_annotation": round(
                avg_inference_time_per_annotation, 3
            ),
            "annotations_id": annotations_id,
            "annotations": annotations,
        }

        annotations_filename = file_path.name.replace(
            "_eng.json", f"_eng_{annotations_id}.json"
        )
        annotations_path = annotations_folder / annotations_filename

        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
