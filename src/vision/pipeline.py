from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .mobilenet_multitask import load_model, predict_image
from .yolo_detector import crop_for_classifier, detect_parts


def build_image_id(image_path: str | Path) -> str:
    return Path(image_path).stem.lower().replace("-", "_").replace(" ", "_")


def run_visual_pipeline(
    image_path: str | Path,
    yolo_model_path: str | Path,
    mobilenet_model_path: str | Path,
    idx_maps_path: str | Path,
    label_maps_path: str | Path | None = None,
    yolo_conf: float = 0.25,
    device: str | None = None,
) -> Dict[str, Any]:
    """Ejecuta imagen -> YOLO -> crop -> MobileNet -> JSON compatible con razonador."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    yolo_output = detect_parts(image_path, yolo_model_path, conf=yolo_conf)
    classifier_image, classifier_input = crop_for_classifier(image_path, yolo_output)

    mobilenet_model, idx_maps, used_device = load_model(
        model_path=mobilenet_model_path,
        idx_maps_path=idx_maps_path,
        label_maps_path=label_maps_path,
        device=device,
    )
    mobilenet_output = predict_image(classifier_image, mobilenet_model, idx_maps, used_device)

    return {
        "image_id": build_image_id(image_path),
        "image_path": str(image_path),
        "classifier_input": classifier_input,
        "mobilenet": mobilenet_output,
        "yolo": {
            "detected_parts": yolo_output["detected_parts"],
            "parts_summary": yolo_output["parts_summary"],
        },
    }


def save_prediction_json(prediction: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
