from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _safe_box(xyxy: Any) -> list[int]:
    try:
        return [int(round(float(v))) for v in xyxy]
    except Exception:
        return [0, 0, 0, 0]


def summarize_parts(detections: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for det in detections:
        part = str(det.get("part", "unknown"))
        conf = _safe_float(det.get("confidence", 0.0))
        if part not in summary:
            summary[part] = {"count": 0, "max_confidence": 0.0}
        summary[part]["count"] += 1
        summary[part]["max_confidence"] = max(summary[part]["max_confidence"], conf)
    return summary


def detect_parts(
    image_path: str | Path,
    model_path: str | Path,
    conf: float = 0.25,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """Ejecuta YOLO sobre una imagen y devuelve detecciones de partes.

    No entrena. Solo carga `best_yolo_insect_parts.pt` y hace inferencia.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depende del ambiente local
        raise RuntimeError(
            "No se pudo importar ultralytics. Instala con: pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    if not results:
        return {"detected_parts": [], "parts_summary": {}, "best_insect_box": None}

    result = results[0]
    names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}
    detections: List[Dict[str, Any]] = []

    for box in result.boxes:
        cls_idx = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
        part_name = str(names.get(cls_idx, cls_idx))
        confidence = _safe_float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])
        xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else box.xyxy[0]
        detections.append({
            "part": part_name,
            "confidence": confidence,
            "box": _safe_box(xyxy),
        })

    insect_boxes = [d for d in detections if str(d["part"]).lower() == "insect"]
    best_insect = max(insect_boxes, key=lambda d: d["confidence"], default=None)

    return {
        "detected_parts": detections,
        "parts_summary": summarize_parts(detections),
        "best_insect_box": best_insect["box"] if best_insect else None,
    }


def crop_for_classifier(image_path: str | Path, yolo_output: Dict[str, Any], padding: int = 12) -> Tuple[Image.Image, str]:
    """Usa el bbox de `insect` como crop para MobileNet; si no existe, usa imagen completa."""
    image = Image.open(image_path).convert("RGB")
    box = yolo_output.get("best_insect_box")
    if not box:
        return image, "full_image"

    w, h = image.size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return image, "full_image_invalid_crop"
    return image.crop((x1, y1, x2, y2)), "yolo_insect_crop"
