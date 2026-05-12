from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_box(box: Any) -> Tuple[int, int, int, int] | None:
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def save_input_copy(image_path: str | Path, artifacts_dir: str | Path) -> Path:
    artifacts_dir = ensure_dir(artifacts_dir)
    image_path = Path(image_path)
    out = artifacts_dir / f"01_input_image{image_path.suffix.lower() or '.jpg'}"
    shutil.copy2(image_path, out)
    return out


def save_yolo_annotated_image(
    image_path: str | Path,
    detections: Iterable[Dict[str, Any]],
    artifacts_dir: str | Path,
) -> Path:
    """Guarda una imagen con cajas YOLO, etiquetas y confianza.

    Se dibuja con PIL para no depender de funciones internas de Ultralytics.
    """
    artifacts_dir = ensure_dir(artifacts_dir)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        box = _as_box(det.get("box"))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        part = str(det.get("part", "unknown"))
        conf = float(det.get("confidence", 0.0) or 0.0)
        label = f"{part} {conf:.2f}"

        # No fijamos una paleta sofisticada para mantenerlo simple y portable.
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        tx1, ty1, tx2, ty2 = text_bbox
        bg_y1 = max(0, y1 - (ty2 - ty1) - 4)
        bg_y2 = y1
        draw.rectangle((x1, bg_y1, x1 + (tx2 - tx1) + 6, bg_y2), fill="red")
        draw.text((x1 + 3, bg_y1 + 2), label, fill="white", font=font)

    out = artifacts_dir / "02_yolo_detections.jpg"
    image.save(out, quality=95)
    return out


def save_classifier_crop(crop_image: Image.Image, artifacts_dir: str | Path) -> Path:
    artifacts_dir = ensure_dir(artifacts_dir)
    out = artifacts_dir / "03_mobilenet_input_crop.jpg"
    crop_image.save(out, quality=95)
    return out



def save_part_crop(crop_image: Image.Image, artifacts_dir: str | Path, index: int, part_name: str) -> Path:
    artifacts_dir = ensure_dir(artifacts_dir) / "part_crops"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_part = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(part_name).lower()) or "part"
    out = artifacts_dir / f"crop_{index:02d}_{safe_part}.jpg"
    crop_image.save(out, quality=95)
    return out

def save_pipeline_trace(trace: List[str], artifacts_dir: str | Path) -> Path:
    artifacts_dir = ensure_dir(artifacts_dir)
    out = artifacts_dir / "04_pipeline_trace.txt"
    out.write_text("\n".join(trace) + "\n", encoding="utf-8")
    return out


def save_artifacts_manifest(artifacts: Dict[str, str], artifacts_dir: str | Path) -> Path:
    artifacts_dir = ensure_dir(artifacts_dir)
    out = artifacts_dir / "artifacts_manifest.json"
    out.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
