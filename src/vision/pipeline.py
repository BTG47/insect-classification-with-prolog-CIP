from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from .artifacts import (
    save_artifacts_manifest,
    save_classifier_crop,
    save_input_copy,
    save_part_crop,
    save_pipeline_trace,
    save_yolo_annotated_image,
)
from .mobilenet_multitask import TRAIT_COLUMNS, load_model, predict_image
from .yolo_detector import crop_detection, crop_for_classifier, detect_parts


# Qué cabezas de MobileNet tiene sentido leer para cada región detectada por YOLO.
# Si una parte no aparece aquí, se ignora para la agregación de rasgos, pero queda
# registrada en el JSON y en los artefactos visuales.
PART_TO_TRAIT_HEADS: Dict[str, List[str]] = {
    "wing": ["wing_count", "forewing_type"],
    "wings": ["wing_count", "forewing_type"],
    "forewing": ["forewing_type"],
    "mouthpart": ["mouthpart_type"],
    "mouthparts": ["mouthpart_type"],
    "antenna": ["antenna_type"],
    "antennae": ["antenna_type"],
    "leg": ["leg_specialization"],
    "legs": ["leg_specialization"],
    "body": ["body_shape", "waist_shape"],
    "abdomen": ["body_shape", "waist_shape"],
    "thorax": ["body_shape"],
    # La caja de insecto completo sirve como fallback global, no como parte especializada.
}


def build_image_id(image_path: str | Path) -> str:
    return Path(image_path).stem.lower().replace("-", "_").replace(" ", "_")


def _prediction_for_head(prediction: Dict[str, Any], head: str) -> Dict[str, Any] | None:
    traits = prediction.get("predicted_traits") or {}
    item = traits.get(head)
    if not item:
        return None
    return {
        "label": item.get("label", "unknown"),
        "confidence": float(item.get("confidence", 0.0) or 0.0),
    }


def _merge_region_traits(
    whole_prediction: Dict[str, Any],
    part_predictions: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Agrega rasgos usando crops de partes cuando existen y fallback global.

    La especie se conserva desde el crop del insecto completo. Para rasgos, cada
    región de YOLO solo alimenta las cabezas relacionadas con esa parte. Ejemplo:
    mouthpart -> mouthpart_type; legs -> leg_specialization; body -> body_shape.

    Si no existe crop relevante para una cabeza, se conserva la predicción global.
    """
    resolved: Dict[str, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {"trait_sources": {}}

    for head in TRAIT_COLUMNS:
        global_item = _prediction_for_head(whole_prediction, head) or {
            "label": "unknown",
            "confidence": 0.0,
        }
        best = {
            "label": global_item["label"],
            "confidence": global_item["confidence"],
            "source": "whole_insect_crop",
            "part": "insect",
            "det_confidence": 1.0,
            "combined_score": global_item["confidence"],
        }

        candidates: List[Dict[str, Any]] = []
        for pp in part_predictions:
            part = str(pp.get("part", "unknown")).lower()
            relevant_heads = PART_TO_TRAIT_HEADS.get(part, [])
            if head not in relevant_heads:
                continue
            pred = _prediction_for_head(pp.get("mobilenet_prediction", {}), head)
            if not pred:
                continue
            det_conf = float(pp.get("detection_confidence", 0.0) or 0.0)
            combined = det_conf * float(pred.get("confidence", 0.0) or 0.0)
            candidate = {
                "label": pred["label"],
                "confidence": pred["confidence"],
                "source": "yolo_part_crop",
                "part": part,
                "det_confidence": det_conf,
                "combined_score": combined,
                "box": pp.get("box"),
                "crop_path": pp.get("crop_path"),
            }
            candidates.append(candidate)

        if candidates:
            # Se elige el crop con mejor producto confianza_YOLO * confianza_MobileNet.
            best = max(candidates, key=lambda c: c["combined_score"])

        resolved[head] = {
            "label": best["label"],
            "confidence": best["confidence"],
            "source": best["source"],
            "part": best["part"],
            "det_confidence": best["det_confidence"],
            "combined_score": best["combined_score"],
        }
        debug["trait_sources"][head] = best

    return resolved, debug


def _run_region_aware_predictions(
    image_path: Path,
    detections: List[Dict[str, Any]],
    model: Any,
    idx_maps: Dict[str, Dict[str, str]],
    device: Any,
    artifacts_dir: str | Path | None,
    save_artifacts: bool,
    max_crops_per_part: int = 2,
) -> List[Dict[str, Any]]:
    """Corre MobileNet sobre crops de partes de YOLO y conserva evidencia por región."""
    # Ordena por confianza y limita para que la demo sea rápida y legible.
    counts_by_part: Dict[str, int] = {}
    selected: List[Dict[str, Any]] = []
    for det in sorted(detections, key=lambda d: float(d.get("confidence", 0.0) or 0.0), reverse=True):
        part = str(det.get("part", "unknown")).lower()
        if part not in PART_TO_TRAIT_HEADS:
            continue
        counts_by_part[part] = counts_by_part.get(part, 0) + 1
        if counts_by_part[part] > max_crops_per_part:
            continue
        selected.append(det)

    outputs: List[Dict[str, Any]] = []
    for idx, det in enumerate(selected, start=1):
        crop, crop_kind = crop_detection(image_path, det)
        crop_path = None
        if save_artifacts and artifacts_dir is not None:
            crop_path = str(save_part_crop(crop, artifacts_dir, idx, str(det.get("part", "unknown"))))
        pred = predict_image(crop, model, idx_maps, device)
        outputs.append({
            "part": str(det.get("part", "unknown")),
            "box": det.get("box"),
            "detection_confidence": float(det.get("confidence", 0.0) or 0.0),
            "crop_kind": crop_kind,
            "crop_path": crop_path,
            "relevant_heads": PART_TO_TRAIT_HEADS.get(str(det.get("part", "unknown")).lower(), []),
            "mobilenet_prediction": pred,
        })
    return outputs


def run_visual_pipeline(
    image_path: str | Path,
    yolo_model_path: str | Path,
    mobilenet_model_path: str | Path,
    idx_maps_path: str | Path,
    label_maps_path: str | Path | None = None,
    yolo_conf: float = 0.25,
    device: str | None = None,
    artifacts_dir: str | Path | None = None,
    save_artifacts: bool = True,
    region_aware: bool = False,
) -> Dict[str, Any]:
    """Ejecuta imagen -> YOLO -> MobileNet -> JSON compatible con razonador.

    Modo default:
        YOLO localiza el insecto completo y MobileNet predice especie + rasgos
        sobre ese crop completo.

    Modo region_aware:
        Además del crop completo, MobileNet se ejecuta sobre crops de partes de
        YOLO. Cada parte solo alimenta las cabezas pertinentes y luego se agregan
        los rasgos con fallback al crop completo. Esta ruta replica mejor la
        arquitectura conceptual YOLO -> regiones -> MobileNet por rasgo.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    trace: list[str] = []
    artifacts: Dict[str, str] = {}

    trace.append(f"1. Imagen cargada: {image_path}")
    if save_artifacts and artifacts_dir is not None:
        artifacts["input_image"] = str(save_input_copy(image_path, artifacts_dir))

    yolo_output = detect_parts(image_path, yolo_model_path, conf=yolo_conf)
    detections = yolo_output.get("detected_parts", [])
    trace.append(f"2. YOLO ejecutado: {len(detections)} detecciones con umbral {yolo_conf}")
    if save_artifacts and artifacts_dir is not None:
        artifacts["yolo_detections_image"] = str(save_yolo_annotated_image(image_path, detections, artifacts_dir))

    classifier_image, classifier_input = crop_for_classifier(image_path, yolo_output)
    trace.append(f"3. Entrada global seleccionada para MobileNet: {classifier_input}")
    if save_artifacts and artifacts_dir is not None:
        artifacts["mobilenet_input_crop"] = str(save_classifier_crop(classifier_image, artifacts_dir))

    mobilenet_model, idx_maps, used_device = load_model(
        model_path=mobilenet_model_path,
        idx_maps_path=idx_maps_path,
        label_maps_path=label_maps_path,
        device=device,
    )
    trace.append(f"4. MobileNet cargado en dispositivo: {used_device}")

    whole_prediction = predict_image(classifier_image, mobilenet_model, idx_maps, used_device)
    top1 = whole_prediction.get("predicted_insect", {})
    trace.append(
        f"5. MobileNet sobre crop global predijo especie: {top1.get('label', 'unknown')} "
        f"({float(top1.get('confidence', 0.0) or 0.0):.4f})"
    )

    part_crop_predictions: List[Dict[str, Any]] = []
    region_debug: Dict[str, Any] = {"trait_sources": {}}
    mobilenet_output = whole_prediction
    pipeline_mode = "whole_insect_crop"

    if region_aware:
        pipeline_mode = "region_aware_part_crops"
        part_crop_predictions = _run_region_aware_predictions(
            image_path=image_path,
            detections=detections,
            model=mobilenet_model,
            idx_maps=idx_maps,
            device=used_device,
            artifacts_dir=artifacts_dir,
            save_artifacts=save_artifacts,
        )
        trace.append(
            f"6. Modo region-aware activo: MobileNet evaluó {len(part_crop_predictions)} crops de partes YOLO"
        )
        resolved_traits, region_debug = _merge_region_traits(whole_prediction, part_crop_predictions)
        mobilenet_output = dict(whole_prediction)
        mobilenet_output["predicted_traits"] = resolved_traits
        mobilenet_output["whole_insect_prediction"] = whole_prediction
        mobilenet_output["part_crop_predictions"] = part_crop_predictions
        mobilenet_output["region_aware_debug"] = region_debug
        if save_artifacts and artifacts_dir is not None and part_crop_predictions:
            artifacts["mobilenet_part_crops_dir"] = str(Path(artifacts_dir) / "part_crops")
    else:
        trace.append("6. Modo global: los rasgos se leen desde el crop completo del insecto")

    trace.append("7. JSON visual estructurado generado para el razonador simbólico")

    if save_artifacts and artifacts_dir is not None:
        artifacts["pipeline_trace"] = str(save_pipeline_trace(trace, artifacts_dir))
        artifacts["manifest"] = str(save_artifacts_manifest(artifacts, artifacts_dir))

    return {
        "image_id": build_image_id(image_path),
        "image_path": str(image_path),
        "classifier_input": classifier_input,
        "pipeline_mode": pipeline_mode,
        "pipeline_trace": trace,
        "artifacts": artifacts,
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
