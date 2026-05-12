from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .symbolic_reasoner import atom, get_parts_summary, get_traits


def _fmt_conf(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "0.0000"


def _short_path(path: Any) -> str:
    return str(path) if path else "N/A"


def print_demo(data: Dict[str, Any], result: Dict[str, Any], source_path: Path | None = None) -> None:
    traits = get_traits(data)
    parts = get_parts_summary(data)
    detections = data.get("yolo", {}).get("detected_parts") or []
    trace = data.get("pipeline_trace") or []
    artifacts = data.get("artifacts") or {}

    print("\n=== Demo neuro-simbólica de clasificación de insectos ===\n")
    if source_path:
        print(f"Imagen/JSON usado: {source_path}")
    if data.get("image_path"):
        print(f"Imagen original: {data['image_path']}")
    print(f"Image ID: {result['image_id']}")
    print(f"Motor de razonamiento: {result['engine']}")
    if data.get("classifier_input"):
        print(f"Entrada global a MobileNet: {data['classifier_input']}")
    if data.get("pipeline_mode"):
        print(f"Modo de visión: {data['pipeline_mode']}")

    if trace:
        print("\n[0] Traza general del pipeline")
        for step in trace:
            print(f"- {step}")

    print("\n[1] Evidencia visual generada por MobileNet")
    print(f"- Especie CNN: {result['cnn_species']} ({result['cnn_confidence']:.4f})")

    top_k = data.get("mobilenet", {}).get("top_k_insects") or []
    if top_k:
        print("- Top-k especies:")
        for rank, item in enumerate(top_k, start=1):
            print(f"  {rank}. {item.get('label')}: {_fmt_conf(item.get('confidence', 0.0))}")

    print("\n[2] Rasgos morfológicos principales usados como evidencia")
    raw_traits = data.get("mobilenet", {}).get("predicted_traits", {})
    for group, (label, conf) in traits.items():
        info = raw_traits.get(group) or raw_traits.get(str(group)) or {}
        source = info.get("source")
        part = info.get("part")
        if source and part:
            print(f"- {group}: {label} ({conf:.4f}) | fuente={source}, región={part}")
        else:
            print(f"- {group}: {label} ({conf:.4f})")

    part_crop_predictions = data.get("mobilenet", {}).get("part_crop_predictions") or []
    if part_crop_predictions:
        print("\n[2.1] MobileNet por regiones detectadas por YOLO")
        for i, item in enumerate(part_crop_predictions, start=1):
            part = atom(item.get("part", "unknown"))
            det_conf = _fmt_conf(item.get("detection_confidence", 0.0))
            heads = item.get("relevant_heads") or []
            crop_path = item.get("crop_path") or "N/A"
            print(f"- Crop #{i}: parte={part}, conf_yolo={det_conf}, cabezas_leídas={heads}, archivo={crop_path}")
            pred_traits = item.get("mobilenet_prediction", {}).get("predicted_traits", {})
            for head in heads:
                head_info = pred_traits.get(head, {})
                if head_info:
                    print(f"  · {head}: {head_info.get('label')} ({_fmt_conf(head_info.get('confidence', 0.0))})")

    print("\n[3] Resumen de partes detectadas por YOLO")
    if parts:
        for part, info in parts.items():
            if isinstance(info, dict):
                print(
                    f"- {atom(part)}: count={info.get('count', 0)}, "
                    f"max_confidence={_fmt_conf(info.get('max_confidence', 0.0))}"
                )
            else:
                print(f"- {atom(part)}: count={info}")
    else:
        print("- No hay partes YOLO disponibles.")

    print("\n[3.1] Detecciones YOLO individuales")
    if detections:
        for i, det in enumerate(detections, start=1):
            part = atom(det.get("part", "unknown"))
            conf = _fmt_conf(det.get("confidence", 0.0))
            box = det.get("box", ["?", "?", "?", "?"])
            print(f"- #{i}: {part} | confidence={conf} | box={box}")
    else:
        print("- No se registraron cajas individuales.")

    print("\n[4] Decisión simbólica")
    print(f"- Caso válido: {result['valid_case']}")
    print(f"- Soporte visual YOLO: {result['visual_support']}")
    print(f"- Requiere revisión: {result['needs_review']}")
    print(f"- Clase final: {result['predicted_class']}")
    print(f"- Especie final: {result['predicted_species']}")

    print("\n[5] Reglas activadas")
    for rule in result.get("activated_rules") or ["Ninguna regla se activó"]:
        print(f"- {rule}")

    print("\n[6] Explicación")
    if result["predicted_class"] not in ("unknown", "review"):
        print(
            f"El sistema predice '{result['predicted_class']}' porque la evidencia visual "
            "cumple una firma morfológica definida en reglas simbólicas y existe soporte "
            "del detector de partes YOLO. En esta corrida, la imagen fue procesada por "
            "YOLO, se eligió la entrada para MobileNet, se generó un JSON de evidencia y "
            "finalmente se aplicaron reglas simbólicas sobre rasgos y partes detectadas."
        )
    else:
        print(
            "El sistema no encontró suficiente soporte simbólico para una clase final; "
            "por eso marca el caso como revisión."
        )

    if artifacts:
        print("\n[7] Evidencia visual guardada en results/")
        readable_names = {
            "input_image": "Imagen original copiada",
            "yolo_detections_image": "Imagen con cajas YOLO",
            "mobilenet_input_crop": "Crop usado por MobileNet",
            "pipeline_trace": "Traza textual del pipeline",
            "mobilenet_part_crops_dir": "Crops individuales por parte",
            "manifest": "Manifiesto de artefactos",
        }
        for key, path in artifacts.items():
            print(f"- {readable_names.get(key, key)}: {_short_path(path)}")

    print("")
