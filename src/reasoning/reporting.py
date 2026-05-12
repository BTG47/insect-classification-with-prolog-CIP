from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .symbolic_reasoner import atom, get_parts_summary, get_traits


def print_demo(data: Dict[str, Any], result: Dict[str, Any], source_path: Path | None = None) -> None:
    traits = get_traits(data)
    parts = get_parts_summary(data)

    print("\n=== Demo neuro-simbólica de clasificación de insectos ===\n")
    if source_path:
        label = "Imagen/JSON usado"
        print(f"{label}: {source_path}")
    print(f"Image ID: {result['image_id']}")
    print(f"Motor de razonamiento: {result['engine']}")
    if data.get("classifier_input"):
        print(f"Entrada a MobileNet: {data['classifier_input']}")

    print("\n[1] Evidencia visual generada por MobileNet")
    print(f"- Especie CNN: {result['cnn_species']} ({result['cnn_confidence']:.4f})")

    top_k = data.get("mobilenet", {}).get("top_k_insects") or []
    if top_k:
        print("- Top-k especies:")
        for item in top_k:
            print(f"  · {item.get('label')}: {float(item.get('confidence', 0.0)):.4f}")

    print("\n[2] Rasgos morfológicos principales")
    for group, (label, conf) in traits.items():
        print(f"- {group}: {label} ({conf:.4f})")

    print("\n[3] Partes detectadas por YOLO")
    if parts:
        for part, info in parts.items():
            if isinstance(info, dict):
                print(f"- {atom(part)}: count={info.get('count', 0)}, max_confidence={float(info.get('max_confidence', 0.0)):.4f}")
            else:
                print(f"- {atom(part)}: count={info}")
    else:
        print("- No hay partes YOLO disponibles.")

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
            "del detector de partes YOLO."
        )
    else:
        print(
            "El sistema no encontró suficiente soporte simbólico para una clase final; "
            "por eso marca el caso como revisión."
        )
    print("")
