"""
run_demo.py
Demo local mínima para el proyecto de clasificación neuro-simbólica de insectos.

Uso rápido:
    python run_demo.py

Uso con otro JSON:
    python run_demo.py --json data/demo_json/case1.json

Qué hace:
1. Carga un JSON generado por el modelo visual YOLO + MobileNet.
2. Convierte la evidencia visual en facts estilo Prolog/ProbLog.
3. Intenta ejecutar ProbLog si está instalado.
4. Si ProbLog no está instalado, usa un razonador Python equivalente para la demo.
5. Imprime una predicción final explicable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CLASS_LABELS = [
    "mosquito",
    "bee",
    "grasshopper",
    "mantis",
    "butterfly",
    "lady_beetle",
]

SPECIES_LABELS = [
    "asian_tiger_mosquito",
    "honey_bee",
    "carolina_grasshopper",
    "european_mantid",
    "monarch_butterfly",
    "sevenspotted_lady_beetle",
]

SPECIES_TO_CLASS = {
    "asian_tiger_mosquito": "mosquito",
    "honey_bee": "bee",
    "carolina_grasshopper": "grasshopper",
    "european_mantid": "mantis",
    "monarch_butterfly": "butterfly",
    "sevenspotted_lady_beetle": "lady_beetle",
}

# Reglas equivalentes a json_multiclass_rules.pl.
# Se dejan también en Python para que la demo funcione incluso si ProbLog no está instalado.
SIGNATURE_RULES = {
    "mosquito": [
        ("wing_count", "one_pair", 0.70),
        ("forewing_type", "membranous", 0.70),
        ("mouthpart_type", "piercing_sucking", 0.70),
        ("body_shape", "slender", 0.70),
    ],
    "bee": [
        ("wing_count", "two_pairs", 0.70),
        ("forewing_type", "membranous", 0.70),
        ("mouthpart_type", "chewing", 0.70),
        ("waist_shape", "narrow_waist", 0.70),
    ],
    "grasshopper": [
        ("wing_count", "two_pairs", 0.70),
        ("forewing_type", "tegmina", 0.70),
        ("mouthpart_type", "chewing", 0.70),
        ("leg_specialization", "jumping", 0.70),
    ],
    "mantis": [
        ("wing_count", "two_pairs", 0.70),
        ("forewing_type", "tegmina", 0.70),
        ("mouthpart_type", "chewing", 0.70),
        ("leg_specialization", "grasping", 0.70),
    ],
    "butterfly": [
        ("wing_count", "two_pairs", 0.70),
        ("forewing_type", "scaly", 0.70),
        ("mouthpart_type", "siphoning", 0.70),
        ("antenna_type", "clavate", 0.70),
    ],
    "lady_beetle": [
        ("wing_count", "two_pairs", 0.70),
        ("forewing_type", "elytra", 0.70),
        ("body_shape", "rounded_domed", 0.70),
    ],
}


def atom(text: Any) -> str:
    """Convierte etiquetas a formato de átomo Prolog seguro."""
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "_")
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el JSON de demo: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"El JSON está vacío: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_visual_prediction(data: Dict[str, Any]) -> Tuple[str, float]:
    pred = data.get("mobilenet", {}).get("predicted_insect", {})
    label = atom(pred.get("label", "unknown"))
    confidence = float(pred.get("confidence", 0.0))
    return label, confidence


def get_traits(data: Dict[str, Any]) -> Dict[str, Tuple[str, float]]:
    traits = {}
    predicted_traits = data.get("mobilenet", {}).get("predicted_traits", {})
    for group, info in predicted_traits.items():
        traits[atom(group)] = (atom(info.get("label", "unknown")), float(info.get("confidence", 0.0)))
    return traits


def get_parts_summary(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    yolo = data.get("yolo", {})
    return yolo.get("parts_summary") or yolo.get("part_counts") or {}


def build_facts(data: Dict[str, Any]) -> Tuple[List[str], str, str, float]:
    image_id = atom(data.get("image_id", "demo_image"))
    cnn_species, cnn_conf = get_visual_prediction(data)
    traits = get_traits(data)
    parts_summary = get_parts_summary(data)

    facts = [f"cnn_insect({image_id}, {cnn_species}, {cnn_conf})."]

    for group, (label, conf) in traits.items():
        facts.append(f"trait({image_id}, {group}, {label}, {conf}).")

    for part_name, info in parts_summary.items():
        part = atom(part_name)
        if isinstance(info, dict):
            count = int(info.get("count", 0))
            max_conf = float(info.get("max_confidence", 0.0))
        else:
            count = int(info)
            max_conf = 1.0
        facts.append(f"part_summary({image_id}, {part}, {count}, {max_conf}).")
        if count > 0:
            facts.append(f"part_seen({image_id}, {part}).")

    return facts, image_id, cnn_species, cnn_conf


def build_problog_program(data: Dict[str, Any], rules_path: Path) -> str:
    facts, image_id, _, _ = build_facts(data)
    rules_text = rules_path.read_text(encoding="utf-8") if rules_path.exists() else ""

    queries = [
        f"query(valid_case({image_id})).",
        f"query(visual_support({image_id})).",
        f"query(needs_review({image_id})).",
    ]
    for cls in CLASS_LABELS:
        queries.append(f"query(signature_{cls}({image_id})).")
        queries.append(f"query(final_class({image_id}, {cls})).")
    for species in SPECIES_LABELS:
        queries.append(f"query(final_species({image_id}, {species})).")

    return "\n".join([
        "% Facts generados desde el JSON de la demo",
        *facts,
        "",
        "% Reglas simbólicas",
        rules_text,
        "",
        "% Consultas",
        *queries,
        "",
    ])


def parse_problog_output(text: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    pattern = re.compile(r"^\s*(.+?):\s+([0-9.]+)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            results[match.group(1).strip()] = float(match.group(2))
    return results


def try_run_problog(data: Dict[str, Any], rules_path: Path) -> Optional[Dict[str, float]]:
    """Ejecuta ProbLog si está instalado. Si no, regresa None y se usa fallback Python."""
    program_text = build_problog_program(data, rules_path)

    with tempfile.NamedTemporaryFile("w", suffix=".pl", encoding="utf-8", delete=False) as tmp:
        tmp.write(program_text)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "problog", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 0:
            return None
        parsed = parse_problog_output(result.stdout)
        return parsed if parsed else None
    except Exception:
        return None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def python_reasoning(data: Dict[str, Any]) -> Dict[str, Any]:
    image_id = atom(data.get("image_id", "demo_image"))
    cnn_species, cnn_conf = get_visual_prediction(data)
    traits = get_traits(data)
    parts = get_parts_summary(data)

    valid_case = cnn_conf >= 0.50
    visual_support = "insect" in {atom(p) for p in parts.keys()}

    activated_rules: List[str] = []
    if valid_case:
        activated_rules.append("valid_case")
    if visual_support:
        activated_rules.append("visual_support")

    signature_scores: Dict[str, float] = {}
    for cls, requirements in SIGNATURE_RULES.items():
        passed = []
        for group, expected_label, threshold in requirements:
            actual_label, conf = traits.get(group, ("unknown", 0.0))
            passed.append(actual_label == expected_label and conf >= threshold)
        signature_scores[cls] = sum(passed) / len(requirements)
        if all(passed):
            activated_rules.append(f"signature_{cls}")

    predicted_class = "unknown"
    for cls in CLASS_LABELS:
        if f"signature_{cls}" in activated_rules and valid_case and visual_support:
            predicted_class = cls
            activated_rules.append(f"final_class:{cls}")
            break

    predicted_species = "unknown"
    if predicted_class != "unknown" and SPECIES_TO_CLASS.get(cnn_species) == predicted_class and cnn_conf >= 0.60:
        predicted_species = cnn_species
        activated_rules.append(f"final_species:{cnn_species}")

    needs_review = predicted_class == "unknown"
    if needs_review:
        activated_rules.append("needs_review")

    return {
        "engine": "python_fallback",
        "image_id": image_id,
        "cnn_species": cnn_species,
        "cnn_confidence": cnn_conf,
        "predicted_class": predicted_class if not needs_review else "review",
        "predicted_species": predicted_species,
        "valid_case": valid_case,
        "visual_support": visual_support,
        "needs_review": needs_review,
        "signature_scores": signature_scores,
        "activated_rules": activated_rules,
    }


def result_from_problog(data: Dict[str, Any], parsed: Dict[str, float]) -> Dict[str, Any]:
    image_id = atom(data.get("image_id", "demo_image"))
    cnn_species, cnn_conf = get_visual_prediction(data)

    activated_rules: List[str] = []
    for rule in ["valid_case", "visual_support", "needs_review"]:
        if parsed.get(f"{rule}({image_id})", 0.0) >= 0.5:
            activated_rules.append(rule)

    predicted_class = "unknown"
    for cls in CLASS_LABELS:
        if parsed.get(f"signature_{cls}({image_id})", 0.0) >= 0.5:
            activated_rules.append(f"signature_{cls}")
        if parsed.get(f"final_class({image_id}, {cls})", 0.0) >= 0.5:
            predicted_class = cls
            activated_rules.append(f"final_class:{cls}")

    predicted_species = "unknown"
    for species in SPECIES_LABELS:
        if parsed.get(f"final_species({image_id}, {species})", 0.0) >= 0.5:
            predicted_species = species
            activated_rules.append(f"final_species:{species}")

    needs_review = parsed.get(f"needs_review({image_id})", 0.0) >= 0.5

    return {
        "engine": "problog",
        "image_id": image_id,
        "cnn_species": cnn_species,
        "cnn_confidence": cnn_conf,
        "predicted_class": predicted_class if not needs_review else "review",
        "predicted_species": predicted_species,
        "valid_case": parsed.get(f"valid_case({image_id})", 0.0) >= 0.5,
        "visual_support": parsed.get(f"visual_support({image_id})", 0.0) >= 0.5,
        "needs_review": needs_review,
        "signature_scores": {},
        "activated_rules": activated_rules,
    }


def print_demo(data: Dict[str, Any], result: Dict[str, Any], json_path: Path) -> None:
    traits = get_traits(data)
    parts = get_parts_summary(data)

    print("\n=== Demo neuro-simbólica de clasificación de insectos ===\n")
    print(f"JSON usado: {json_path}")
    print(f"Image ID: {result['image_id']}")
    print(f"Motor de razonamiento: {result['engine']}")

    print("\n[1] Evidencia visual generada por MobileNet")
    print(f"- Especie CNN: {result['cnn_species']} ({result['cnn_confidence']:.4f})")

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
        print("- No hay partes YOLO disponibles en este JSON.")

    print("\n[4] Decisión simbólica")
    print(f"- Caso válido: {result['valid_case']}")
    print(f"- Soporte visual YOLO: {result['visual_support']}")
    print(f"- Requiere revisión: {result['needs_review']}")
    print(f"- Clase final: {result['predicted_class']}")
    print(f"- Especie final: {result['predicted_species']}")

    print("\n[5] Reglas activadas")
    if result["activated_rules"]:
        for rule in result["activated_rules"]:
            print(f"- {rule}")
    else:
        print("- Ninguna regla se activó.")

    print("\n[6] Explicación")
    if result["predicted_class"] not in ("unknown", "review"):
        print(
            f"El sistema predice '{result['predicted_class']}' porque la evidencia visual "
            "cumple una firma morfológica definida en reglas simbólicas y además existe "
            "soporte visual del detector de partes."
        )
    else:
        print(
            "El sistema no encontró suficiente soporte simbólico para una clase final. "
            "Por eso marca el caso como revisión."
        )
    print("")


def main() -> None:
    root = Path(__file__).resolve().parent
    default_json = root / "data" / "demo_json" / "case1.json"
    default_rules = root / "src" / "reasoning" / "json_multiclass_rules.pl"

    parser = argparse.ArgumentParser(description="Demo local del clasificador neuro-simbólico de insectos.")
    parser.add_argument("--json", type=Path, default=default_json, help="Ruta al JSON generado por el modelo visual.")
    parser.add_argument("--rules", type=Path, default=default_rules, help="Ruta al archivo de reglas ProbLog.")
    parser.add_argument("--force-python", action="store_true", help="Usa el razonador Python aunque ProbLog esté instalado.")
    args = parser.parse_args()

    data = load_json(args.json)

    parsed = None if args.force_python else try_run_problog(data, args.rules)
    if parsed is not None:
        result = result_from_problog(data, parsed)
    else:
        result = python_reasoning(data)

    print_demo(data, result, args.json)


if __name__ == "__main__":
    main()
