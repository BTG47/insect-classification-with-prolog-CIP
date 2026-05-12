from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CLASS_LABELS = ["mosquito", "bee", "grasshopper", "mantis", "butterfly", "lady_beetle"]
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
SIGNATURE_RULES = {
    "mosquito": [("wing_count", "one_pair", 0.70), ("forewing_type", "membranous", 0.70), ("mouthpart_type", "piercing_sucking", 0.70), ("body_shape", "slender", 0.70)],
    "bee": [("wing_count", "two_pairs", 0.70), ("forewing_type", "membranous", 0.70), ("mouthpart_type", "chewing", 0.70), ("waist_shape", "narrow_waist", 0.70)],
    "grasshopper": [("wing_count", "two_pairs", 0.70), ("forewing_type", "tegmina", 0.70), ("mouthpart_type", "chewing", 0.70), ("leg_specialization", "jumping", 0.70)],
    "mantis": [("wing_count", "two_pairs", 0.70), ("forewing_type", "tegmina", 0.70), ("mouthpart_type", "chewing", 0.70), ("leg_specialization", "grasping", 0.70)],
    "butterfly": [("wing_count", "two_pairs", 0.70), ("forewing_type", "scaly", 0.70), ("mouthpart_type", "siphoning", 0.70), ("antenna_type", "clavate", 0.70)],
    "lady_beetle": [("wing_count", "two_pairs", 0.70), ("forewing_type", "elytra", 0.70), ("body_shape", "rounded_domed", 0.70)],
}


def atom(text: Any) -> str:
    return str(text).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "").replace(".", "_")


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_visual_prediction(data: Dict[str, Any]) -> Tuple[str, float]:
    pred = data.get("mobilenet", {}).get("predicted_insect", {})
    return atom(pred.get("label", "unknown")), float(pred.get("confidence", 0.0))


def get_traits(data: Dict[str, Any]) -> Dict[str, Tuple[str, float]]:
    traits: Dict[str, Tuple[str, float]] = {}
    for group, info in data.get("mobilenet", {}).get("predicted_traits", {}).items():
        traits[atom(group)] = (atom(info.get("label", "unknown")), float(info.get("confidence", 0.0)))
    return traits


def get_parts_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    yolo = data.get("yolo", {})
    return yolo.get("parts_summary") or yolo.get("part_counts") or {}


def build_facts(data: Dict[str, Any]) -> Tuple[List[str], str]:
    image_id = atom(data.get("image_id", "demo_image"))
    cnn_species, cnn_conf = get_visual_prediction(data)
    facts = [f"cnn_insect({image_id}, {cnn_species}, {cnn_conf})."]

    for group, (label, conf) in get_traits(data).items():
        facts.append(f"trait({image_id}, {group}, {label}, {conf}).")

    for part_name, info in get_parts_summary(data).items():
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
    return facts, image_id


def build_problog_program(data: Dict[str, Any], rules_path: str | Path) -> str:
    facts, image_id = build_facts(data)
    rules_text = Path(rules_path).read_text(encoding="utf-8") if Path(rules_path).exists() else ""
    queries = [f"query(valid_case({image_id})).", f"query(visual_support({image_id})).", f"query(needs_review({image_id}))."]
    for cls in CLASS_LABELS:
        queries += [f"query(signature_{cls}({image_id})).", f"query(final_class({image_id}, {cls}))."]
    for species in SPECIES_LABELS:
        queries.append(f"query(final_species({image_id}, {species})).")
    return "\n".join(["% Facts generados desde JSON", *facts, "", "% Reglas", rules_text, "", "% Queries", *queries, ""])


def parse_problog_output(text: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    pattern = re.compile(r"^\s*(.+?):\s+([0-9.]+)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            results[match.group(1).strip()] = float(match.group(2))
    return results


def try_run_problog(data: Dict[str, Any], rules_path: str | Path, timeout: int = 20) -> Optional[Dict[str, float]]:
    program_text = build_problog_program(data, rules_path)
    with tempfile.NamedTemporaryFile("w", suffix=".pl", encoding="utf-8", delete=False) as tmp:
        tmp.write(program_text)
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run([sys.executable, "-m", "problog", str(tmp_path)], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        parsed = parse_problog_output(result.stdout)
        return parsed or None
    except Exception:
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def python_reasoning(data: Dict[str, Any]) -> Dict[str, Any]:
    image_id = atom(data.get("image_id", "demo_image"))
    cnn_species, cnn_conf = get_visual_prediction(data)
    traits = get_traits(data)
    parts = {atom(p) for p in get_parts_summary(data).keys()}
    valid_case = cnn_conf >= 0.50
    visual_support = "insect" in parts or bool(parts)
    activated_rules: List[str] = []
    if valid_case:
        activated_rules.append("valid_case")
    if visual_support:
        activated_rules.append("visual_support")

    signature_scores: Dict[str, float] = {}
    for cls, requirements in SIGNATURE_RULES.items():
        passed = []
        for group, expected, threshold in requirements:
            actual, conf = traits.get(group, ("unknown", 0.0))
            passed.append(actual == expected and conf >= threshold)
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


def run_reasoning(data: Dict[str, Any], rules_path: str | Path, force_python: bool = False) -> Dict[str, Any]:
    parsed = None if force_python else try_run_problog(data, rules_path)
    return result_from_problog(data, parsed) if parsed is not None else python_reasoning(data)
