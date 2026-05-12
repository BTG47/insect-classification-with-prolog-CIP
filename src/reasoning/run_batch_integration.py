import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

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

EXPECTED_SPECIES_MAP = {
    "aedes_albopictus": "asian_tiger_mosquito",
    "apis_mellifera": "honey_bee",
    "dissosteira_carolina": "carolina_grasshopper",
    "mantis_religiosa": "european_mantid",
    "danaus_plexippus": "monarch_butterfly",
    "coccinella_septempunctata": "sevenspotted_lady_beetle",
}

SPECIES_TO_CLASS = {
    "asian_tiger_mosquito": "mosquito",
    "honey_bee": "bee",
    "carolina_grasshopper": "grasshopper",
    "european_mantid": "mantis",
    "monarch_butterfly": "butterfly",
    "sevenspotted_lady_beetle": "lady_beetle",
}

RULE_QUERY_NAMES = [
    "valid_case",
    "visual_support",
    "signature_mosquito",
    "signature_bee",
    "signature_grasshopper",
    "signature_mantis",
    "signature_butterfly",
    "signature_lady_beetle",
    "needs_review",
]


def atom(text: str) -> str:
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_facts(data: dict):
    facts = []
    image_id = atom(data["image_id"])

    pred_insect = data["mobilenet"]["predicted_insect"]
    insect_label = atom(pred_insect["label"])
    insect_conf = float(pred_insect["confidence"])
    facts.append(f"cnn_insect({image_id}, {insect_label}, {insect_conf}).")

    predicted_traits = data["mobilenet"]["predicted_traits"]
    for trait_name, trait_info in predicted_traits.items():
        trait_group = atom(trait_name)
        trait_label = atom(trait_info["label"])
        trait_conf = float(trait_info["confidence"])
        facts.append(
            f"trait({image_id}, {trait_group}, {trait_label}, {trait_conf})."
        )

    yolo = data.get("yolo", {})
    parts_data = yolo.get("parts_summary") or yolo.get("part_counts") or {}
    for part_name, info in parts_data.items():
        part_atom = atom(part_name)
        if isinstance(info, dict):
            count = int(info.get("count", 0))
            max_conf = float(info.get("max_confidence", 0.0))
        else:
            count = int(info)
            max_conf = 1.0
        facts.append(
            f"part_summary({image_id}, {part_atom}, {count}, {max_conf})."
        )
        if count > 0:
            facts.append(f"part_seen({image_id}, {part_atom}).")

    return facts, image_id, insect_label, insect_conf


def build_program(facts, rules_text, image_id):
    lines = []
    lines.append("% =====================================")
    lines.append("% Facts generados desde JSON")
    lines.append("% =====================================")
    lines.extend(facts)
    lines.append("")
    lines.append("% =====================================")
    lines.append("% Reglas simbólicas")
    lines.append("% =====================================")
    lines.append(rules_text)
    lines.append("")
    lines.append("% =====================================")
    lines.append("% Consultas")
    lines.append("% =====================================")

    for rule_name in RULE_QUERY_NAMES:
        lines.append(f"query({rule_name}({image_id})).")

    for cls in CLASS_LABELS:
        lines.append(f"query(final_class({image_id}, {cls})).")

    for sp in SPECIES_LABELS:
        lines.append(f"query(final_species({image_id}, {sp})).")

    return "\n".join(lines)


def parse_problog_output(text: str) -> dict:
    results = {}
    pattern = re.compile(r"^\s*(.+?):\s+([0-9.]+)\s*$")
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            query = m.group(1).strip()
            value = float(m.group(2))
            results[query] = value
    return results


def expected_from_image_id(image_id: str):
    prefix = image_id.split("_img_")[0]
    expected_species = EXPECTED_SPECIES_MAP.get(prefix, "unknown")
    expected_class = SPECIES_TO_CLASS.get(expected_species, "unknown")
    return prefix, expected_species, expected_class



def get_predicted_class(results: dict, image_id: str):
    for cls in CLASS_LABELS:
        q = f"final_class({image_id}, {cls})"
        if results.get(q, 0.0) >= 0.5:
            return cls

    signature_map = {
        "signature_mosquito": "mosquito",
        "signature_bee": "bee",
        "signature_grasshopper": "grasshopper",
        "signature_mantis": "mantis",
        "signature_butterfly": "butterfly",
        "signature_lady_beetle": "lady_beetle",
    }

    valid = results.get(f"valid_case({image_id})", 0.0) >= 0.5
    visual = results.get(f"visual_support({image_id})", 0.0) >= 0.5

    if valid and visual:
        for sig, cls in signature_map.items():
            q = f"{sig}({image_id})"
            if results.get(q, 0.0) >= 0.5:
                return cls

    if results.get(f"needs_review({image_id})", 0.0) >= 0.5:
        return "review"

    return "unknown"



def get_predicted_species(results: dict, image_id: str, cnn_species=None, cnn_conf=0.0, predicted_class="unknown"):
    for sp in SPECIES_LABELS:
        q = f"final_species({image_id}, {sp})"
        if results.get(q, 0.0) >= 0.5:
            return sp

    if cnn_species and predicted_class not in ("unknown", "review"):
        if SPECIES_TO_CLASS.get(cnn_species) == predicted_class and float(cnn_conf) >= 0.60:
            return cnn_species

    return "unknown"

    for sp in SPECIES_LABELS:
        q = f"final_species({image_id}, {sp})"
        if results.get(q, 0.0) >= 0.5:
            return sp
    return "unknown"


def get_activated_rules(results: dict, image_id: str):
    activated = []
    for rule_name in RULE_QUERY_NAMES:
        q = f"{rule_name}({image_id})"
        if results.get(q, 0.0) >= 0.5:
            activated.append(rule_name)

    for cls in CLASS_LABELS:
        q = f"final_class({image_id}, {cls})"
        if results.get(q, 0.0) >= 0.5:
            activated.append(f"final_class:{cls}")

    for sp in SPECIES_LABELS:
        q = f"final_species({image_id}, {sp})"
        if results.get(q, 0.0) >= 0.5:
            activated.append(f"final_species:{sp}")

    return activated


def class_precision(rows, class_name: str):
    tp = sum(1 for r in rows if r["predicted_class"] == class_name and r["expected_class"] == class_name)
    fp = sum(1 for r in rows if r["predicted_class"] == class_name and r["expected_class"] != class_name)
    denom = tp + fp
    return (tp / denom) if denom > 0 else 0.0


def write_markdown_report(rows, out_md: Path):
    total = len(rows)
    class_correct = sum(1 for r in rows if r["is_class_correct"] == "1")
    species_correct = sum(1 for r in rows if r["is_species_correct"] == "1")

    class_accuracy = (class_correct / total) if total else 0.0
    species_accuracy = (species_correct / total) if total else 0.0

    precisions = {cls: class_precision(rows, cls) for cls in CLASS_LABELS}
    macro_precision = sum(precisions.values()) / len(CLASS_LABELS) if CLASS_LABELS else 0.0

    counter = Counter()
    for r in rows:
        if r["is_class_correct"] == "1":
            for rule in r["activated_rules"].split(";"):
                if rule:
                    counter[rule] += 1

    lines = []
    lines.append("# Análisis de resultados de la capa simbólica")
    lines.append("")
    lines.append(f"- Total de casos evaluados: **{total}**")
    lines.append(f"- Accuracy de clase: **{class_accuracy:.4f}**")
    lines.append(f"- Accuracy de especie: **{species_accuracy:.4f}**")
    lines.append(f"- Macro precision por clase: **{macro_precision:.4f}**")
    lines.append("")
    lines.append("## Precision por clase")
    lines.append("")
    for cls, value in precisions.items():
        lines.append(f"- {cls}: **{value:.4f}**")
    lines.append("")
    lines.append("## Reglas más importantes en clasificaciones correctas")
    lines.append("")
    for rule, count in counter.most_common(10):
        lines.append(f"- {rule}: **{count}** activaciones")
    lines.append("")
    lines.append("## Limitaciones")
    lines.append("")
    lines.append("- El sistema depende de que la estructura del JSON se mantenga estable.")
    lines.append("- Los umbrales de las reglas fueron definidos manualmente.")
    lines.append("- Las reglas usan traits ya calculados por el modelo visual; no se entrenaron predicados `nn(...)` dentro de DeepProbLog.")
    lines.append("- Si faltan partes detectadas por YOLO o traits correctos, la inferencia simbólica puede fallar.")
    lines.append("- El mapeo de especie real a clase simbólica se basa en el prefijo del `image_id`.")
    lines.append("")
    lines.append("## Conclusión")
    lines.append("")
    lines.append("La integración JSON -> facts -> reglas -> consulta funciona y produce inferencias interpretables. Además, permite registrar métricas, reglas activadas y limitaciones del modelo.")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    json_dir = root / "batch_data" / "predictions" / "test_jsons"
    rules_file = root / "json_multiclass_rules.pl"
    programs_dir = root / "outputs" / "programs"
    raw_dir = root / "outputs" / "raw"
    summary_csv = root / "outputs" / "summary_results.csv"
    analysis_md = root / "outputs" / "analysis_results.md"

    if not json_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de JSONs: {json_dir}")

    if not rules_file.exists():
        raise FileNotFoundError(f"No existe el archivo de reglas: {rules_file}")

    programs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rules_text = rules_file.read_text(encoding="utf-8")

    rows = []
    json_files = sorted(
        p for p in json_dir.glob("*.json")
        if p.name != "all_predictions_summary.json"
    )

    for json_file in json_files:
        data = load_json(json_file)
        facts, image_id, cnn_species, cnn_conf = build_facts(data)
        program_text = build_program(facts, rules_text, image_id)

        out_pl = programs_dir / f"{json_file.stem}.pl"
        out_raw = raw_dir / f"{json_file.stem}.txt"
        out_pl.write_text(program_text, encoding="utf-8")

        result = subprocess.run(
            ["python", "-m", "problog", str(out_pl)],
            capture_output=True,
            text=True
        )

        raw_text = "=== STDOUT ===\n" + result.stdout + "\n=== STDERR ===\n" + result.stderr
        out_raw.write_text(raw_text, encoding="utf-8")

        parsed = parse_problog_output(result.stdout)
        source_prefix, expected_species, expected_class = expected_from_image_id(image_id)
        predicted_class = get_predicted_class(parsed, image_id)
        predicted_species = get_predicted_species(parsed, image_id, cnn_species, cnn_conf, predicted_class)
        activated_rules = get_activated_rules(parsed, image_id)

        if predicted_class not in ("unknown", "review"):
            tag = f"final_class:{predicted_class}"
            if tag not in activated_rules:
                activated_rules.append(tag)

        if predicted_species != "unknown":
            tag = f"final_species:{predicted_species}"
            if tag not in activated_rules:
                activated_rules.append(tag)

        rows.append({
            "file_name": json_file.name,
            "image_id": image_id,
            "source_prefix": source_prefix,
            "expected_class": expected_class,
            "expected_species": expected_species,
            "cnn_species": cnn_species,
            "cnn_confidence": f"{cnn_conf:.4f}",
            "predicted_class": predicted_class,
            "predicted_species": predicted_species,
            "valid_case": "1" if parsed.get(f"valid_case({image_id})", 0.0) >= 0.5 else "0",
            "visual_support": "1" if parsed.get(f"visual_support({image_id})", 0.0) >= 0.5 else "0",
            "needs_review": "1" if parsed.get(f"needs_review({image_id})", 0.0) >= 0.5 else "0",
            "activated_rules": ";".join(activated_rules),
            "is_class_correct": "1" if predicted_class == expected_class else "0",
            "is_species_correct": "1" if predicted_species == expected_species else "0",
        })

    fieldnames = [
        "file_name",
        "image_id",
        "source_prefix",
        "expected_class",
        "expected_species",
        "cnn_species",
        "cnn_confidence",
        "predicted_class",
        "predicted_species",
        "valid_case",
        "visual_support",
        "needs_review",
        "activated_rules",
        "is_class_correct",
        "is_species_correct",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown_report(rows, analysis_md)

    total = len(rows)
    class_correct = sum(1 for r in rows if r["is_class_correct"] == "1")
    class_accuracy = (class_correct / total) if total else 0.0

    print(f"Casos evaluados: {total}")
    print(f"Accuracy de clase: {class_accuracy:.4f}")
    print(f"CSV resumen: {summary_csv}")
    print(f"Reporte markdown: {analysis_md}")


if __name__ == "__main__":
    main()