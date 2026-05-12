from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.reasoning.reporting import print_demo
from src.reasoning.symbolic_reasoner import load_json, run_reasoning
from src.vision.pipeline import run_visual_pipeline, save_prediction_json


def default_paths(root: Path) -> Dict[str, Path]:
    return {
        "json": root / "data" / "demo_json" / "case1.json",
        "image": root / "data" / "demo_images" / "mosquito_demo.jpg",
        "rules": root / "src" / "reasoning" / "json_multiclass_rules.pl",
        "yolo": root / "models" / "best_yolo_insect_parts.pt",
        "mobilenet": root / "models" / "best_mobilenet_insect_multitask.pth",
        "idx_maps": root / "models" / "idx_maps.json",
        "label_maps": root / "models" / "label_maps.json",
        "vision_out": root / "results" / "vision_outputs" / "last_prediction.json",
        "reasoning_out": root / "results" / "reasoning_outputs" / "last_reasoning.json",
    }


def save_reasoning_result(result: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = default_paths(root)

    parser = argparse.ArgumentParser(
        description="Demo local del clasificador neuro-simbólico de insectos."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Imagen de entrada. Si se usa, ejecuta YOLO + MobileNet + razonamiento.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSON visual ya generado. Si no se pasa --image, corre la demo rápida desde JSON.",
    )
    parser.add_argument("--rules", type=Path, default=paths["rules"], help="Archivo de reglas ProbLog.")
    parser.add_argument("--yolo-model", type=Path, default=paths["yolo"], help="Pesos YOLO .pt.")
    parser.add_argument("--mobilenet-model", type=Path, default=paths["mobilenet"], help="Pesos MobileNet .pth.")
    parser.add_argument("--idx-maps", type=Path, default=paths["idx_maps"], help="Mapa índice→etiqueta.")
    parser.add_argument("--label-maps", type=Path, default=paths["label_maps"], help="Mapa etiqueta→índice.")
    parser.add_argument("--vision-output", type=Path, default=paths["vision_out"], help="Dónde guardar el JSON visual generado.")
    parser.add_argument("--reasoning-output", type=Path, default=paths["reasoning_out"], help="Dónde guardar la salida simbólica.")
    parser.add_argument("--force-python", action="store_true", help="Usa razonador Python aunque ProbLog esté instalado.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc. Por defecto detecta automático.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="Umbral de confianza para YOLO.")
    args = parser.parse_args()

    # Modo completo: imagen -> YOLO -> crop -> MobileNet -> JSON -> ProbLog/Python.
    if args.image is not None:
        data = run_visual_pipeline(
            image_path=args.image,
            yolo_model_path=args.yolo_model,
            mobilenet_model_path=args.mobilenet_model,
            idx_maps_path=args.idx_maps,
            label_maps_path=args.label_maps,
            yolo_conf=args.yolo_conf,
            device=args.device,
        )
        source_path = save_prediction_json(data, args.vision_output)
    else:
        # Modo seguro y rápido: usa JSON ya generado por los modelos visuales.
        json_path = args.json or paths["json"]
        data = load_json(json_path)
        source_path = json_path

    result = run_reasoning(data, rules_path=args.rules, force_python=args.force_python)
    save_reasoning_result(result, args.reasoning_output)
    print_demo(data, result, source_path=source_path)
    print(f"Salida visual guardada/consultada en: {source_path}")
    print(f"Salida simbólica guardada en: {args.reasoning_output}\n")


if __name__ == "__main__":
    main()
