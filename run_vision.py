from __future__ import annotations

import argparse
from pathlib import Path

from src.vision.pipeline import run_visual_pipeline, save_prediction_json


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Ejecuta solo visión: imagen -> YOLO -> MobileNet -> JSON.")
    parser.add_argument("--image", type=Path, default=root / "data" / "demo_images" / "mosquito_demo.jpg")
    parser.add_argument("--yolo-model", type=Path, default=root / "models" / "best_yolo_insect_parts.pt")
    parser.add_argument("--mobilenet-model", type=Path, default=root / "models" / "best_mobilenet_insect_multitask.pth")
    parser.add_argument("--idx-maps", type=Path, default=root / "models" / "idx_maps.json")
    parser.add_argument("--label-maps", type=Path, default=root / "models" / "label_maps.json")
    parser.add_argument("--output", type=Path, default=root / "results" / "vision_outputs" / "last_prediction.json")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "results" / "demo_artifacts")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--region-aware", action="store_true", help="Ejecuta MobileNet sobre crops de partes YOLO y agrega rasgos por región.")
    args = parser.parse_args()

    prediction = run_visual_pipeline(
        image_path=args.image,
        yolo_model_path=args.yolo_model,
        mobilenet_model_path=args.mobilenet_model,
        idx_maps_path=args.idx_maps,
        label_maps_path=args.label_maps,
        yolo_conf=args.yolo_conf,
        device=args.device,
        artifacts_dir=args.artifacts_dir,
        save_artifacts=not args.no_artifacts,
        region_aware=args.region_aware,
    )
    out = save_prediction_json(prediction, args.output)
    print(f"JSON visual guardado en: {out}")
    if prediction.get("artifacts"):
        print("Artefactos visuales generados:")
        for name, path in prediction["artifacts"].items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
