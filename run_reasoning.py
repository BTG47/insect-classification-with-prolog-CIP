from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.reasoning.reporting import print_demo
from src.reasoning.symbolic_reasoner import load_json, run_reasoning


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Ejecuta solo la capa simbólica desde un JSON visual.")
    parser.add_argument("--json", type=Path, default=root / "data" / "demo_json" / "case1.json")
    parser.add_argument("--rules", type=Path, default=root / "src" / "reasoning" / "json_multiclass_rules.pl")
    parser.add_argument("--output", type=Path, default=root / "results" / "reasoning_outputs" / "last_reasoning.json")
    parser.add_argument("--force-python", action="store_true")
    args = parser.parse_args()

    data = load_json(args.json)
    result = run_reasoning(data, args.rules, force_python=args.force_python)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print_demo(data, result, source_path=args.json)
    print(f"Salida simbólica guardada en: {args.output}")


if __name__ == "__main__":
    main()
