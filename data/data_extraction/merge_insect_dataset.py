import argparse
import csv
import random
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import yaml


CANONICAL_CLASSES = [
    "antennae",
    "body",
    "insect",
    "legs",
    "mouthpart",
    "wings",
]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace(" ", "_")
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def remove_roboflow_suffix(stem: str) -> str:
    """
    Roboflow often produces names like:
    image_001_jpg.rf.abc123

    This returns:
    image_001
    """
    stem = stem.replace("_jpg.rf.", ".rf.")
    stem = stem.replace("_png.rf.", ".rf.")
    stem = stem.replace("_jpeg.rf.", ".rf.")

    if ".rf." in stem:
        stem = stem.split(".rf.")[0]

    return stem


def find_data_yaml(root: Path) -> Path:
    yaml_files = list(root.rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No se encontró data.yaml dentro de {root}")
    return yaml_files[0]


def read_yolo_yaml(yaml_path: Path) -> list[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")

    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]

    if not isinstance(names, list):
        raise ValueError(f"Formato inválido de names en {yaml_path}")

    return [str(x).strip() for x in names]


def build_class_remap(source_classes: list[str], target_classes: list[str]) -> dict[int, int]:
    source_norm = [normalize_text(x) for x in source_classes]
    target_norm = [normalize_text(x) for x in target_classes]

    missing = set(source_norm) - set(target_norm)
    if missing:
        raise ValueError(f"Clases no esperadas encontradas: {missing}")

    remap = {}
    for old_idx, class_name in enumerate(source_norm):
        new_idx = target_norm.index(class_name)
        remap[old_idx] = new_idx

    return remap


def remap_label_file(src_label: Path, dst_label: Path, remap: dict[int, int]) -> None:
    new_lines = []

    if src_label.exists():
        with open(src_label, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            old_cls = int(parts[0])
            if old_cls not in remap:
                raise ValueError(f"Clase {old_cls} no existe en remap para {src_label}")

            new_cls = remap[old_cls]
            new_line = " ".join([str(new_cls)] + parts[1:])
            new_lines.append(new_line)

    dst_label.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_label, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")


def infer_species_from_zip(zip_path: Path) -> str:
    name = zip_path.stem.lower()

    if "aedes" in name:
        return "Aedes albopictus"
    if "apis" in name:
        return "Apis mellifera"
    if "coccinella" in name:
        return "Coccinella septempunctata"
    if "danaus" in name:
        return "Danaus plexippus"
    if "dissosteira" in name:
        return "Dissosteira carolina"
    if "mantis" in name:
        return "Mantis religiosa"

    return zip_path.stem


def read_metadata(metadata_path: Path) -> pd.DataFrame:
    if metadata_path.suffix.lower() in [".xlsx", ".xls"]:
        sheets = pd.read_excel(metadata_path, sheet_name=None)

        # Se intenta usar la hoja que tenga image_id.
        selected = None
        for sheet_name, df in sheets.items():
            cols = [str(c).strip() for c in df.columns]
            if "image_id" in cols:
                selected = df
                break

        if selected is None:
            # fallback: primera hoja
            selected = list(sheets.values())[0]

        df = selected.copy()

    elif metadata_path.suffix.lower() == ".csv":
        df = pd.read_csv(metadata_path)
    else:
        raise ValueError("El archivo de metadata debe ser .xlsx, .xls o .csv")

    df.columns = [str(c).strip().replace("\n", "").replace('"', "") for c in df.columns]

    if "image_id" not in df.columns:
        raise ValueError("La metadata debe tener una columna image_id")

    df["image_id_norm"] = df["image_id"].astype(str).apply(normalize_text)

    if "file_path" in df.columns:
        df["file_name_norm"] = (
            df["file_path"]
            .astype(str)
            .apply(lambda x: normalize_text(Path(x).stem))
        )
    else:
        df["file_name_norm"] = ""

    return df


def find_metadata_row(metadata_df: pd.DataFrame, image_stem: str):
    base = remove_roboflow_suffix(image_stem)
    base_norm = normalize_text(base)
    stem_norm = normalize_text(image_stem)

    by_image_id = metadata_df[metadata_df["image_id_norm"] == base_norm]
    if len(by_image_id) > 0:
        return by_image_id.iloc[0].to_dict()

    by_file_name = metadata_df[metadata_df["file_name_norm"] == base_norm]
    if len(by_file_name) > 0:
        return by_file_name.iloc[0].to_dict()

    by_file_name_full = metadata_df[metadata_df["file_name_norm"] == stem_norm]
    if len(by_file_name_full) > 0:
        return by_file_name_full.iloc[0].to_dict()

    return None


def collect_dataset_records(zip_files: list[Path], metadata_df: pd.DataFrame, temp_root: Path):
    records = []
    yaml_class_reference = None

    for zip_path in zip_files:
        species = infer_species_from_zip(zip_path)
        prefix = normalize_text(species)

        extract_dir = temp_root / prefix
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcesando ZIP: {zip_path.name}")
        print(f"Especie inferida: {species}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        data_yaml = find_data_yaml(extract_dir)
        source_classes = read_yolo_yaml(data_yaml)

        if yaml_class_reference is None:
            yaml_class_reference = source_classes

        remap = build_class_remap(source_classes, CANONICAL_CLASSES)

        found_images = 0

        for split in ["train", "valid", "test"]:
            images_dir = extract_dir / split / "images"
            labels_dir = extract_dir / split / "labels"

            if not images_dir.exists():
                continue

            for img_path in images_dir.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                label_path = labels_dir / f"{img_path.stem}.txt"

                if not label_path.exists():
                    print(f"Advertencia: imagen sin label: {img_path.name}")
                    continue

                meta = find_metadata_row(metadata_df, img_path.stem)

                clean_id = remove_roboflow_suffix(img_path.stem)
                new_stem = f"{prefix}_{normalize_text(clean_id)}"

                record = {
                    "original_zip": zip_path.name,
                    "original_split": split,
                    "species_inferred_from_zip": species,
                    "image_path": img_path,
                    "label_path": label_path,
                    "new_stem": new_stem,
                    "extension": img_path.suffix.lower(),
                    "class_remap": remap,
                }

                if meta is not None:
                    for k, v in meta.items():
                        if k not in ["image_id_norm", "file_name_norm"]:
                            record[f"metadata_{k}"] = v
                else:
                    record["metadata_image_id"] = clean_id
                    record["metadata_species"] = species
                    record["metadata_notes"] = "No se encontró match exacto en metadata original."

                records.append(record)
                found_images += 1

        print(f"Imágenes encontradas: {found_images}")

    return records


def stratified_split(records, train_ratio=0.70, valid_ratio=0.20, test_ratio=0.10, seed=42):
    random.seed(seed)

    groups = {}
    for record in records:
        species = record.get("metadata_species", None)
        if pd.isna(species) or species is None:
            species = record["species_inferred_from_zip"]

        species = str(species)
        groups.setdefault(species, []).append(record)

    final_records = []

    for species, group_records in groups.items():
        random.shuffle(group_records)

        n = len(group_records)
        n_train = round(n * train_ratio)
        n_valid = round(n * valid_ratio)

        # Asegurar al menos 1 valid y 1 test si hay suficientes imágenes.
        if n >= 10:
            n_valid = max(1, n_valid)
            n_test = max(1, n - n_train - n_valid)

            while n_train + n_valid + n_test > n:
                n_train -= 1

            while n_train + n_valid + n_test < n:
                n_train += 1
        else:
            n_test = n - n_train - n_valid

        train_records = group_records[:n_train]
        valid_records = group_records[n_train:n_train + n_valid]
        test_records = group_records[n_train + n_valid:]

        for r in train_records:
            r["final_split"] = "train"
            final_records.append(r)

        for r in valid_records:
            r["final_split"] = "valid"
            final_records.append(r)

        for r in test_records:
            r["final_split"] = "test"
            final_records.append(r)

        print(
            f"{species}: total={n}, "
            f"train={len(train_records)}, valid={len(valid_records)}, test={len(test_records)}"
        )

    return final_records


def write_combined_dataset(records, output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    used_names = set()
    metadata_rows = []

    for record in records:
        split = record["final_split"]

        new_stem = record["new_stem"]
        if new_stem in used_names:
            counter = 2
            candidate = f"{new_stem}_{counter}"
            while candidate in used_names:
                counter += 1
                candidate = f"{new_stem}_{counter}"
            new_stem = candidate

        used_names.add(new_stem)

        dst_img = output_dir / split / "images" / f"{new_stem}{record['extension']}"
        dst_label = output_dir / split / "labels" / f"{new_stem}.txt"

        shutil.copy2(record["image_path"], dst_img)
        remap_label_file(record["label_path"], dst_label, record["class_remap"])

        row = {
            "merged_image_id": new_stem,
            "merged_file_path": str(dst_img.relative_to(output_dir)).replace("\\", "/"),
            "merged_label_path": str(dst_label.relative_to(output_dir)).replace("\\", "/"),
            "split": split,
            "original_zip": record["original_zip"],
            "original_split": record["original_split"],
            "species_inferred_from_zip": record["species_inferred_from_zip"],
        }

        for k, v in record.items():
            if k.startswith("metadata_"):
                clean_key = k.replace("metadata_", "")
                row[clean_key] = v

        metadata_rows.append(row)

    metadata_csv = output_dir / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_csv, index=False, encoding="utf-8-sig")

    data_yaml = output_dir / "data.yaml"
    yaml_content = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES,
    }

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False, allow_unicode=True)

    return metadata_csv, data_yaml


def export_morphology_templates(metadata_path: Path, output_dir: Path):
    """
    Si el Excel tiene varias hojas, exporta cada hoja adicional como CSV.
    Esto es útil para conservar la plantilla morfológica por especie.
    """
    if metadata_path.suffix.lower() not in [".xlsx", ".xls"]:
        return

    sheets = pd.read_excel(metadata_path, sheet_name=None)

    templates_dir = output_dir / "metadata_tables"
    templates_dir.mkdir(parents=True, exist_ok=True)

    for sheet_name, df in sheets.items():
        safe_name = normalize_text(sheet_name) or "sheet"
        out_csv = templates_dir / f"{safe_name}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def verify_dataset(output_dir: Path):
    print("\nVerificación final:")

    for split in ["train", "valid", "test"]:
        images_dir = output_dir / split / "images"
        labels_dir = output_dir / split / "labels"

        images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        labels = list(labels_dir.glob("*.txt"))

        print(f"{split}: imágenes={len(images)}, labels={len(labels)}")

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing_labels = image_stems - label_stems
        orphan_labels = label_stems - image_stems

        if missing_labels:
            print(f"  Advertencia: imágenes sin label: {len(missing_labels)}")

        if orphan_labels:
            print(f"  Advertencia: labels sin imagen: {len(orphan_labels)}")

    class_counts = {name: 0 for name in CANONICAL_CLASSES}

    for label_file in output_dir.rglob("labels/*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= len(CANONICAL_CLASSES):
                    raise ValueError(f"Clase fuera de rango en {label_file}: {cls_id}")

                coords = list(map(float, parts[1:]))
                if len(coords) != 4:
                    raise ValueError(f"Label inválido en {label_file}: {line}")

                if any(c < 0 or c > 1 for c in coords):
                    raise ValueError(f"Coordenadas fuera de rango en {label_file}: {line}")

                class_counts[CANONICAL_CLASSES[cls_id]] += 1

    print("\nCajas por clase:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")


def zip_output_folder(output_dir: Path):
    zip_path = output_dir.with_suffix(".zip")

    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(
        base_name=str(output_dir),
        format="zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )

    print(f"\nZIP final creado en: {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Une múltiples datasets YOLOv8 de Roboflow y metadata de Excel/CSV en un dataset final."
    )

    parser.add_argument(
        "--zips",
        nargs="+",
        required=True,
        help="Rutas de los archivos .zip exportados desde Roboflow."
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Ruta del Excel o CSV con la base de datos de imágenes."
    )

    parser.add_argument(
        "--output",
        default="insect_parts_dataset_combined",
        help="Carpeta de salida del dataset combinado."
    )

    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--valid", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip-output", action="store_true")

    args = parser.parse_args()

    total_ratio = args.train + args.valid + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Los porcentajes train + valid + test deben sumar 1.0")

    zip_files = [Path(p) for p in args.zips]
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output)

    for z in zip_files:
        if not z.exists():
            raise FileNotFoundError(f"No existe ZIP: {z}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe metadata: {metadata_path}")

    metadata_df = read_metadata(metadata_path)

    with tempfile.TemporaryDirectory() as tmp:
        temp_root = Path(tmp)

        records = collect_dataset_records(
            zip_files=zip_files,
            metadata_df=metadata_df,
            temp_root=temp_root,
        )

        print(f"\nTotal de imágenes recolectadas: {len(records)}")

        records = stratified_split(
            records,
            train_ratio=args.train,
            valid_ratio=args.valid,
            test_ratio=args.test,
            seed=args.seed,
        )

        metadata_csv, data_yaml = write_combined_dataset(records, output_dir)
        export_morphology_templates(metadata_path, output_dir)
        verify_dataset(output_dir)

        print(f"\nDataset final creado en: {output_dir}")
        print(f"Metadata final: {metadata_csv}")
        print(f"YAML final: {data_yaml}")

        if args.zip_output:
            zip_output_folder(output_dir)


if __name__ == "__main__":
    main()