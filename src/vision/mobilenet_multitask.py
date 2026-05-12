from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from PIL import Image


IMG_SIZE = 224
OUTPUT_COLUMNS = [
    "insect_label",
    "wing_count",
    "forewing_type",
    "mouthpart_type",
    "antenna_type",
    "leg_specialization",
    "body_shape",
    "waist_shape",
]
TRAIT_COLUMNS = [c for c in OUTPUT_COLUMNS if c != "insect_label"]


class MultiHeadMobileNet(nn.Module):
    """MobileNetV3 Small con una cabeza por salida.

    Esta clase replica la arquitectura usada en el notebook de entrenamiento:
    backbone MobileNetV3 Small + dropout + heads lineales por etiqueta.
    """

    def __init__(self, label_maps: Dict[str, Dict[str, int]], use_pretrained: bool = False):
        super().__init__()
        try:
            from torchvision import models
            from torchvision.models import MobileNet_V3_Small_Weights
        except Exception as exc:  # pragma: no cover - depende del ambiente local
            raise RuntimeError(
                "No se pudo importar torchvision. Instala una versión compatible con torch. "
                "Ejemplo: pip install torch torchvision"
            ) from exc

        weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(0.25)
        self.heads = nn.ModuleDict({
            col: nn.Linear(in_features, len(mapping))
            for col, mapping in label_maps.items()
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        return {col: head(features) for col, head in self.heads.items()}


def load_maps(idx_maps_path: str | Path, label_maps_path: str | Path | None = None) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, int]]]:
    idx_maps = json.loads(Path(idx_maps_path).read_text(encoding="utf-8"))
    if label_maps_path is not None and Path(label_maps_path).exists():
        label_maps = json.loads(Path(label_maps_path).read_text(encoding="utf-8"))
    else:
        label_maps = {
            group: {label: int(idx) for idx, label in values.items()}
            for group, values in idx_maps.items()
        }
    return idx_maps, label_maps


def load_model(
    model_path: str | Path,
    idx_maps_path: str | Path,
    label_maps_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> Tuple[MultiHeadMobileNet, Dict[str, Dict[str, str]], torch.device]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    idx_maps, label_maps = load_maps(idx_maps_path, label_maps_path)
    model = MultiHeadMobileNet(label_maps=label_maps, use_pretrained=False).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Compatibilidad con versiones viejas de PyTorch que no soportan weights_only.
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, idx_maps, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    try:
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "No se pudo importar torchvision.transforms. Instala torch/torchvision compatibles."
        ) from exc

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def _topk_for_output(logits: torch.Tensor, idx_map: Dict[str, str], k: int = 3) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=1)[0]
    k = min(k, probs.numel())
    values, indices = torch.topk(probs, k=k)
    return [
        {"label": idx_map[str(int(idx))], "confidence": float(val)}
        for val, idx in zip(values, indices)
    ]


def predict_image(
    image: Image.Image,
    model: MultiHeadMobileNet,
    idx_maps: Dict[str, Dict[str, str]],
    device: torch.device,
    top_k: int = 3,
) -> Dict[str, Any]:
    x = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(x)

    insect_topk = _topk_for_output(outputs["insect_label"], idx_maps["insect_label"], k=top_k)
    predicted_traits: Dict[str, Dict[str, Any]] = {}
    all_trait_probabilities: Dict[str, Dict[str, float]] = {}

    for group in TRAIT_COLUMNS:
        logits = outputs[group]
        probs = torch.softmax(logits, dim=1)[0]
        best_idx = int(torch.argmax(probs).item())
        predicted_traits[group] = {
            "label": idx_maps[group][str(best_idx)],
            "confidence": float(probs[best_idx].item()),
        }
        all_trait_probabilities[group] = {
            idx_maps[group][str(i)]: float(probs[i].item())
            for i in range(probs.numel())
        }

    return {
        "predicted_insect": insect_topk[0],
        "top_k_insects": insect_topk,
        "predicted_traits": predicted_traits,
        "trait_probabilities": all_trait_probabilities,
    }
