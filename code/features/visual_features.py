"""Image features: ResNet-50 embedding plus interpretable low-level statistics.

Only rows whose image belongs to this review are passed in. Shared web photos
of human-written fakes are excluded upstream, because the photo says nothing
about that review.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, project_path  # noqa: E402

STAT_COLUMNS = [
    "brightness", "saturation", "contrast", "lap_var", "edge_density",
    "colorfulness", "gray_entropy", "fft_high_ratio",
    "saliency_offset", "third_rule_score", "border_energy_ratio", "lr_asymmetry",
]


def _composition(gray: np.ndarray, energy: np.ndarray) -> dict:
    """Composition statistics on the gradient-energy map.

    saliency_offset        distance of the energy centroid from image centre,
                           0 means the subject sits in the middle.
    third_rule_score       largest energy share among the four thirds windows;
                           high means the subject sits near a thirds crossing.
    border_energy_ratio    energy in the outer band over total energy.
    lr_asymmetry           mean absolute difference against the mirror image.
    """
    height, width = gray.shape
    total = float(energy.sum())
    if not total:
        return {
            "saliency_offset": float("nan"),
            "third_rule_score": float("nan"),
            "border_energy_ratio": float("nan"),
            "lr_asymmetry": float("nan"),
        }
    yy, xx = np.mgrid[0:height, 0:width]
    center_y = float((energy * yy).sum() / total)
    center_x = float((energy * xx).sum() / total)
    diagonal = np.hypot(height / 2.0, width / 2.0)
    offset = float(np.hypot(center_y - height / 2.0, center_x - width / 2.0) / diagonal)

    window = max(4, min(height, width) // 8)
    score = 0.0
    for row in (height / 3.0, 2 * height / 3.0):
        for column in (width / 3.0, 2 * width / 3.0):
            top = int(min(max(row - window / 2, 0), height - window))
            left = int(min(max(column - window / 2, 0), width - window))
            score = max(score, float(energy[top : top + window, left : left + window].sum() / total))

    band = max(2, int(round(min(height, width) * 0.2)))
    border = energy.copy()
    border[band : height - band, band : width - band] = 0.0
    return {
        "saliency_offset": offset,
        "third_rule_score": score,
        "border_energy_ratio": float(border.sum() / total),
        "lr_asymmetry": float(np.abs(gray - gray[:, ::-1]).mean()),
    }


def image_stats(path: Path) -> dict:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )
    histogram, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    probability = histogram / max(1, histogram.sum())
    probability = probability[probability > 0]
    gray_entropy = float(-(probability * np.log2(probability)).sum() / np.log2(32))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    height, width = gray.shape
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt((yy - height / 2) ** 2 + (xx - width / 2) ** 2)
    limit = 0.25 * radius.max()
    total = spectrum.sum()
    high = spectrum[radius > limit].sum()
    return {
        "brightness": float(hsv[:, :, 2].mean()),
        "saturation": float(hsv[:, :, 1].mean()),
        "contrast": float(hsv[:, :, 2].std()),
        "lap_var": float(lap.var()),
        "edge_density": float((magnitude > 0.08).mean()),
        "colorfulness": colorfulness,
        "gray_entropy": gray_entropy,
        "fft_high_ratio": float(high / total) if total else float("nan"),
        **_composition(gray, magnitude ** 2),
    }


def load_resnet(device: torch.device):
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms()


def visual_features(frame: pd.DataFrame, device: torch.device, progress: int = 500):
    rows = []
    paths = []
    for record in frame.itertuples(index=False):
        path = project_path(record.image_file)
        try:
            rows.append(image_stats(path))
            paths.append(path)
        except Exception as exc:  # unreadable image: keep the row, record NaN
            print(f"image failed {record.id}: {exc}", flush=True)
            rows.append({name: float("nan") for name in STAT_COLUMNS})
            paths.append(None)
    stats = pd.DataFrame(rows, index=frame.index)[STAT_COLUMNS]

    model, transform = load_resnet(device)
    embed = np.zeros((len(frame), 2048), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(frame), 32):
            batch = []
            for path in paths[start : start + 32]:
                if path is None:
                    batch.append(torch.zeros(3, 224, 224))
                else:
                    with Image.open(path) as handle:
                        batch.append(transform(handle.convert("RGB")))
            output = model(torch.stack(batch).to(device))
            embed[start : start + len(batch)] = output.cpu().numpy()
            if start % progress == 0:
                print(f"resnet {start}/{len(frame)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    norm = np.clip(np.linalg.norm(embed, axis=1, keepdims=True), 1e-8, None)
    return stats, (embed / norm).astype(np.float16)
