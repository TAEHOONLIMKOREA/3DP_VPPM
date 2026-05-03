"""DSCNN_Dataset (Peregrine semantic-segmentation GT) 레이어 시각화.

지원 레이아웃:
    1) v2021-03 LPBF — flat 구조
        ORNL_Data/DSCNN_Dataset/Peregrine Dataset v2021-03/Laser Powder Bed Fusion/
            ├── annotations/<layer>.npy
            └── data/visible/{0,1}/<layer>.tif
       (info.txt 없음 → 숫자 라벨만 표시)

    2) v2022-10.1 — <machine>/<material>/training/
        EOS_M290/17-4_PH_Stainless_Steel/training/
            ├── info.txt          (클래스 ID ↔ 이름)
            ├── annotations/<layer>.npy
            ├── data/visible/{0,1}/<layer>.tif
            ├── parts/<layer>.png
            └── thumbnails/<layer>.tif

출력: visible/0, visible/1, annotation(컬러맵), overlay 의 1x4 PNG.

사용:
    python -m Sources.vppm.tools.visualize_dscnn \
        --root "ORNL_Data/DSCNN_Dataset/Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/EOS_M290/17-4_PH_Stainless_Steel/training" \
        --layer 0000004 \
        --out Sources/pipeline_outputs/figures/dscnn_vis/17-4PH_0000004.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import ListedColormap
from PIL import Image

# 13가지 + unlabeled 까지 커버하는 색 팔레트 (-1 unlabeled 는 검정으로 별도 처리)
_PALETTE = np.array([
    [0.85, 0.85, 0.85],  # 0  Powder       (light gray)
    [0.30, 0.50, 0.85],  # 1  Printed      (blue)
    [0.95, 0.55, 0.10],  # 2
    [0.20, 0.70, 0.30],  # 3
    [0.85, 0.15, 0.15],  # 4
    [0.55, 0.30, 0.75],  # 5
    [0.50, 0.35, 0.20],  # 6
    [0.95, 0.40, 0.65],  # 7
    [0.40, 0.40, 0.40],  # 8
    [0.75, 0.75, 0.10],  # 9
    [0.10, 0.70, 0.70],  # 10
    [0.95, 0.75, 0.30],  # 11
    [0.30, 0.30, 0.85],  # 12
    [0.60, 0.10, 0.10],  # 13
    [0.10, 0.60, 0.10],  # 14
])


def parse_info(info_path: Path) -> dict[int, str]:
    """info.txt 의 'class ids and names:' 블록에서 {id: name} 사전 파싱."""
    if not info_path.is_file():
        return {}
    text = info_path.read_text()
    mapping: dict[int, str] = {}
    in_block = False
    for line in text.splitlines():
        if "class ids and names" in line.lower():
            in_block = True
            continue
        if in_block:
            m = re.match(r"\s*(-?\d+)\s*=\s*(.+?)\s*$", line)
            if not m:
                if line.strip() and not line.startswith(" "):
                    break
                continue
            mapping[int(m.group(1))] = m.group(2)
    return mapping


def colorize(annotation: np.ndarray, max_id: int) -> np.ndarray:
    """annotation (H,W) int → RGB (H,W,3). -1 은 검정."""
    h, w = annotation.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    n_colors = len(_PALETTE)
    for cid in range(0, max_id + 1):
        rgb[annotation == cid] = _PALETTE[cid % n_colors]
    rgb[annotation == -1] = 0.0
    return rgb


def visualize_layer(root: Path, layer_id: str, out_path: Path) -> None:
    ann_path = root / "annotations" / f"{layer_id}.npy"
    v0_path = root / "data" / "visible" / "0" / f"{layer_id}.tif"
    v1_path = root / "data" / "visible" / "1" / f"{layer_id}.tif"
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)
    if not v0_path.is_file():
        raise FileNotFoundError(v0_path)
    if not v1_path.is_file():
        raise FileNotFoundError(v1_path)

    ann = np.load(ann_path)
    v0 = np.array(Image.open(v0_path))
    v1 = np.array(Image.open(v1_path))

    info_path = root / "info.txt"
    if not info_path.is_file():
        info_path = root.parent / "info.txt"
    class_names = parse_info(info_path)
    present = sorted(int(c) for c in np.unique(ann))
    max_id = max(present) if present else 0

    ann_rgb = colorize(ann, max_id=max(max_id, max(class_names) if class_names else 0))
    overlay = np.stack([v0] * 3, axis=-1).astype(np.float32) / 255.0
    mask = ann >= 0
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * ann_rgb[mask]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(v0, cmap="gray")
    axes[0].set_title(f"visible/0 ({v0.shape[0]}x{v0.shape[1]})")
    axes[1].imshow(v1, cmap="gray")
    axes[1].set_title(f"visible/1 ({v1.shape[0]}x{v1.shape[1]})")
    axes[2].imshow(ann_rgb)
    axes[2].set_title(f"annotation (classes: {present})")
    axes[3].imshow(overlay)
    axes[3].set_title("visible/0 + annotation overlay")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = []
    for cid in present:
        if cid == -1:
            color = (0.0, 0.0, 0.0)
            label = "-1 unlabeled"
        else:
            color = tuple(_PALETTE[cid % len(_PALETTE)])
            name = class_names.get(cid, f"class_{cid}")
            label = f"{cid} {name}"
        legend_handles.append(patches.Patch(color=color, label=label))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 6),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(f"{root.name} / {layer_id}", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="annotations/, data/visible/{0,1}/ 가 있는 디렉토리. v2021-03 LPBF 또는 v2022-10.1 의 .../<material>/training/",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="레이어 ID (예: '0000004'). 미지정 시 첫 번째 라벨 레이어.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="저장 경로. 미지정 시 Sources/pipeline_outputs/figures/dscnn_vis/<root_basename>_<layer>.png",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not (root / "annotations").is_dir():
        raise SystemExit(f"annotations/ not found under {root}")

    if args.layer is None:
        first = sorted(p.stem for p in (root / "annotations").glob("*.npy"))
        if not first:
            raise SystemExit(f"no .npy under {root / 'annotations'}")
        layer_id = first[0]
    else:
        layer_id = args.layer

    out = args.out
    if out is None:
        repo = Path(__file__).resolve().parents[3]
        out = repo / "Sources/pipeline_outputs/figures/dscnn_vis" / f"{root.name}_{layer_id}.png"

    visualize_layer(root, layer_id, out)


if __name__ == "__main__":
    main()
