"""
시각화/설명용 샘플 데이터 추출 스크립트.

B1.1 (기준 공정 조건) 빌드에서 중간 레이어 1개를 기준으로
각 데이터 카테고리의 대표 샘플을 ORNL_Data_Open/sample_showcase/ 에 저장한다.
"""

from pathlib import Path
import json
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.collections as collections

PROJECT_ROOT = Path("/home/taehoon/3DP_TensileProp_Prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from Sources.hdf5_parser.ornl_data_loader import ORNLDataLoader, DSCNN_CLASSES

HDF5_PATH = PROJECT_ROOT / "ORNL_Data_Origin" / "2021-07-13 TCR Phase 1 Build 1.hdf5"
OUT_DIR = PROJECT_ROOT / "ORNL_Data_Open" / "sample_showcase"
PLOTS = OUT_DIR / "plots"
RAW = OUT_DIR / "raw"
PLOTS.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)


def save_img(fig, name):
    path = PLOTS / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.relative_to(PROJECT_ROOT)}")


def main():
    print(f"[1] 파일 로드: {HDF5_PATH.name}")
    with ORNLDataLoader(str(HDF5_PATH)) as loader:
        num_layers = loader.get_num_layers()
        # 시편 단면이 실제로 보이는 레이어를 자동 선택 (sample_ids unique가 최대가 되는 곳)
        probe_layers = np.linspace(50, num_layers - 50, 25, dtype=int)
        best_layer, best_unique = probe_layers[0], 0
        for L in probe_layers:
            sids = loader.get_sample_ids(int(L))
            u = len(np.unique(sids))
            if u > best_unique:
                best_unique, best_layer = u, int(L)
        target_layer = best_layer
        print(f"    총 레이어 수: {num_layers}, 대표 레이어: {target_layer} (sample_id unique={best_unique})")

        summary = {
            "build_id": loader.build_id,
            "build_name": loader.get_build_name(),
            "num_layers": int(num_layers),
            "num_samples": int(loader.num_samples),
            "sample_layer": int(target_layer),
        }

        # ---- 2) 카메라 이미지 (용융 직후, 분말 도포 직후) ----
        print("[2] 카메라 이미지 2종 추출")
        img_visible0 = loader.get_camera_image(target_layer, camera_id=0)  # 용융 직후
        img_visible1 = loader.get_camera_image(target_layer, camera_id=1)  # 분말 도포 직후
        np.save(RAW / "camera_visible_0_postmelt.npy", img_visible0)
        np.save(RAW / "camera_visible_1_postpowder.npy", img_visible1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_visible0, cmap="gray", interpolation="none")
        axes[0].set_title(f"visible/0 : post-melt (Layer {target_layer})\nshape={img_visible0.shape}")
        axes[0].axis("off")
        axes[1].imshow(img_visible1, cmap="gray", interpolation="none")
        axes[1].set_title(f"visible/1 : post-powder (Layer {target_layer})\nshape={img_visible1.shape}")
        axes[1].axis("off")
        save_img(fig, "01_camera_images.png")
        summary["camera_image_shape"] = list(img_visible0.shape)

        # ---- 3) Part / Sample ID 맵 ----
        print("[3] Part / Sample ID 맵 추출")
        part_ids = loader.get_part_ids(target_layer)
        sample_ids = loader.get_sample_ids(target_layer)
        np.save(RAW / "part_ids.npy", part_ids)
        np.save(RAW / "sample_ids.npy", sample_ids)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(part_ids, cmap="tab20", interpolation="none")
        axes[0].set_title(f"slices/part_ids (Layer {target_layer})\n"
                          f"unique IDs = {len(np.unique(part_ids))}")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(sample_ids, cmap="tab20", interpolation="none")
        axes[1].set_title(f"slices/sample_ids (Layer {target_layer})\n"
                          f"unique IDs = {len(np.unique(sample_ids))}")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        save_img(fig, "02_id_maps.png")

        # ---- 4) 세그멘테이션 대표 클래스 ----
        print("[4] 세그멘테이션 결과 (대표 클래스 4종)")
        # Powder(0), Printed(1), Recoater Streaking(3), Swelling(5)
        pick = [0, 1, 3, 5]
        fig, axes = plt.subplots(1, len(pick), figsize=(4 * len(pick), 4.2))
        for ax, cls_id in zip(axes, pick):
            seg = loader.get_segmentation_result(target_layer, cls_id)
            np.save(RAW / f"segmentation_class_{cls_id}.npy", seg)
            ax.imshow(seg, cmap="viridis", interpolation="none")
            ax.set_title(f"class {cls_id}: {DSCNN_CLASSES[cls_id]}")
            ax.axis("off")
        fig.suptitle(f"slices/segmentation_results/# (Layer {target_layer})", y=1.02)
        save_img(fig, "03_segmentation.png")

        # ---- 5) 스캔 경로 ----
        print("[5] 레이저 스캔 경로")
        try:
            scan = loader.get_scan_path(target_layer)
            np.save(RAW / "scan_path.npy", scan)
            x = scan[:, 0:2]
            y = scan[:, 2:4]
            t = scan[:, 4]
            colorizer = cm.ScalarMappable(norm=mcolors.Normalize(t.min(), t.max()), cmap="jet")
            segs = np.stack([x, y], axis=-1)
            line_collection = collections.LineCollection(segs, colors=colorizer.to_rgba(t), linewidths=0.4)
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.add_collection(line_collection)
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_aspect("equal")
            ax.set_title(f"scans/{target_layer}  |  vectors={scan.shape[0]}  (color=time)")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            save_img(fig, "04_scan_path.png")
            summary["scan_num_vectors"] = int(scan.shape[0])
        except Exception as e:
            print(f"    스캔 경로 없음: {e}")
            summary["scan_num_vectors"] = None

        # ---- 6) 시간 센서 데이터 ----
        print("[6] Temporal 센서 데이터")
        sensors = [
            "top_flow_rate", "bottom_flow_rate",
            "gas_loop_oxygen", "module_oxygen",
            "build_time", "layer_times",
            "top_chamber_temperature", "build_plate_temperature",
        ]
        temporal_df = {}
        fig, axes = plt.subplots(4, 2, figsize=(13, 10))
        for ax, key in zip(axes.ravel(), sensors):
            arr = loader.get_temporal_data(key)
            temporal_df[key] = arr
            ax.plot(arr, lw=0.7)
            ax.set_title(key)
            ax.set_xlabel("layer")
            ax.grid(alpha=0.3)
        fig.suptitle(f"temporal/* (len={len(arr)})", y=1.00)
        fig.tight_layout()
        save_img(fig, "05_temporal_sensors.png")
        pd.DataFrame(temporal_df).to_csv(RAW / "temporal_sensors.csv", index_label="layer")

        # ---- 7) 공정 파라미터 ----
        print("[7] 공정 파라미터")
        pp = loader.get_process_parameters()
        pp_df = pd.DataFrame({k: v for k, v in pp.items() if isinstance(v, np.ndarray) and v.ndim == 1})
        pp_df.to_csv(RAW / "process_parameters.csv", index_label="part_id")
        summary["num_parts"] = int(len(pp_df))

        # ---- 8) 인장 시험 결과 ----
        print("[8] 인장 시험 결과 분포")
        tensile = loader.get_tensile_properties()
        tens_df = pd.DataFrame({k: v for k, v in tensile.items()
                                if isinstance(v, np.ndarray) and v.ndim == 1})
        tens_df.to_csv(RAW / "tensile_results_samples.csv", index_label="sample_id")

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        labels = {
            "yield_strength": "YS (MPa)",
            "ultimate_tensile_strength": "UTS (MPa)",
            "uniform_elongation": "UE (%)",
            "total_elongation": "TE (%)",
        }
        for ax, (key, lbl) in zip(axes, labels.items()):
            vals = tens_df[key].values
            vals = vals[np.isfinite(vals) & (vals > 0)]
            ax.hist(vals, bins=40, edgecolor="k", alpha=0.8)
            ax.set_title(f"{lbl}  n={len(vals)}")
            ax.grid(alpha=0.3)
        fig.suptitle("samples/test_results/* distribution", y=1.03)
        fig.tight_layout()
        save_img(fig, "06_tensile_histograms.png")

        # ---- 9) 레퍼런스 이미지 1장 ----
        print("[9] 레퍼런스 이미지 (썸네일)")
        try:
            with h5py.File(HDF5_PATH, "r") as f:
                if "reference_images/thumbnail" in f:
                    thumb = f["reference_images/thumbnail"][...]
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(thumb, interpolation="none")
                    ax.set_title("reference_images/thumbnail")
                    ax.axis("off")
                    save_img(fig, "07_reference_thumbnail.png")
        except Exception as e:
            print(f"    레퍼런스 이미지 없음: {e}")

        # ---- 10) 요약 JSON ----
        (OUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n완료 -> {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
