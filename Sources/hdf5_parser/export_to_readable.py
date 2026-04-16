"""
ORNL L-PBF Dataset Export Script
================================
HDF5 데이터를 사람이 읽을 수 있는 형식으로 저장

출력 형식:
- CSV: 인장 시험 결과, 공정 파라미터, 시간적 데이터
- PNG: 카메라 이미지, 세그멘테이션 결과, 스캔 경로
- JSON: 빌드 메타데이터

실행 방법:
    python export_to_readable.py
    python export_to_readable.py --build B1.1 --layers 50,100,150
    python export_to_readable.py --all-builds --sample-layers 10
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.collections as collections

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))
from ornl_data_loader import (
    ORNLDataLoader,
    list_hdf5_files,
    BUILD_INFO,
    DSCNN_CLASSES
)


class ORNLDataExporter:
    """HDF5 데이터를 사람이 읽을 수 있는 형식으로 내보내기"""

    def __init__(self, loader: ORNLDataLoader, output_dir: Path):
        self.loader = loader
        self.output_dir = output_dir
        self.build_id = loader.build_id

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, layers: list = None, sample_layers: int = 5):
        """모든 데이터 내보내기"""
        print(f"\n{'='*60}")
        print(f"Exporting {self.build_id} to {self.output_dir}")
        print(f"{'='*60}")

        # 레이어 선택
        if layers is None:
            num_layers = self.loader.get_num_layers()
            if num_layers > 0:
                step = max(1, num_layers // sample_layers)
                layers = list(range(0, num_layers, step))[:sample_layers]
            else:
                layers = [50]  # 기본값

        print(f"Selected layers: {layers}")

        # 1. 메타데이터 (JSON)
        self.export_metadata()

        # 2. 인장 시험 결과 (CSV)
        self.export_tensile_results()

        # 3. 공정 파라미터 (CSV)
        self.export_process_parameters()

        # 4. 시간적 데이터 (CSV)
        self.export_temporal_data()

        # 5. 카메라 이미지 (PNG)
        self.export_camera_images(layers)

        # 6. 세그멘테이션 결과 (PNG)
        self.export_segmentation_images(layers)

        # 7. 스캔 경로 (PNG + CSV)
        self.export_scan_paths(layers)

        # 8. 요약 리포트 생성
        self.export_summary_report()

        print(f"\nExport completed: {self.output_dir}")

    def export_metadata(self):
        """빌드 메타데이터를 JSON으로 저장"""
        print("\n[1/8] Exporting metadata...")

        metadata = {}

        # 빌드 속성
        for key in self.loader.file.attrs:
            val = self.loader.file.attrs[key]
            # numpy 타입을 Python 기본 타입으로 변환
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, (np.integer, np.floating)):
                val = val.item()
            elif isinstance(val, bytes):
                val = val.decode('utf-8', errors='ignore')
            metadata[key] = val

        # 추가 정보
        metadata['_export_info'] = {
            'build_id': self.build_id,
            'num_layers': self.loader.get_num_layers(),
            'num_samples': self.loader.num_samples,
            'export_date': datetime.now().isoformat(),
        }

        # JSON 저장
        json_path = self.output_dir / "metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        print(f"  Saved: {json_path}")

    def export_tensile_results(self):
        """인장 시험 결과를 CSV로 저장"""
        print("\n[2/8] Exporting tensile test results...")

        try:
            results = self.loader.get_test_results('samples')
            if not results:
                print("  No tensile results found")
                return

            # DataFrame 생성
            df = pd.DataFrame(results)
            df.index.name = 'sample_id'

            # 유효한 데이터만 필터링
            valid_mask = ~df.isna().all(axis=1)
            df_valid = df[valid_mask]

            # CSV 저장
            csv_path = self.output_dir / "tensile_results.csv"
            df.to_csv(csv_path)
            print(f"  Saved: {csv_path} ({len(df_valid)} valid samples)")

            # 통계 요약도 저장
            stats_path = self.output_dir / "tensile_results_stats.csv"
            df_valid.describe().to_csv(stats_path)
            print(f"  Saved: {stats_path}")

        except Exception as e:
            print(f"  Error: {e}")

    def export_process_parameters(self):
        """공정 파라미터를 CSV로 저장"""
        print("\n[3/8] Exporting process parameters...")

        try:
            params = self.loader.get_process_parameters()
            if not params:
                print("  No process parameters found")
                return

            # DataFrame 생성
            df_data = {}
            for key, val in params.items():
                if val is not None:
                    # bytes를 문자열로 변환
                    if len(val) > 0 and isinstance(val[0], (bytes, np.bytes_)):
                        val = [v.decode('utf-8', errors='ignore') if isinstance(v, bytes) else str(v) for v in val]
                    df_data[key] = val

            df = pd.DataFrame(df_data)
            df.index.name = 'part_id'

            # CSV 저장
            csv_path = self.output_dir / "process_parameters.csv"
            df.to_csv(csv_path)
            print(f"  Saved: {csv_path} ({len(df)} parts)")

        except Exception as e:
            print(f"  Error: {e}")

    def export_temporal_data(self):
        """시간적 데이터를 CSV로 저장"""
        print("\n[4/8] Exporting temporal data...")

        try:
            temporal_keys = self.loader.list_temporal_keys()
            if not temporal_keys:
                print("  No temporal data found")
                return

            # 모든 시간적 데이터 수집
            df_data = {}
            for key in temporal_keys:
                try:
                    data = self.loader.get_temporal_data(key)
                    df_data[key] = data
                except Exception:
                    continue

            # DataFrame 생성
            df = pd.DataFrame(df_data)
            df.index.name = 'layer'

            # CSV 저장
            csv_path = self.output_dir / "temporal_data.csv"
            df.to_csv(csv_path)
            print(f"  Saved: {csv_path} ({len(df)} layers, {len(df.columns)} sensors)")

        except Exception as e:
            print(f"  Error: {e}")

    def export_camera_images(self, layers: list):
        """카메라 이미지를 PNG로 저장"""
        print("\n[5/8] Exporting camera images...")

        img_dir = self.output_dir / "camera_images"
        img_dir.mkdir(exist_ok=True)

        for layer in layers:
            for camera_id in [0, 1]:
                try:
                    img = self.loader.get_camera_image(layer, camera_id)

                    # 이미지 정규화
                    img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)

                    # PNG 저장
                    camera_name = "post_melt" if camera_id == 0 else "post_powder"
                    img_path = img_dir / f"layer_{layer:05d}_{camera_name}.png"

                    plt.figure(figsize=(10, 10))
                    plt.imshow(img_normalized, cmap='gray', interpolation='none')
                    plt.title(f'Layer {layer} - {camera_name}')
                    plt.colorbar(label='Intensity')
                    plt.savefig(img_path, dpi=150, bbox_inches='tight')
                    plt.close()

                except Exception as e:
                    continue

        print(f"  Saved to: {img_dir}")

    def export_segmentation_images(self, layers: list):
        """DSCNN 세그멘테이션 결과를 PNG로 저장"""
        print("\n[6/8] Exporting segmentation results...")

        seg_dir = self.output_dir / "segmentation"
        seg_dir.mkdir(exist_ok=True)

        for layer in layers:
            try:
                # 모든 클래스의 세그멘테이션 결과를 하나의 이미지로
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                axes = axes.flatten()

                for class_id in range(12):
                    try:
                        seg = self.loader.get_segmentation_result(layer, class_id)
                        ax = axes[class_id]
                        im = ax.imshow(seg, cmap='jet', interpolation='none')
                        ax.set_title(f'{class_id}: {DSCNN_CLASSES.get(class_id, "Unknown")}',
                                     fontsize=8)
                        ax.axis('off')
                    except Exception:
                        axes[class_id].set_visible(False)

                plt.suptitle(f'Layer {layer} - DSCNN Segmentation Results')
                plt.tight_layout()

                img_path = seg_dir / f"layer_{layer:05d}_segmentation.png"
                plt.savefig(img_path, dpi=150, bbox_inches='tight')
                plt.close()

            except Exception as e:
                continue

        print(f"  Saved to: {seg_dir}")

    def export_scan_paths(self, layers: list):
        """스캔 경로를 PNG와 CSV로 저장"""
        print("\n[7/8] Exporting scan paths...")

        scan_dir = self.output_dir / "scan_paths"
        scan_dir.mkdir(exist_ok=True)

        scan_layers = self.loader.list_scan_layers()

        for layer in layers:
            if layer not in scan_layers:
                continue

            try:
                x, y, t = self.loader.get_scan_path_xy(layer)

                # CSV 저장
                scan_df = pd.DataFrame({
                    'x_start': x[:, 0],
                    'x_end': x[:, 1],
                    'y_start': y[:, 0],
                    'y_end': y[:, 1],
                    'relative_time': t
                })
                csv_path = scan_dir / f"layer_{layer:05d}_scanpath.csv"
                scan_df.to_csv(csv_path, index=False)

                # PNG 저장
                fig, ax = plt.subplots(figsize=(10, 10))

                colorizer = cm.ScalarMappable(
                    norm=mcolors.Normalize(np.min(t), np.max(t)),
                    cmap='jet'
                )
                line_collection = collections.LineCollection(
                    np.stack([x, y], axis=-1),
                    colors=colorizer.to_rgba(t),
                    linewidths=0.5
                )

                ax.add_collection(line_collection)
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())
                ax.set_aspect('equal')
                ax.set_title(f'Layer {layer} - Scan Path ({len(t)} vectors)')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                plt.colorbar(colorizer, ax=ax, label='Relative Time')

                img_path = scan_dir / f"layer_{layer:05d}_scanpath.png"
                plt.savefig(img_path, dpi=150, bbox_inches='tight')
                plt.close()

            except Exception as e:
                continue

        print(f"  Saved to: {scan_dir}")

    def export_summary_report(self):
        """요약 리포트 생성"""
        print("\n[8/8] Generating summary report...")

        summary = self.loader.get_summary()
        results = self.loader.get_test_results('samples')

        report_lines = [
            "=" * 60,
            f"ORNL L-PBF Dataset Export Summary",
            "=" * 60,
            "",
            f"Build Name: {summary['build_name']}",
            f"Build ID: {summary['build_id']}",
            f"Description: {summary['description']}",
            f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=" * 60,
            "Dataset Statistics",
            "=" * 60,
            "",
            f"Total Layers: {summary['num_layers']}",
            f"Expected Samples: {summary['num_samples_expected']}",
            f"Valid Samples: {summary['num_valid_samples']}",
            "",
        ]

        # 인장 특성 통계
        if results:
            report_lines.extend([
                "Tensile Properties Summary:",
                "-" * 40,
            ])
            for key, data in results.items():
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    report_lines.append(
                        f"  {key}:"
                    )
                    report_lines.append(
                        f"    Mean: {np.mean(valid_data):.2f}, Std: {np.std(valid_data):.2f}"
                    )
                    report_lines.append(
                        f"    Min: {np.min(valid_data):.2f}, Max: {np.max(valid_data):.2f}"
                    )
            report_lines.append("")

        # 파일 목록
        report_lines.extend([
            "=" * 60,
            "Exported Files",
            "=" * 60,
            "",
        ])

        for item in sorted(self.output_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(self.output_dir)
                size_kb = item.stat().st_size / 1024
                report_lines.append(f"  {rel_path} ({size_kb:.1f} KB)")

        report_lines.extend([
            "",
            "=" * 60,
            "DSCNN Anomaly Classes Reference",
            "=" * 60,
            "",
        ])
        for class_id, class_name in DSCNN_CLASSES.items():
            report_lines.append(f"  {class_id}: {class_name}")

        # 리포트 저장
        report_path = self.output_dir / "SUMMARY_REPORT.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export ORNL L-PBF HDF5 data to human-readable formats'
    )
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to ORNL_Data_Origin directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for exported files')
    parser.add_argument('--build', type=str, default=None,
                        help='Build ID to export (e.g., B1.1). If not specified, exports first available.')
    parser.add_argument('--all-builds', action='store_true',
                        help='Export all available builds')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layer numbers (e.g., 50,100,150)')
    parser.add_argument('--sample-layers', type=int, default=5,
                        help='Number of evenly-spaced layers to sample (default: 5)')
    args = parser.parse_args()

    # 데이터 디렉토리 설정
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent.parent / "ORNL_Data_Origin"

    # 출력 디렉토리 설정
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(__file__).parent.parent.parent / "ORNL_Data_Open"

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_base}")

    # HDF5 파일 목록
    hdf5_files = list_hdf5_files(data_dir)
    if not hdf5_files:
        print(f"\nNo HDF5 files found in {data_dir}")
        return

    print(f"\nAvailable builds: {len(hdf5_files)}")
    for f in hdf5_files:
        info = BUILD_INFO.get(f.name, {})
        print(f"  - {info.get('id', 'Unknown')}: {f.name}")

    # 레이어 파싱
    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    # 빌드 선택 및 내보내기
    if args.all_builds:
        builds_to_export = hdf5_files
    elif args.build:
        # 특정 빌드 찾기
        builds_to_export = []
        for f in hdf5_files:
            info = BUILD_INFO.get(f.name, {})
            if info.get('id') == args.build:
                builds_to_export.append(f)
                break
        if not builds_to_export:
            print(f"\nBuild '{args.build}' not found")
            return
    else:
        builds_to_export = [hdf5_files[0]]

    # 내보내기 실행
    for hdf5_path in builds_to_export:
        info = BUILD_INFO.get(hdf5_path.name, {})
        build_id = info.get('id', hdf5_path.stem)
        output_dir = output_base / build_id

        print(f"\n{'#'*60}")
        print(f"Processing: {build_id}")
        print(f"{'#'*60}")

        with ORNLDataLoader(hdf5_path) as loader:
            exporter = ORNLDataExporter(loader, output_dir)
            exporter.export_all(layers=layers, sample_layers=args.sample_layers)

    print(f"\n{'='*60}")
    print("All exports completed!")
    print(f"Output location: {output_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
