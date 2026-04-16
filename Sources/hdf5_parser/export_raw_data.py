"""
ORNL L-PBF Dataset Raw Data Export Script
=========================================
HDF5 데이터를 원본 그대로 추출

출력 형식:
- NPY: 이미지, 세그멘테이션 배열 (numpy 원본)
- CSV: 인장 시험 결과, 공정 파라미터, 시간적 데이터, 스캔 경로
- JSON: 빌드 메타데이터

실행 방법:
    python export_raw_data.py --build B1.2
    python export_raw_data.py --build B1.2 --layers 50,100,150
    python export_raw_data.py --build B1.2 --all-layers
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))
from ornl_data_loader import (
    ORNLDataLoader,
    list_hdf5_files,
    BUILD_INFO,
    DSCNN_CLASSES
)


class ORNLRawDataExporter:
    """HDF5 데이터를 원본 그대로 내보내기"""

    def __init__(self, loader: ORNLDataLoader, output_dir: Path):
        self.loader = loader
        self.output_dir = output_dir
        self.build_id = loader.build_id

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, layers: list = None, all_layers: bool = False):
        """모든 데이터 내보내기"""
        print(f"\n{'='*60}")
        print(f"Exporting RAW data: {self.build_id}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}")

        num_layers = self.loader.get_num_layers()
        print(f"Total layers in build: {num_layers}")

        # 레이어 선택
        if all_layers:
            layers = list(range(num_layers))
            print(f"Exporting ALL {num_layers} layers")
        elif layers is None:
            # 기본: 10개 샘플 레이어
            step = max(1, num_layers // 10)
            layers = list(range(0, num_layers, step))
            print(f"Exporting sample layers: {layers}")
        else:
            print(f"Exporting specified layers: {layers}")

        # 1. 메타데이터 (JSON)
        self.export_metadata()

        # 2. 인장 시험 결과 (CSV) - 전체
        self.export_tensile_results()

        # 3. 공정 파라미터 (CSV) - 전체
        self.export_process_parameters()

        # 4. 시간적 데이터 (CSV) - 전체 레이어
        self.export_temporal_data()

        # 5. 카메라 이미지 (NPY + TIFF)
        self.export_camera_images_raw(layers)

        # 6. 세그멘테이션 결과 (NPY)
        self.export_segmentation_raw(layers)

        # 7. 파트/샘플 ID 맵 (NPY)
        self.export_id_maps_raw(layers)

        # 8. 스캔 경로 (CSV)
        self.export_scan_paths_raw(layers)

        # 9. 요약 정보
        self.export_summary()

        print(f"\n{'='*60}")
        print(f"Export completed: {self.output_dir}")
        print(f"{'='*60}")

    def export_metadata(self):
        """빌드 메타데이터를 JSON으로 저장"""
        print("\n[1/9] Exporting metadata...")

        metadata = {}

        # 빌드 속성 - 전체 추출
        for key in self.loader.file.attrs:
            val = self.loader.file.attrs[key]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, (np.integer, np.floating)):
                val = val.item()
            elif isinstance(val, bytes):
                val = val.decode('utf-8', errors='ignore')
            metadata[key] = val

        metadata['_export_info'] = {
            'build_id': self.build_id,
            'num_layers': self.loader.get_num_layers(),
            'num_samples': self.loader.num_samples,
            'export_date': datetime.now().isoformat(),
            'export_type': 'raw'
        }

        json_path = self.output_dir / "metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        print(f"  Saved: {json_path}")

    def export_tensile_results(self):
        """인장 시험 결과를 CSV로 저장 (전체 데이터)"""
        print("\n[2/9] Exporting tensile test results (all samples)...")

        try:
            # samples 데이터
            results_samples = self.loader.get_test_results('samples')
            if results_samples:
                df = pd.DataFrame(results_samples)
                df.index.name = 'sample_id'
                csv_path = self.output_dir / "tensile_results_samples.csv"
                df.to_csv(csv_path)
                valid_count = df.notna().any(axis=1).sum()
                print(f"  Saved: {csv_path} ({valid_count} valid / {len(df)} total)")

            # parts 데이터
            results_parts = self.loader.get_test_results('parts')
            if results_parts:
                df = pd.DataFrame(results_parts)
                df.index.name = 'part_id'
                csv_path = self.output_dir / "tensile_results_parts.csv"
                df.to_csv(csv_path)
                valid_count = df.notna().any(axis=1).sum()
                print(f"  Saved: {csv_path} ({valid_count} valid / {len(df)} total)")

        except Exception as e:
            print(f"  Error: {e}")

    def export_process_parameters(self):
        """공정 파라미터를 CSV로 저장 (전체 데이터)"""
        print("\n[3/9] Exporting process parameters (all parts)...")

        try:
            params = self.loader.get_process_parameters()
            if not params:
                print("  No process parameters found")
                return

            df_data = {}
            for key, val in params.items():
                if val is not None:
                    if len(val) > 0 and isinstance(val[0], (bytes, np.bytes_)):
                        val = [v.decode('utf-8', errors='ignore') if isinstance(v, bytes) else str(v) for v in val]
                    df_data[key] = val

            df = pd.DataFrame(df_data)
            df.index.name = 'part_id'

            csv_path = self.output_dir / "process_parameters.csv"
            df.to_csv(csv_path)
            print(f"  Saved: {csv_path} ({len(df)} parts)")

        except Exception as e:
            print(f"  Error: {e}")

    def export_temporal_data(self):
        """시간적 데이터를 CSV로 저장 (전체 레이어)"""
        print("\n[4/9] Exporting temporal data (all layers)...")

        try:
            temporal_keys = self.loader.list_temporal_keys()
            if not temporal_keys:
                print("  No temporal data found")
                return

            # 각 센서 데이터를 개별 CSV로 저장 (길이가 다를 수 있음)
            temporal_dir = self.output_dir / "temporal_data"
            temporal_dir.mkdir(exist_ok=True)

            max_len = 0
            saved_count = 0

            for key in temporal_keys:
                try:
                    data = self.loader.get_temporal_data(key)
                    max_len = max(max_len, len(data))

                    # 개별 CSV 저장
                    df = pd.DataFrame({key: data})
                    df.index.name = 'layer'
                    csv_path = temporal_dir / f"{key}.csv"
                    df.to_csv(csv_path)
                    saved_count += 1

                except Exception as e:
                    continue

            # 동일 길이 데이터만 합쳐서 저장 시도
            try:
                df_data = {}
                for key in temporal_keys:
                    try:
                        data = self.loader.get_temporal_data(key)
                        if len(data) == max_len:
                            df_data[key] = data
                    except:
                        continue

                if df_data:
                    df = pd.DataFrame(df_data)
                    df.index.name = 'layer'
                    csv_path = self.output_dir / "temporal_data_combined.csv"
                    df.to_csv(csv_path)
                    print(f"  Combined CSV: {csv_path} ({len(df)} layers, {len(df.columns)} sensors)")

            except Exception:
                pass

            print(f"  Saved to: {temporal_dir}")
            print(f"  Individual CSVs: {saved_count} sensors")

        except Exception as e:
            print(f"  Error: {e}")

    def export_camera_images_raw(self, layers: list):
        """카메라 이미지를 원본 그대로 저장 (NPY + TIFF)"""
        print(f"\n[5/9] Exporting camera images RAW ({len(layers)} layers)...")

        img_dir = self.output_dir / "camera_images"
        img_dir.mkdir(exist_ok=True)

        # 이미지 정보 기록용
        image_info = []

        for i, layer in enumerate(layers):
            if i % 100 == 0:
                print(f"  Processing layer {layer} ({i+1}/{len(layers)})...")

            for camera_id in [0, 1]:
                try:
                    img = self.loader.get_camera_image(layer, camera_id)
                    camera_name = "post_melt" if camera_id == 0 else "post_powder"

                    # NPY 저장 (완전한 원본)
                    npy_path = img_dir / f"layer_{layer:05d}_{camera_name}.npy"
                    np.save(npy_path, img)

                    # TIFF 저장 (무손실 이미지 포맷)
                    tiff_path = img_dir / f"layer_{layer:05d}_{camera_name}.tiff"
                    if img.dtype == np.uint16:
                        Image.fromarray(img).save(tiff_path)
                    elif img.dtype == np.uint8:
                        Image.fromarray(img).save(tiff_path)
                    else:
                        # float 등 다른 타입은 16bit로 변환
                        img_16bit = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 65535).astype(np.uint16)
                        Image.fromarray(img_16bit).save(tiff_path)

                    image_info.append({
                        'layer': layer,
                        'camera_id': camera_id,
                        'camera_name': camera_name,
                        'dtype': str(img.dtype),
                        'shape': img.shape,
                        'min': float(img.min()),
                        'max': float(img.max()),
                        'mean': float(img.mean())
                    })

                except Exception as e:
                    continue

        # 이미지 정보 CSV 저장
        if image_info:
            info_df = pd.DataFrame(image_info)
            info_df.to_csv(img_dir / "image_info.csv", index=False)

        print(f"  Saved to: {img_dir}")
        print(f"  Format: NPY (raw numpy) + TIFF (lossless)")

    def export_segmentation_raw(self, layers: list):
        """세그멘테이션 결과를 원본 그대로 저장 (NPY)"""
        print(f"\n[6/9] Exporting segmentation RAW ({len(layers)} layers)...")

        seg_dir = self.output_dir / "segmentation"
        seg_dir.mkdir(exist_ok=True)

        # 클래스 이름 정보 저장
        class_names_df = pd.DataFrame([
            {'class_id': k, 'class_name': v} for k, v in DSCNN_CLASSES.items()
        ])
        class_names_df.to_csv(seg_dir / "class_names.csv", index=False)

        for i, layer in enumerate(layers):
            if i % 100 == 0:
                print(f"  Processing layer {layer} ({i+1}/{len(layers)})...")

            # 모든 클래스를 하나의 npz 파일로 저장
            seg_data = {}
            for class_id in range(12):
                try:
                    seg = self.loader.get_segmentation_result(layer, class_id)
                    seg_data[f'class_{class_id}'] = seg
                except Exception:
                    continue

            if seg_data:
                npz_path = seg_dir / f"layer_{layer:05d}_segmentation.npz"
                np.savez_compressed(npz_path, **seg_data)

        print(f"  Saved to: {seg_dir}")
        print(f"  Format: NPZ (compressed numpy)")

    def export_id_maps_raw(self, layers: list):
        """파트/샘플 ID 맵을 원본 그대로 저장 (NPY)"""
        print(f"\n[7/9] Exporting ID maps RAW ({len(layers)} layers)...")

        id_dir = self.output_dir / "id_maps"
        id_dir.mkdir(exist_ok=True)

        for i, layer in enumerate(layers):
            if i % 100 == 0:
                print(f"  Processing layer {layer} ({i+1}/{len(layers)})...")

            try:
                part_ids = self.loader.get_part_ids(layer)
                sample_ids = self.loader.get_sample_ids(layer)

                npz_path = id_dir / f"layer_{layer:05d}_ids.npz"
                np.savez_compressed(npz_path, part_ids=part_ids, sample_ids=sample_ids)

            except Exception as e:
                continue

        print(f"  Saved to: {id_dir}")

    def export_scan_paths_raw(self, layers: list):
        """스캔 경로를 원본 그대로 저장 (CSV + NPY)"""
        print(f"\n[8/9] Exporting scan paths RAW...")

        scan_dir = self.output_dir / "scan_paths"
        scan_dir.mkdir(exist_ok=True)

        scan_layers = self.loader.list_scan_layers()
        exported_count = 0

        for layer in layers:
            if layer not in scan_layers:
                continue

            try:
                scan_data = self.loader.get_scan_path(layer)

                # NPY 저장 (원본 배열)
                npy_path = scan_dir / f"layer_{layer:05d}_scanpath.npy"
                np.save(npy_path, scan_data)

                # CSV 저장 (읽기 쉬운 형식)
                scan_df = pd.DataFrame(scan_data, columns=[
                    'x_start', 'x_end', 'y_start', 'y_end', 'relative_time'
                ])
                csv_path = scan_dir / f"layer_{layer:05d}_scanpath.csv"
                scan_df.to_csv(csv_path, index=False)

                exported_count += 1

            except Exception as e:
                continue

        print(f"  Saved to: {scan_dir}")
        print(f"  Exported {exported_count} layers with scan data")

    def export_summary(self):
        """내보내기 요약 정보"""
        print("\n[9/9] Generating export summary...")

        summary = self.loader.get_summary()

        # 폴더 크기 계산
        total_size = sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())

        summary_text = f"""
{'='*60}
ORNL L-PBF Raw Data Export Summary
{'='*60}

Build Information:
  Build ID: {summary['build_id']}
  Build Name: {summary['build_name']}
  Description: {summary['description']}
  Total Layers: {summary['num_layers']}
  Expected Samples: {summary['num_samples_expected']}
  Valid Samples: {summary['num_valid_samples']}

Export Information:
  Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Output Directory: {self.output_dir}
  Total Size: {total_size / (1024*1024):.2f} MB

File Formats:
  - metadata.json: Build metadata (JSON)
  - tensile_results_*.csv: Tensile test results (CSV)
  - process_parameters.csv: Process parameters (CSV)
  - temporal_data.csv: Sensor data per layer (CSV)
  - camera_images/*.npy: Raw camera images (NumPy)
  - camera_images/*.tiff: Camera images (Lossless TIFF)
  - segmentation/*.npz: DSCNN results (Compressed NumPy)
  - id_maps/*.npz: Part/Sample ID maps (Compressed NumPy)
  - scan_paths/*.npy: Scan path raw data (NumPy)
  - scan_paths/*.csv: Scan path readable (CSV)

DSCNN Classes:
"""
        for class_id, class_name in DSCNN_CLASSES.items():
            summary_text += f"  {class_id}: {class_name}\n"

        summary_text += f"""
{'='*60}
"""

        # 저장
        summary_path = self.output_dir / "EXPORT_SUMMARY.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print(f"  Saved: {summary_path}")
        print(summary_text)


def main():
    parser = argparse.ArgumentParser(
        description='Export ORNL L-PBF HDF5 data as RAW (unprocessed) format'
    )
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to ORNL_Data_Origin directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for exported files')
    parser.add_argument('--build', type=str, default=None,
                        help='Build ID to export (e.g., B1.1, B1.2)')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layer numbers (e.g., 50,100,150)')
    parser.add_argument('--all-layers', action='store_true',
                        help='Export all layers (warning: large output)')
    parser.add_argument('--sample-layers', type=int, default=10,
                        help='Number of evenly-spaced layers to sample if --layers not specified')
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

    print(f"\nAvailable builds:")
    for f in hdf5_files:
        info = BUILD_INFO.get(f.name, {})
        print(f"  - {info.get('id', 'Unknown')}: {f.name}")

    # 빌드 선택
    if args.build:
        selected_file = None
        for f in hdf5_files:
            info = BUILD_INFO.get(f.name, {})
            if info.get('id') == args.build:
                selected_file = f
                break
        if not selected_file:
            print(f"\nBuild '{args.build}' not found")
            return
    else:
        selected_file = hdf5_files[0]
        print(f"\nNo build specified, using first: {selected_file.name}")

    # 레이어 파싱
    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    # 내보내기 실행
    info = BUILD_INFO.get(selected_file.name, {})
    build_id = info.get('id', selected_file.stem)
    output_dir = output_base / f"{build_id}_raw"

    with ORNLDataLoader(selected_file) as loader:
        exporter = ORNLRawDataExporter(loader, output_dir)
        exporter.export_all(layers=layers, all_layers=args.all_layers)


if __name__ == "__main__":
    main()
