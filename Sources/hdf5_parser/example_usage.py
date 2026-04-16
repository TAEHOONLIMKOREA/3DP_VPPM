"""
ORNL L-PBF Dataset 사용 예제
============================
ornl_data_loader.py 모듈을 사용하여 HDF5 데이터에 접근하는 예제

실행 방법:
    python example_usage.py
    python example_usage.py --layer 100 --build B1.1
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))
from ornl_data_loader import (
    ORNLDataLoader,
    list_hdf5_files,
    BUILD_INFO,
    DSCNN_CLASSES
)

# Matplotlib 백엔드 설정 (GUI 환경에서 사용)
try:
    plt.switch_backend('TkAgg')
except:
    pass


def get_hdf5_path(data_dir: Path, build_id: str = None) -> Path:
    """빌드 ID로 HDF5 파일 경로 찾기"""
    hdf5_files = list_hdf5_files(data_dir)

    if not hdf5_files:
        raise FileNotFoundError(f"HDF5 파일을 찾을 수 없습니다: {data_dir}")

    if build_id is None:
        return hdf5_files[0]

    for hdf5_file in hdf5_files:
        info = BUILD_INFO.get(hdf5_file.name, {})
        if info.get('id') == build_id:
            return hdf5_file

    print(f"빌드 ID '{build_id}'를 찾을 수 없습니다. 첫 번째 파일을 사용합니다.")
    return hdf5_files[0]


def example_metadata(loader: ORNLDataLoader):
    """1. 빌드 메타데이터 접근 예제"""
    print("\n" + "="*60)
    print("1. BUILD METADATA")
    print("="*60)

    # 빌드 요약
    summary = loader.get_summary()
    print(f"\nBuild Name: {summary['build_name']}")
    print(f"Build ID: {summary['build_id']}")
    print(f"Description: {summary['description']}")
    print(f"Total Layers: {summary['num_layers']}")
    print(f"Expected Samples: {summary['num_samples_expected']}")
    print(f"Valid Samples: {summary['num_valid_samples']}")

    # 재료 정보
    print("\n[ Material Info ]")
    material_info = loader.get_material_info()
    for key, val in list(material_info.items())[:5]:
        print(f"  {key}: {val}")

    # 프린터 정보
    print("\n[ Printer Info ]")
    printer_info = loader.get_printer_info()
    for key, val in list(printer_info.items())[:5]:
        print(f"  {key}: {val}")


def example_slice_data(loader: ORNLDataLoader, layer: int):
    """2. 슬라이스 데이터 (레이어별 이미지) 접근 예제"""
    print("\n" + "="*60)
    print(f"2. SLICE DATA (Layer {layer})")
    print("="*60)

    # 카메라 이미지
    try:
        img_post_melt = loader.get_camera_image(layer, camera_id=0)
        img_post_powder = loader.get_camera_image(layer, camera_id=1)
        print(f"\nCamera Image (post-melt) shape: {img_post_melt.shape}")
        print(f"Camera Image (post-powder) shape: {img_post_powder.shape}")
    except KeyError as e:
        print(f"Camera image not available: {e}")

    # 파트/샘플 ID 맵
    try:
        part_ids = loader.get_part_ids(layer)
        sample_ids = loader.get_sample_ids(layer)
        print(f"Part IDs: {np.unique(part_ids)[:10]}... (unique values)")
        print(f"Sample IDs: {np.unique(sample_ids)[:10]}... (unique values)")
    except KeyError as e:
        print(f"Part/Sample IDs not available: {e}")

    # DSCNN 세그멘테이션 결과
    print("\n[ DSCNN Segmentation Classes ]")
    for class_id, class_name in list(DSCNN_CLASSES.items())[:6]:
        try:
            seg = loader.get_segmentation_result(layer, class_id)
            coverage = np.sum(seg > 0) / seg.size * 100
            print(f"  Class {class_id} ({class_name}): {coverage:.2f}% coverage")
        except KeyError:
            print(f"  Class {class_id} ({class_name}): Not available")


def example_temporal_data(loader: ORNLDataLoader):
    """3. 시간적 데이터 (센서 로그) 접근 예제"""
    print("\n" + "="*60)
    print("3. TEMPORAL DATA (Sensor Logs)")
    print("="*60)

    # 사용 가능한 키 목록
    temporal_keys = loader.list_temporal_keys()
    print(f"\nAvailable temporal data ({len(temporal_keys)} keys):")
    for key in temporal_keys[:10]:
        data = loader.get_temporal_data(key)
        print(f"  {key}: shape={data.shape}, range=[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")

    # 레이어 시간
    try:
        layer_times = loader.get_layer_times()
        print(f"\nLayer Times:")
        print(f"  Total build time: {np.sum(layer_times):.1f} seconds")
        print(f"  Mean layer time: {np.mean(layer_times):.2f} seconds")
        print(f"  Min/Max layer time: {np.min(layer_times):.2f} / {np.max(layer_times):.2f} seconds")
    except KeyError:
        print("Layer times not available")

    # 산소 농도
    try:
        gas_loop_o2, module_o2 = loader.get_oxygen_levels()
        print(f"\nOxygen Levels:")
        print(f"  Gas loop O2: mean={np.nanmean(gas_loop_o2):.4f}")
        print(f"  Module O2: mean={np.nanmean(module_o2):.4f}")
    except KeyError:
        print("Oxygen data not available")


def example_scan_path(loader: ORNLDataLoader, layer: int):
    """4. 스캔 경로 데이터 접근 예제"""
    print("\n" + "="*60)
    print(f"4. SCAN PATH DATA (Layer {layer})")
    print("="*60)

    # 스캔 경로가 있는 레이어 확인
    scan_layers = loader.list_scan_layers()
    print(f"\nLayers with scan data: {len(scan_layers)} layers")
    if scan_layers:
        print(f"  First 10: {scan_layers[:10]}")

    # 특정 레이어 스캔 경로
    if layer in scan_layers:
        x, y, t = loader.get_scan_path_xy(layer)
        print(f"\nScan path for layer {layer}:")
        print(f"  Number of vectors: {len(t)}")
        print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"  Time range: [{t.min():.4f}, {t.max():.4f}]")
    else:
        print(f"\nNo scan path available for layer {layer}")
        if scan_layers:
            print(f"  Try layer {scan_layers[len(scan_layers)//2]} instead")


def example_tensile_data(loader: ORNLDataLoader):
    """5. 인장 시험 결과 접근 예제"""
    print("\n" + "="*60)
    print("5. TENSILE TEST RESULTS")
    print("="*60)

    # 시험 결과
    results = loader.get_test_results('samples')
    print("\nAvailable test results:")
    for key, data in results.items():
        valid_count = np.sum(~np.isnan(data))
        if valid_count > 0:
            valid_data = data[~np.isnan(data)]
            print(f"  {key}:")
            print(f"    Valid samples: {valid_count}")
            print(f"    Mean: {np.mean(valid_data):.2f}")
            print(f"    Std: {np.std(valid_data):.2f}")
            print(f"    Range: [{np.min(valid_data):.2f}, {np.max(valid_data):.2f}]")

    # 특정 샘플 인장 특성
    valid_samples = loader.get_valid_samples()
    if len(valid_samples) > 0:
        sample_id = valid_samples[0]
        props = loader.get_tensile_properties(sample_id)
        print(f"\nSample {sample_id} tensile properties:")
        for key, val in props.items():
            if val is not None:
                print(f"  {key}: {val:.2f}")


def example_process_params(loader: ORNLDataLoader):
    """6. 공정 파라미터 접근 예제"""
    print("\n" + "="*60)
    print("6. PROCESS PARAMETERS")
    print("="*60)

    params = loader.get_process_parameters()
    print("\nProcess parameters (all parts):")
    for key, data in params.items():
        if data is not None and len(data) > 0:
            if isinstance(data[0], (np.bytes_, bytes)):
                unique_vals = set(v.decode() if isinstance(v, bytes) else str(v)
                                  for v in data if v)
                print(f"  {key}: {unique_vals}")
            else:
                valid_data = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data
                if len(valid_data) > 0:
                    print(f"  {key}: mean={np.mean(valid_data):.2f}, unique={len(np.unique(valid_data))}")


def example_visualization(loader: ORNLDataLoader, layer: int, save_dir: Path = None):
    """7. 시각화 예제"""
    print("\n" + "="*60)
    print("7. VISUALIZATION EXAMPLES")
    print("="*60)

    # Figure 생성
    fig = plt.figure(figsize=(16, 12))

    # 1. 카메라 이미지
    try:
        ax1 = fig.add_subplot(2, 3, 1)
        loader.plot_camera_image(layer, camera_id=0, ax=ax1, cmap='gray')
        print("  - Camera image plotted")
    except Exception as e:
        print(f"  - Camera image failed: {e}")

    # 2. DSCNN 세그멘테이션 (Printed 클래스)
    try:
        ax2 = fig.add_subplot(2, 3, 2)
        loader.plot_segmentation(layer, class_id=1, ax=ax2)
        print("  - Segmentation plotted")
    except Exception as e:
        print(f"  - Segmentation failed: {e}")

    # 3. 스캔 경로
    try:
        ax3 = fig.add_subplot(2, 3, 3)
        loader.plot_scan_path(layer, ax=ax3)
        print("  - Scan path plotted")
    except Exception as e:
        print(f"  - Scan path failed: {e}")

    # 4. 시간적 데이터 (산소 농도)
    try:
        ax4 = fig.add_subplot(2, 3, 4)
        loader.plot_temporal_data('module_oxygen', ax=ax4)
        print("  - Temporal data plotted")
    except Exception as e:
        print(f"  - Temporal data failed: {e}")

    # 5. 인장강도 분포
    try:
        ax5 = fig.add_subplot(2, 3, 5)
        loader.plot_tensile_distribution('ultimate_tensile_strength', ax=ax5)
        print("  - Tensile distribution plotted")
    except Exception as e:
        print(f"  - Tensile distribution failed: {e}")

    # 6. 항복강도 vs 인장강도
    try:
        ax6 = fig.add_subplot(2, 3, 6)
        results = loader.get_test_results('samples')
        ys = results.get('yield_strength', np.array([]))
        uts = results.get('ultimate_tensile_strength', np.array([]))
        valid_mask = ~np.isnan(ys) & ~np.isnan(uts)
        ax6.scatter(ys[valid_mask], uts[valid_mask], alpha=0.5, s=10)
        ax6.set_xlabel('Yield Strength (MPa)')
        ax6.set_ylabel('Ultimate Tensile Strength (MPa)')
        ax6.set_title('YS vs UTS')
        ax6.grid(True, alpha=0.3)
        print("  - YS vs UTS plotted")
    except Exception as e:
        print(f"  - YS vs UTS failed: {e}")

    plt.tight_layout()

    # 저장 또는 표시
    if save_dir:
        save_path = save_dir / f"visualization_{loader.build_id}_layer{layer}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        print("\nDisplaying figure... (close window to continue)")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='ORNL L-PBF Dataset Example Usage')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to ORNL_Data_Origin directory')
    parser.add_argument('--build', type=str, default=None,
                        help='Build ID (e.g., B1.1, B1.2)')
    parser.add_argument('--layer', type=int, default=50,
                        help='Layer number to analyze')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save visualization')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization')
    args = parser.parse_args()

    # 데이터 디렉토리 설정
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent.parent / "ORNL_Data_Origin"

    print(f"Data directory: {data_dir}")

    # HDF5 파일 목록 출력
    hdf5_files = list_hdf5_files(data_dir)
    print(f"\nAvailable HDF5 files ({len(hdf5_files)}):")
    for f in hdf5_files:
        info = BUILD_INFO.get(f.name, {})
        print(f"  - {f.name}")
        print(f"    ID: {info.get('id', 'Unknown')}, Samples: {info.get('samples', '?')}")

    if not hdf5_files:
        print("\nNo HDF5 files found. Please check the data directory.")
        return

    # HDF5 파일 선택
    hdf5_path = get_hdf5_path(data_dir, args.build)
    print(f"\nSelected file: {hdf5_path.name}")

    # 데이터 로더 사용
    with ORNLDataLoader(hdf5_path) as loader:
        # 1. 메타데이터
        example_metadata(loader)

        # 2. 슬라이스 데이터
        example_slice_data(loader, args.layer)

        # 3. 시간적 데이터
        example_temporal_data(loader)

        # 4. 스캔 경로
        example_scan_path(loader, args.layer)

        # 5. 인장 시험 결과
        example_tensile_data(loader)

        # 6. 공정 파라미터
        example_process_params(loader)

        # 7. 시각화
        if not args.no_viz:
            save_dir = Path(args.save_dir) if args.save_dir else None
            example_visualization(loader, args.layer, save_dir)

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
