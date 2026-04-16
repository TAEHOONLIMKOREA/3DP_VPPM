"""
ORNL L-PBF Tensile Property Dataset Loader
==========================================
HDF5 데이터 파싱 및 접근을 위한 모듈

Reference:
- Scime, L. et al. "A Data-Driven Framework for Direct Local Tensile Property
  Prediction of Laser Powder Bed Fusion Parts" Materials 2023, 16, 7293

Dataset: Peregrine v2023-11
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.collections as collections
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import warnings


# DSCNN 이상 분류 클래스
DSCNN_CLASSES = {
    0: "Powder",              # 분말 베드 영역
    1: "Printed",             # 정상 프린트 영역
    2: "Recoater Hopping",    # 리코터 호핑
    3: "Recoater Streaking",  # 리코터 줄무늬
    4: "Incomplete Spreading", # 불완전 분말 도포
    5: "Swelling",            # 표면 팽창/변형
    6: "Debris",              # 이물질
    7: "Super-Elevation",     # 분말 커버리지 부족
    8: "Spatter",             # 스패터 입자
    9: "Misprint",            # 오프린트
    10: "Over Melting",       # 과용융
    11: "Under Melting",      # 저용융
}

# 빌드 정보
BUILD_INFO = {
    "2021-07-13 TCR Phase 1 Build 1.hdf5": {"id": "B1.1", "samples": 503, "description": "기준 공정 조건"},
    "2021-04-16 TCR Phase 1 Build 2.hdf5": {"id": "B1.2", "samples": 2705, "description": "다양한 공정 파라미터"},
    "2021-04-28 TCR Phase 1 Build 3.hdf5": {"id": "B1.3", "samples": 813, "description": "오버행 형상"},
    "2021-08-03 TCR Phase 1 Build 4.hdf5": {"id": "B1.4", "samples": 694, "description": "스패터/가스 유량 변화"},
    "2021-08-23 TCR Phase 1 Build 5.hdf5": {"id": "B1.5", "samples": 1584, "description": "리코터 손상/분말 공급 부족"},
}


class ORNLDataLoader:
    """ORNL L-PBF HDF5 데이터 로더 클래스"""

    def __init__(self, hdf5_path: str):
        """
        Args:
            hdf5_path: HDF5 파일 경로
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 파일을 찾을 수 없습니다: {hdf5_path}")

        self._file = None
        self._build_name = self.hdf5_path.name
        self._build_info = BUILD_INFO.get(self._build_name, {})

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """HDF5 파일 열기"""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
        return self

    def close(self):
        """HDF5 파일 닫기"""
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        """열린 HDF5 파일 객체 반환"""
        if self._file is None:
            raise RuntimeError("파일이 열려있지 않습니다. open() 또는 with 문을 사용하세요.")
        return self._file

    @property
    def build_id(self) -> str:
        """빌드 ID 반환 (예: B1.1)"""
        return self._build_info.get("id", "Unknown")

    @property
    def num_samples(self) -> int:
        """총 샘플 수 반환"""
        return self._build_info.get("samples", 0)

    # =========================================================================
    # Build Metadata
    # =========================================================================

    def get_build_metadata(self) -> Dict:
        """빌드 메타데이터 반환"""
        metadata = {}
        for key in self.file.attrs:
            metadata[key] = self.file.attrs[key]
        return metadata

    def get_build_name(self) -> str:
        """빌드 이름 반환"""
        return str(self.file.attrs.get('core/build_name', 'Unknown'))

    def get_material_info(self) -> Dict:
        """재료 정보 반환"""
        info = {}
        for key in self.file.attrs:
            if key.startswith('material/'):
                info[key.replace('material/', '')] = self.file.attrs[key]
        return info

    def get_printer_info(self) -> Dict:
        """프린터 정보 반환"""
        info = {}
        for key in self.file.attrs:
            if key.startswith('printer/'):
                info[key.replace('printer/', '')] = self.file.attrs[key]
        return info

    def get_num_layers(self) -> int:
        """총 레이어 수 반환"""
        # slices/camera_data/visible/0 의 shape에서 레이어 수 추출
        if 'slices/camera_data/visible/0' in self.file:
            return self.file['slices/camera_data/visible/0'].shape[0]
        return 0

    # =========================================================================
    # Reference Images
    # =========================================================================

    def get_reference_image(self, image_name: str = 'thumbnail') -> np.ndarray:
        """
        참조 이미지 반환

        Args:
            image_name: 이미지 이름 (예: 'thumbnail')
        """
        key = f'reference_images/{image_name}'
        if key in self.file:
            return self.file[key][...]
        raise KeyError(f"참조 이미지를 찾을 수 없습니다: {image_name}")

    def list_reference_images(self) -> List[str]:
        """사용 가능한 참조 이미지 목록 반환"""
        if 'reference_images' in self.file:
            return list(self.file['reference_images'].keys())
        return []

    # =========================================================================
    # Slice Data (Layer-wise 2D Arrays)
    # =========================================================================

    def get_camera_image(self, layer: int, camera_id: int = 0) -> np.ndarray:
        """
        특정 레이어의 카메라 이미지 반환

        Args:
            layer: 레이어 번호
            camera_id: 카메라 ID (0=용융 직후, 1=분말 도포 직후)
        """
        key = f'slices/camera_data/visible/{camera_id}'
        if key in self.file:
            return self.file[key][layer, ...]
        raise KeyError(f"카메라 데이터를 찾을 수 없습니다: camera_id={camera_id}")

    def get_part_ids(self, layer: int) -> np.ndarray:
        """특정 레이어의 파트 ID 맵 반환"""
        key = 'slices/part_ids'
        if key in self.file:
            return self.file[key][layer, ...]
        raise KeyError("파트 ID 데이터를 찾을 수 없습니다")

    def get_sample_ids(self, layer: int) -> np.ndarray:
        """특정 레이어의 샘플 ID 맵 반환"""
        key = 'slices/sample_ids'
        if key in self.file:
            return self.file[key][layer, ...]
        raise KeyError("샘플 ID 데이터를 찾을 수 없습니다")

    def get_segmentation_result(self, layer: int, class_id: int) -> np.ndarray:
        """
        특정 레이어의 DSCNN 세그멘테이션 결과 반환

        Args:
            layer: 레이어 번호
            class_id: DSCNN 클래스 ID (0-11)
        """
        key = f'slices/segmentation_results/{class_id}'
        if key in self.file:
            return self.file[key][layer, ...]
        raise KeyError(f"세그멘테이션 결과를 찾을 수 없습니다: class_id={class_id}")

    def get_all_segmentation_results(self, layer: int) -> Dict[int, np.ndarray]:
        """특정 레이어의 모든 DSCNN 세그멘테이션 결과 반환"""
        results = {}
        for class_id in range(12):
            try:
                results[class_id] = self.get_segmentation_result(layer, class_id)
            except KeyError:
                continue
        return results

    def get_segmentation_class_names(self) -> List[str]:
        """DSCNN 클래스 이름 목록 반환"""
        key = 'slices/segmentation_results/class_names'
        if key in self.file:
            return [name.decode() if isinstance(name, bytes) else name
                    for name in self.file[key][...]]
        return list(DSCNN_CLASSES.values())

    def get_slice_origin(self) -> np.ndarray:
        """슬라이스 좌표계 원점 반환"""
        if 'slices/origin' in self.file:
            return self.file['slices/origin'][...]
        return None

    # =========================================================================
    # Temporal Data (Layer-wise 1D Arrays)
    # =========================================================================

    def get_temporal_data(self, key: str) -> np.ndarray:
        """
        시간적 데이터 반환

        Args:
            key: 데이터 키 (예: 'top_flow_rate', 'module_oxygen')
        """
        full_key = f'temporal/{key}'
        if full_key in self.file:
            return self.file[full_key][...]
        raise KeyError(f"시간적 데이터를 찾을 수 없습니다: {key}")

    def list_temporal_keys(self) -> List[str]:
        """사용 가능한 시간적 데이터 키 목록 반환"""
        if 'temporal' in self.file:
            return list(self.file['temporal'].keys())
        return []

    def get_all_temporal_data(self) -> Dict[str, np.ndarray]:
        """모든 시간적 데이터 반환"""
        data = {}
        for key in self.list_temporal_keys():
            data[key] = self.get_temporal_data(key)
        return data

    def get_layer_times(self) -> np.ndarray:
        """각 레이어 프린트 소요 시간 반환"""
        return self.get_temporal_data('layer_times')

    def get_oxygen_levels(self) -> Tuple[np.ndarray, np.ndarray]:
        """산소 농도 데이터 반환 (gas_loop, module)"""
        gas_loop = self.get_temporal_data('gas_loop_oxygen')
        module = self.get_temporal_data('module_oxygen')
        return gas_loop, module

    def get_temperatures(self) -> Dict[str, np.ndarray]:
        """온도 관련 데이터 반환"""
        temp_keys = [
            'bottom_chamber_temperature',
            'top_chamber_temperature',
            'build_plate_temperature',
            'bottom_flow_temperature',
            'top_flow_temperature',
            'glass_scale_temperature',
            'laser_rail_temperature'
        ]
        temps = {}
        for key in temp_keys:
            try:
                temps[key] = self.get_temporal_data(key)
            except KeyError:
                continue
        return temps

    # =========================================================================
    # Scan Path Data
    # =========================================================================

    def get_scan_path(self, layer: int) -> np.ndarray:
        """
        특정 레이어의 스캔 경로 반환

        Returns:
            5열 배열: [x_start, x_end, y_start, y_end, relative_time]
        """
        key = f'scans/{layer}'
        if key in self.file:
            return self.file[key][...]
        raise KeyError(f"스캔 경로를 찾을 수 없습니다: layer={layer}")

    def get_scan_path_xy(self, layer: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        스캔 경로의 X, Y 좌표 및 시간 반환

        Returns:
            (x_coords, y_coords, times): 각각 (N, 2) 배열과 (N,) 배열
        """
        scan_data = self.get_scan_path(layer)
        x = scan_data[:, 0:2]  # [x_start, x_end]
        y = scan_data[:, 2:4]  # [y_start, y_end]
        t = scan_data[:, 4]    # relative_time
        return x, y, t

    def list_scan_layers(self) -> List[int]:
        """스캔 경로가 있는 레이어 번호 목록 반환"""
        if 'scans' in self.file:
            return sorted([int(k) for k in self.file['scans'].keys() if k.isdigit()])
        return []

    # =========================================================================
    # Part and Sample Data
    # =========================================================================

    def get_process_parameters(self, part_id: Optional[int] = None) -> Union[Dict, np.ndarray]:
        """
        공정 파라미터 반환

        Args:
            part_id: 특정 파트 ID (None이면 모든 파트)
        """
        params = {}
        param_keys = [
            'hatch_spacing', 'laser_beam_power', 'laser_beam_speed',
            'laser_module', 'laser_spot_size', 'parameter_set',
            'scan_rotation', 'stripe_width'
        ]

        for key in param_keys:
            full_key = f'parts/process_parameters/{key}'
            if full_key in self.file:
                data = self.file[full_key][...]
                if part_id is not None:
                    params[key] = data[part_id] if part_id < len(data) else None
                else:
                    params[key] = data

        return params

    def get_test_results(self, sample_type: str = 'samples') -> Dict[str, np.ndarray]:
        """
        인장 시험 결과 반환

        Args:
            sample_type: 'samples' 또는 'parts'
        """
        results = {}
        result_keys = [
            'yield_strength', 'ultimate_tensile_strength',
            'uniform_elongation', 'total_elongation',
            'burst_pressure', 'burst_temperature'
        ]

        for key in result_keys:
            full_key = f'{sample_type}/test_results/{key}'
            if full_key in self.file:
                results[key] = self.file[full_key][...]

        return results

    def get_tensile_properties(self, sample_id: Optional[int] = None) -> Dict:
        """
        인장 특성 반환 (YS, UTS, UE, TE)

        Args:
            sample_id: 특정 샘플 ID (None이면 모든 샘플)
        """
        results = self.get_test_results('samples')

        if sample_id is not None:
            return {
                'yield_strength': results.get('yield_strength', [None])[sample_id],
                'ultimate_tensile_strength': results.get('ultimate_tensile_strength', [None])[sample_id],
                'uniform_elongation': results.get('uniform_elongation', [None])[sample_id],
                'total_elongation': results.get('total_elongation', [None])[sample_id],
            }
        return results

    def get_valid_samples(self) -> np.ndarray:
        """유효한 (NaN이 아닌) 샘플 인덱스 반환"""
        results = self.get_test_results('samples')
        uts = results.get('ultimate_tensile_strength', np.array([]))
        if len(uts) > 0:
            return np.where(~np.isnan(uts))[0]
        return np.array([])

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_camera_image(self, layer: int, camera_id: int = 0,
                          cmap: str = 'gray', ax=None, **kwargs):
        """카메라 이미지 시각화"""
        img = self.get_camera_image(layer, camera_id)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        im = ax.imshow(img, cmap=cmap, interpolation='none', **kwargs)
        ax.set_title(f'Layer {layer} - Camera {camera_id}')
        plt.colorbar(im, ax=ax)
        return ax

    def plot_segmentation(self, layer: int, class_id: int,
                          cmap: str = 'jet', ax=None, **kwargs):
        """DSCNN 세그멘테이션 결과 시각화"""
        seg = self.get_segmentation_result(layer, class_id)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        class_name = DSCNN_CLASSES.get(class_id, f'Class {class_id}')
        im = ax.imshow(seg, cmap=cmap, interpolation='none', **kwargs)
        ax.set_title(f'Layer {layer} - {class_name}')
        plt.colorbar(im, ax=ax)
        return ax

    def plot_scan_path(self, layer: int, cmap: str = 'jet', ax=None):
        """스캔 경로 시각화"""
        x, y, t = self.get_scan_path_xy(layer)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        colorizer = cm.ScalarMappable(
            norm=mcolors.Normalize(np.min(t), np.max(t)),
            cmap=cmap
        )
        line_collection = collections.LineCollection(
            np.stack([x, y], axis=-1),
            colors=colorizer.to_rgba(t)
        )

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.add_collection(line_collection)
        ax.set_aspect('equal')
        ax.set_title(f'Scan Path - Layer {layer}')
        plt.colorbar(colorizer, ax=ax, label='Relative Time')
        return ax

    def plot_temporal_data(self, key: str, ax=None, **kwargs):
        """시간적 데이터 시각화"""
        data = self.get_temporal_data(key)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(np.arange(len(data)), data, **kwargs)
        ax.set_xlabel('Layer')
        ax.set_ylabel(key.replace('_', ' ').title())
        ax.set_title(f'Temporal Data: {key}')
        ax.grid(True, alpha=0.3)
        return ax

    def plot_tensile_distribution(self, property_name: str = 'ultimate_tensile_strength',
                                  ax=None, **kwargs):
        """인장 특성 분포 시각화"""
        results = self.get_test_results('samples')
        data = results.get(property_name, np.array([]))
        valid_data = data[~np.isnan(data)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(valid_data, bins=50, edgecolor='black', alpha=0.7, **kwargs)
        ax.axvline(np.mean(valid_data), color='r', linestyle='--',
                   label=f'Mean: {np.mean(valid_data):.1f}')
        ax.set_xlabel(property_name.replace('_', ' ').title())
        ax.set_ylabel('Count')
        ax.set_title(f'{property_name.replace("_", " ").title()} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def print_structure(self, max_depth: int = 3):
        """HDF5 파일 구조 출력"""
        print(f"\n{'='*60}")
        print(f"HDF5 File: {self._build_name}")
        print(f"Build ID: {self.build_id}")
        print(f"{'='*60}\n")

        # Top-level attributes
        print("[ Attributes ]")
        for key in sorted(self.file.attrs.keys())[:10]:
            val = str(self.file.attrs[key]).split('\n')[0][:50]
            print(f"  {key}: {val}")
        print(f"  ... ({len(self.file.attrs)} total attributes)\n")

        # Groups and datasets
        print("[ Groups & Datasets ]")
        def print_item(name, obj, depth=0):
            if depth > max_depth:
                return
            indent = "  " * depth
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 {name}: {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 {name}/")
                for key in list(obj.keys())[:5]:
                    print_item(f"{name}/{key}", obj[key], depth + 1)
                if len(obj.keys()) > 5:
                    print(f"{indent}  ... ({len(obj.keys())} items)")

        for key in self.file.keys():
            print_item(key, self.file[key])

    def get_summary(self) -> Dict:
        """데이터셋 요약 정보 반환"""
        results = self.get_test_results('samples')
        uts = results.get('ultimate_tensile_strength', np.array([]))
        valid_mask = ~np.isnan(uts) if len(uts) > 0 else np.array([])

        return {
            'build_name': self._build_name,
            'build_id': self.build_id,
            'description': self._build_info.get('description', ''),
            'num_layers': self.get_num_layers(),
            'num_samples_expected': self.num_samples,
            'num_valid_samples': np.sum(valid_mask) if len(valid_mask) > 0 else 0,
            'temporal_keys': self.list_temporal_keys(),
            'reference_images': self.list_reference_images(),
        }


def list_hdf5_files(data_dir: str) -> List[Path]:
    """ORNL_Data 디렉토리의 HDF5 파일 목록 반환"""
    data_path = Path(data_dir)
    return sorted(data_path.glob("*.hdf5"))


def load_all_builds(data_dir: str) -> Dict[str, ORNLDataLoader]:
    """모든 빌드 데이터 로더 생성"""
    loaders = {}
    for hdf5_file in list_hdf5_files(data_dir):
        build_info = BUILD_INFO.get(hdf5_file.name, {})
        build_id = build_info.get('id', hdf5_file.stem)
        loaders[build_id] = ORNLDataLoader(hdf5_file)
    return loaders


if __name__ == "__main__":
    # 기본 테스트
    import sys

    if len(sys.argv) > 1:
        hdf5_path = sys.argv[1]
    else:
        # 기본 경로 설정
        default_dir = Path(__file__).parent.parent.parent / "ORNL_Data_Origin"
        hdf5_files = list(default_dir.glob("*.hdf5"))
        if hdf5_files:
            hdf5_path = hdf5_files[0]
        else:
            print("HDF5 파일을 찾을 수 없습니다.")
            print("사용법: python ornl_data_loader.py <hdf5_file_path>")
            sys.exit(1)

    print(f"Loading: {hdf5_path}")

    with ORNLDataLoader(hdf5_path) as loader:
        loader.print_structure()
        summary = loader.get_summary()
        print("\n[ Summary ]")
        for key, val in summary.items():
            print(f"  {key}: {val}")
