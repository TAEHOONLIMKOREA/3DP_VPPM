"""Phase L2 — 샘플 단위 시퀀스 Dataset & collate.

캐시 (sample_stacks/{build}.h5) 를 읽어 가변 길이 시퀀스를 batch 로 묶는다.
타겟은 4 인장 물성 — VPPM 학습과 동일한 [-1, 1] 정규화.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..common import config


def load_targets_for_samples() -> dict:
    """각 sample_id 의 4 인장 물성 + build_id 를 모은다.

    빌드별 HDF5 의 samples/test_results 에서 직접 로드.
    sample_id 는 빌드 안에서 0-based, 빌드 간 중복 가능 — 본 모듈에서는 (build_id, sample_id_in_build)
    가 아닌 **글로벌 sample_id** 를 가정 (sample_stack 의 캐시 파일들이 빌드별로 분리됨).
    """
    out = {}  # (build_id, local_sid) -> {YS, UTS, UE, TE}
    for bid, fname in config.BUILDS.items():
        path = config.HDF5_DIR / fname
        with h5py.File(path, "r") as f:
            tr = f["samples/test_results"]
            ys = tr["yield_strength"][...]
            uts = tr["ultimate_tensile_strength"][...]
            ue = tr["uniform_elongation"][...]
            te = tr["total_elongation"][...]
        for sid in range(len(ys)):
            out[(bid, int(sid))] = {
                "yield_strength": float(ys[sid]),
                "ultimate_tensile_strength": float(uts[sid]),
                "uniform_elongation": float(ue[sid]),
                "total_elongation": float(te[sid]),
            }
    return out


class SampleStackIndex:
    """캐시된 모든 빌드의 (build_id, local_sid) 인덱스.

    Iterator 가 아닌 random-access 인덱스. fold split 시 사용.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.entries: list[tuple[str, int, int, int]] = []  # (build, local_sid, offset_l, offset_r)
        for bid in config.BUILDS:
            path = self.cache_dir / f"{bid}.h5"
            if not path.exists():
                continue
            with h5py.File(path, "r") as f:
                sids = f["sample_ids"][...]
                offs = f["layer_offsets"][...]
                lens = f["seq_lengths"][...]
                for i, sid in enumerate(sids):
                    if lens[i] == 0:
                        continue
                    self.entries.append((bid, int(sid), int(offs[i]), int(offs[i + 1])))

    def __len__(self):
        return len(self.entries)


class SampleSequenceDataset(Dataset):
    """주어진 entry list 에 대해 (sequence, mask, targets) 를 반환.

    빌드별 h5 파일을 lazy open — multi-worker 안전.
    """

    def __init__(
        self,
        entries: list[tuple[str, int, int, int]],
        cache_dir: Path,
        targets_map: dict,
        target_norm: dict,        # {prop: (t_min, t_max)}
        raw_min: float,
        raw_max: float,
        max_seq_len: int | None = None,
    ):
        self.entries = entries
        self.cache_dir = Path(cache_dir)
        self.targets_map = targets_map
        self.target_norm = target_norm
        self.raw_min = float(raw_min)
        self.raw_max = float(raw_max)
        self.max_seq_len = max_seq_len
        self._files: dict[str, h5py.File] = {}

    def _open(self, build_id: str) -> h5py.File:
        if build_id not in self._files:
            self._files[build_id] = h5py.File(self.cache_dir / f"{build_id}.h5", "r")
        return self._files[build_id]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        build_id, sid, lo, hi = self.entries[idx]
        f = self._open(build_id)
        seq = f["sequences"][lo:hi].astype(np.float32)         # (T, H, W)
        # 정규화 [0, 1]  ← per-build cache 는 raw float 저장. 학습 시 정규화.
        denom = max(self.raw_max - self.raw_min, 1e-6)
        seq = (seq - self.raw_min) / denom
        seq = np.clip(seq, 0.0, 1.0)

        if self.max_seq_len is not None and seq.shape[0] > self.max_seq_len:
            seq = seq[: self.max_seq_len]

        # (T, 1, H, W)
        seq_t = torch.from_numpy(seq).unsqueeze(1).float()
        T = seq_t.shape[0]

        tgt = self.targets_map[(build_id, sid)]
        y = []
        for prop in config.TARGET_PROPERTIES:
            tmin, tmax = self.target_norm[prop]
            v = tgt[prop]
            y.append(2.0 * (v - tmin) / max(tmax - tmin, 1e-6) - 1.0)
        y_t = torch.tensor(y, dtype=torch.float32)             # (4,)

        return {
            "seq": seq_t,
            "length": T,
            "y": y_t,
            "build_id": build_id,
            "sample_id": sid,
        }


def collate_pad(batch: list[dict]) -> dict:
    """가변 길이 시퀀스를 padding."""
    seqs = [b["seq"] for b in batch]                            # list of (T_i, 1, H, W)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    y = torch.stack([b["y"] for b in batch], dim=0)             # (B, 4)

    # pad_sequence: (max_T, B, 1, H, W) → permute 해 (B, max_T, 1, H, W)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]   # (B, max_T)
    return {
        "seq": padded,
        "lengths": lengths,
        "mask": mask,
        "y": y,
        "build_ids": [b["build_id"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
    }


def fit_target_normalization(entries, targets_map) -> dict:
    """학습 데이터의 4 물성 min/max 통계."""
    out = {}
    for prop in config.TARGET_PROPERTIES:
        vals = []
        for (bid, sid, _, _) in entries:
            v = targets_map[(bid, sid)][prop]
            if not np.isnan(v):
                vals.append(v)
        out[prop] = (float(min(vals)), float(max(vals)))
    return out


def load_normalization_json(cache_dir: Path) -> dict:
    p = Path(cache_dir) / "normalization.json"
    with open(p, "r") as f:
        return json.load(f)
