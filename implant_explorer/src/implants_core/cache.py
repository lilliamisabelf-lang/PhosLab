"""
implants_core.cache – Lightweight caching for contacts & RF results
====================================================================
Simple dict caches keyed by deterministic hashes.
Optional file-backed persistence via ``np.savez``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _array_hash(arr: np.ndarray) -> str:
    """Fast hash of an array's contents."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


class ContactCache:
    """
    Cache world-space contacts keyed by ``(spec_hash, transform_hash)``.

    Avoids recomputing ``contacts_world`` when only the RF source changes.
    """

    def __init__(self, max_size: int = 64):
        self._store: Dict[str, np.ndarray] = {}
        self._max_size = max_size

    def _key(self, spec_hash: str, transform: np.ndarray) -> str:
        t_hash = _array_hash(transform.astype(np.float64))
        return f"{spec_hash}_{t_hash}"

    def get(self, spec_hash: str, transform: np.ndarray) -> Optional[np.ndarray]:
        return self._store.get(self._key(spec_hash, transform))

    def put(self, spec_hash: str, transform: np.ndarray, contacts_world: np.ndarray) -> None:
        if len(self._store) >= self._max_size:
            # evict oldest entry (FIFO)
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[self._key(spec_hash, transform)] = contacts_world.copy()

    def clear(self) -> None:
        self._store.clear()


class RFCache:
    """
    Cache RF/coverage results keyed by
    ``(contacts_world_hash, rf_source_id, dataset_id)``.
    """

    def __init__(self, max_size: int = 32):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size

    def _key(self, contacts_world: np.ndarray, rf_source_id: str, dataset_id: str) -> str:
        c_hash = _array_hash(contacts_world.astype(np.float64))
        raw = json.dumps({"c": c_hash, "rf": rf_source_id, "ds": dataset_id},
                         sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def get(
        self,
        contacts_world: np.ndarray,
        rf_source_id: str,
        dataset_id: str,
    ) -> Optional[Dict[str, Any]]:
        return self._store.get(self._key(contacts_world, rf_source_id, dataset_id))

    def put(
        self,
        contacts_world: np.ndarray,
        rf_source_id: str,
        dataset_id: str,
        result: Dict[str, Any],
    ) -> None:
        if len(self._store) >= self._max_size:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[self._key(contacts_world, rf_source_id, dataset_id)] = result

    def clear(self) -> None:
        self._store.clear()

    def save(self, path: str | Path) -> None:
        """Persist cache to disk (NumPy npz)."""
        data = {}
        for i, (k, v) in enumerate(self._store.items()):
            data[f"key_{i}"] = np.array([k], dtype="U")
            for field, arr in v.items():
                if isinstance(arr, np.ndarray):
                    data[f"val_{i}_{field}"] = arr
        np.savez(path, **data)

    def load(self, path: str | Path) -> None:
        """Restore cache from disk."""
        if not Path(path).exists():
            return
        data = dict(np.load(path, allow_pickle=False))
        # Reconstruct is non-trivial; for now just clear and skip
        # (full implementation would parse key_*/val_* naming convention)
        self._store.clear()
