from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes


AREA_ORDER: Tuple[str, ...] = ("V1", "V2", "V3", "V4")

NHP_AREA_LABELS: Dict[str, List[int]] = {
    "V1": [34],
    "V2": [6, 17, 84, 131, 172, 174],
    "V3": [60, 93, 123],
    "V4": [20, 39, 75],
}

HUMAN_AREA_LABELS: Dict[str, List[int]] = {
    "V1": [1],
    "V2": [2],
    "V3": [3],
    "V4": [4],
}

COORD_FRAME_VOXEL_RAS_V1 = "voxel_ras_v1"


def _orientation_code_from_affine(affine: np.ndarray) -> str:
    return "".join(str(c) for c in aff2axcodes(affine))


def _legacy_vector_map_to_ras(orientation_code: str) -> np.ndarray:
    """3x3 signed permutation matrix for v_ras = M @ v_legacy."""
    code = (orientation_code or "").upper().strip()
    if len(code) != 3:
        return np.eye(3, dtype=np.float64)

    axis_row = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
    sign = {"R": 1.0, "L": -1.0, "A": 1.0, "P": -1.0, "S": 1.0, "I": -1.0}
    m = np.zeros((3, 3), dtype=np.float64)
    for j, c in enumerate(code):
        if c not in axis_row:
            return np.eye(3, dtype=np.float64)
        m[axis_row[c], j] = sign[c]
    return m


def _load_canonical_image(path: Path):
    """Load image and canonicalize it to RAS voxel axis order."""
    img_orig = nib.load(str(path))
    orientation_original = _orientation_code_from_affine(img_orig.affine)
    img_canon = nib.as_closest_canonical(img_orig)
    return img_canon, orientation_original


def _build_area_structures(
    atlas: np.ndarray,
    area_labels: Dict[str, List[int]],
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]]:
    area_volume = np.zeros(atlas.shape, dtype=np.uint8)
    visual_areas: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    for idx, area_name in enumerate(AREA_ORDER, start=1):
        labels = area_labels.get(area_name, [])
        if labels:
            mask = np.isin(atlas, labels)
        else:
            mask = np.zeros(atlas.shape, dtype=bool)
        area_volume[mask] = idx
        coords = np.where(mask)
        visual_areas.append((area_name, coords[0], coords[1], coords[2]))

    return area_volume, visual_areas


def _human_demo_paths(repo_root: Path, subject_id: str) -> Dict[str, Path]:
    mri_root = (
        repo_root
        / "data"
        / "human"
        / "demo_subject"
        / "subjects"
        / subject_id
        / "T1w"
        / subject_id
        / "mri"
    )
    return {
        "atlas": mri_root / "inferred_varea.mgz",
        "ecc": mri_root / "inferred_eccen.mgz",
        "pol": mri_root / "inferred_angle.mgz",
    }


def _human_anatomy_paths(repo_root: Path, subject_id: str) -> Dict[str, Path]:
    demo_mri_root = (
        repo_root
        / "data"
        / "human"
        / "demo_subject"
        / "subjects"
        / subject_id
        / "T1w"
        / subject_id
        / "mri"
    )
    builtin_mri_root = (
        repo_root
        / "data"
        / "human"
        / "neuropythy_builtin"
        / "benson_winawer_2018"
        / "freesurfer_subjects"
        / subject_id
        / "mri"
    )

    # Prefer anatomy colocated with inferred_* maps; fall back to built-in FreeSurfer trees.
    roots = [demo_mri_root, builtin_mri_root]
    names = {
        "brainmask": "brainmask.mgz",
        "ribbon": "ribbon.mgz",
        "brain": "brain.mgz",
    }
    out: Dict[str, Path] = {}
    for key, fname in names.items():
        chosen = None
        for root in roots:
            cand = root / fname
            if cand.exists():
                chosen = cand
                break
        out[key] = chosen if chosen is not None else (demo_mri_root / fname)
    return out


def is_human_demo_available(repo_root: Path, subject_id: str = "100610") -> bool:
    paths = _human_demo_paths(repo_root, subject_id)
    return all(p.exists() for p in paths.values())


def available_datasets(repo_root: Path, subject_id: str = "100610") -> List[Tuple[str, str]]:
    datasets = [("nhp", "NHP (Macaque D99)")]
    if is_human_demo_available(repo_root, subject_id):
        datasets.append(("human_demo", f"Human Demo ({subject_id})"))
    return datasets


def _load_nhp_bundle(repo_root: Path) -> Dict[str, object]:
    data_root = repo_root / "data" / "nhp"
    atlas_file = data_root / "atlas" / "D99_in_Spike_iso.nii.gz"
    prf_root = data_root / "prf_maps"

    atlas_img, orientation_original = _load_canonical_image(atlas_file)
    legacy_vector_map = _legacy_vector_map_to_ras(orientation_original)
    atlas = atlas_img.get_fdata().astype(np.int16)
    area_volume, visual_areas = _build_area_structures(atlas, NHP_AREA_LABELS)

    def _load_sz(path):
        """Load a pRF size (sigma) NIfTI map, masking sentinel/invalid values as NaN."""
        img, _ = _load_canonical_image(path)
        data = img.get_fdata().astype("float32")
        # Background voxels are stored as -99 or other negative sentinels; zero is also not a valid sigma
        data[data <= 0] = np.nan
        return data

    def _load_map(path):
        img, _ = _load_canonical_image(path)
        return img.get_fdata().astype("float32")

    prf_maps = {
        "Monkey D": {
            "ecc": _load_map(prf_root / "danny" / "ECC_Danny_inSpike.nii.gz"),
            "pol": _load_map(prf_root / "danny" / "POL_Danny_inSpike.nii.gz"),
            "r2":  _load_map(prf_root / "danny" / "R2_Danny_inSpike.nii.gz"),
            "sz":  _load_sz(prf_root / "danny" / "SZ_Danny_inSpike.nii.gz"),
        },
        "Monkey E": {
            "ecc": _load_map(prf_root / "eddy" / "ECC_Eddy_inSpike.nii.gz"),
            "pol": _load_map(prf_root / "eddy" / "POL_Eddy_inSpike.nii.gz"),
            "r2":  _load_map(prf_root / "eddy" / "R2_Eddy_inSpike.nii.gz"),
            "sz":  _load_sz(prf_root / "eddy" / "SZ_Eddy_inSpike.nii.gz"),
        },
    }

    return {
        "dataset_id": "nhp",
        "atlas": atlas,
        "brain_data": atlas_img.get_fdata(),
        "mm_per_pixel": atlas_img.header.get_zooms(),
        "area_volume": area_volume,
        "visual_areas": visual_areas,
        "prf_maps": prf_maps,
        "default_prf_source": "Monkey D",
        "supports_nn_predictions": True,
        "invert_fiducial_z": True,
        "coord_frame": COORD_FRAME_VOXEL_RAS_V1,
        "orientation_original": orientation_original,
        "orientation_canonical": "RAS",
        "legacy_vector_map": legacy_vector_map.tolist(),
        # 0° = right horizontal meridian, counter-clockwise (standard math convention)
        "pol_convention": "standard",
        # Virtual anatomy source available in all NHP loads.
        "anatomy_paths": {},
        "anatomy_candidates": ["atlas_nonzero"],
    }


def _load_human_demo_bundle(repo_root: Path, subject_id: str) -> Dict[str, object]:
    paths = _human_demo_paths(repo_root, subject_id)
    if not all(p.exists() for p in paths.values()):
        missing = [str(p) for p in paths.values() if not p.exists()]
        raise FileNotFoundError(
            "Human demo dataset is incomplete. Missing files:\n" + "\n".join(missing)
        )

    area_img, orientation_original = _load_canonical_image(paths["atlas"])
    legacy_vector_map = _legacy_vector_map_to_ras(orientation_original)
    area_map = area_img.get_fdata()
    atlas = area_map.astype(np.int16)
    area_volume, visual_areas = _build_area_structures(atlas, HUMAN_AREA_LABELS)

    ecc_img, _ = _load_canonical_image(paths["ecc"])
    ecc_map = ecc_img.get_fdata().astype("float32")
    pol_path = paths["pol"]
    viewer_sidecar = pol_path.parent / "inferred_angle_viewer_contralateral.mgz"
    # If a subject-specific viewer sidecar exists, prefer it to avoid
    # recomputing RH conversion from affine heuristics in the loader.
    use_preconverted_polar = viewer_sidecar.exists()
    if use_preconverted_polar:
        pol_path = viewer_sidecar
    pol_img, _ = _load_canonical_image(pol_path)
    polar_map = pol_img.get_fdata().astype("float32")

    # The inferred angle map may be radians; convert to degrees.
    finite = polar_map[np.isfinite(polar_map)]
    if finite.size > 0 and np.nanmax(np.abs(finite)) <= (2 * np.pi + 0.5):
        polar_map = np.degrees(polar_map)

    if not use_preconverted_polar:
        # neuropythy stores unsigned angles (0→180°) for BOTH hemispheres:
        #   LH voxels: 0→180 = UVM→RHM→LVM  (right visual field)
        #   RH voxels: 0→180 = UVM→LHM→LVM  (left visual field)
        # Negate RH angles so that after mod360 they become 180→360°, which
        # the neuropythy sin/cos display transform correctly maps to the left VF.
        # Hemisphere is determined by the sign of the x-RAS coordinate derived
        # from the image affine (positive x-RAS = right hemisphere).
        affine = pol_img.affine
        # x_ras ≈ affine[0,0]*i + affine[0,3]  (dominant term for a near-RAS volume)
        i_indices = np.arange(polar_map.shape[0])
        x_ras = affine[0, 0] * i_indices + affine[0, 3]
        rh_slices = i_indices[x_ras > 0]
        polar_map[rh_slices, :, :] = -polar_map[rh_slices, :, :]

    # Normalise to [0, 360): negative RH angles e.g. -90° → 270°
    polar_map = np.mod(polar_map, 360.0).astype("float32")

    valid = (np.abs(ecc_map) > 0) & np.isfinite(ecc_map) & np.isfinite(polar_map)
    # Keep default threshold behavior permissive, similar to NN maps in NHP mode.
    r2_map = np.where(valid, 1e6, -1.0).astype("float32")

    # Load pRF size (sigma) if available; apply same radians→degrees scaling as eccen.
    sz_path = paths["pol"].parent / "inferred_sigma.mgz"
    sz_map = None
    if sz_path.exists():
        sz_img, _ = _load_canonical_image(sz_path)
        sz_map = sz_img.get_fdata().astype("float32")
        sz_finite = sz_map[np.isfinite(sz_map)]
        if sz_finite.size > 0 and np.nanmax(sz_finite) <= (2 * np.pi + 0.5):
            sz_map = np.degrees(sz_map).astype("float32")
        sz_map = np.where(sz_map > 0, sz_map, np.nan).astype("float32")

    prf_entry: Dict[str, object] = {"ecc": ecc_map, "pol": polar_map, "r2": r2_map}
    if sz_map is not None:
        prf_entry["sz"] = sz_map

    prf_maps = {
        f"Human Demo ({subject_id})": prf_entry,
    }

    anatomy_all_paths = _human_anatomy_paths(repo_root, subject_id)
    anatomy_paths = {k: p for k, p in anatomy_all_paths.items() if p.exists()}
    anatomy_candidates = [k for k in ("brainmask", "ribbon", "brain") if k in anatomy_paths]
    anatomy_candidates.append("atlas_nonzero")

    return {
        "dataset_id": "human_demo",
        "atlas": atlas,
        "brain_data": atlas,
        "mm_per_pixel": area_img.header.get_zooms(),
        "area_volume": area_volume,
        "visual_areas": visual_areas,
        "prf_maps": prf_maps,
        "default_prf_source": f"Human Demo ({subject_id})",
        "supports_nn_predictions": True,
        "invert_fiducial_z": False,
        "coord_frame": COORD_FRAME_VOXEL_RAS_V1,
        "orientation_original": orientation_original,
        "orientation_canonical": "RAS",
        "legacy_vector_map": legacy_vector_map.tolist(),
        "polar_map_path": str(pol_path),
        "polar_map_preconverted": bool(use_preconverted_polar),
        # 0° = upper vertical meridian (UVM), clockwise positive (neuropythy convention)
        # +90° = right horizontal meridian, ±180° = lower vertical meridian
        "pol_convention": "neuropythy",
        # Optional anatomy context layers for visualization.
        "anatomy_paths": anatomy_paths,
        "anatomy_candidates": anatomy_candidates,
    }


def load_dataset_bundle(
    repo_root: Path,
    dataset_id: str,
    subject_id: str = "100610",
) -> Dict[str, object]:
    if dataset_id == "nhp":
        return _load_nhp_bundle(repo_root)
    if dataset_id == "human_demo":
        return _load_human_demo_bundle(repo_root, subject_id=subject_id)
    raise ValueError(f"Unsupported dataset id: {dataset_id!r}")


def synthetic_fiducials_from_visual_areas(
    visual_areas: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    num_combs: int = 16,
    spacing_vox: float = 2.0,
    depth_span_vox: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    v1_entry = next((entry for entry in visual_areas if entry[0] == "V1"), None)
    if v1_entry is None or len(v1_entry[1]) == 0:
        fallback = next((entry for entry in visual_areas if len(entry[1]) > 0), None)
        if fallback is None:
            center = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        else:
            center = np.array(
                [
                    float(np.median(fallback[1])),
                    float(np.median(fallback[2])),
                    float(np.median(fallback[3])),
                ],
                dtype=np.float32,
            )
    else:
        center = np.array(
            [
                float(np.median(v1_entry[1])),
                float(np.median(v1_entry[2])),
                float(np.median(v1_entry[3])),
            ],
            dtype=np.float32,
        )

    offsets = (np.arange(num_combs, dtype=np.float32) - (num_combs - 1) / 2.0) * spacing_vox
    anterior = np.column_stack(
        [
            center[0] + offsets,
            np.full(num_combs, center[1], dtype=np.float32),
            np.full(num_combs, center[2], dtype=np.float32),
        ]
    ).astype(np.float32)
    posterior = np.column_stack(
        [
            center[0] + offsets,
            np.full(num_combs, center[1] + depth_span_vox, dtype=np.float32),
            np.full(num_combs, center[2] + depth_span_vox, dtype=np.float32),
        ]
    ).astype(np.float32)
    return anterior, posterior
