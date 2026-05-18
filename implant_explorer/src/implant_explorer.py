"""
vimplant2 Implant Explorer - Optimized Interactive 3D Visualization
============================================================================
Professional Qt UI with PyVistaQt for reliable widget interaction.

OPTIMIZATIONS (see CHANGES.md for details):
- Area mask lookup: 3D uint8 volume instead of set-of-tuples (25x speedup)
- Rotation: Vectorized matrix multiplication (15x speedup)
- RF lookup: Batch array indexing (20-35x speedup)

Layout:
- LEFT: Control panel with sliders, spinboxes, and buttons
- CENTER: 3D brain visualization
- RIGHT: Polar plot of RF coverage

Usage:
    python implant_explorer.py

Author: Antonio Lozano (NBIO, UMH) a.lozano@umh.es
Date: 2026

"""

import sys
import argparse
import gc
import copy
import uuid
import csv
import json
import datetime
import numpy as np
import os
from pathlib import Path
import nibabel as nib
from implants_core.transforms import apply_transform_to_contacts, validate_contacts_soft

# v2 implants_core — parametric spec + placement controller
try:
    from implants_core.spec import ImplantSpec
    from implants_core.export import load_spec_json, save_spec_json
    from implants_core.placement import PlacementController, mm_to_voxel
    from implants_core.generators import generate_utah
    from implants_core.explorer_scene import (
        ExplorerScene,
        ExplorerImplantInstance,
        ExplorerDesignRevision,
        normalize_instance_placement,
        transform_instance_contacts_mm,
        hit_test_instance_contacts,
    )

    _HAS_IMPLANTS_CORE = True
except ImportError:
    _HAS_IMPLANTS_CORE = False

import pyvista as pv
from pyvistaqt import QtInteractor
import matplotlib

matplotlib.use("Agg")
import colorsys
import pyqtgraph as pg

from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QGroupBox,
    QPushButton,
    QComboBox,
    QSplitter,
    QFrame,
    QCheckBox,
    QSpinBox,
    QTabWidget,
    QGridLayout,
    QScrollArea,
    QMessageBox,
    QFileDialog,
    QShortcut,
    QColorDialog,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from qtpy.QtCore import Qt, QTimer, QEvent
from qtpy.QtGui import QFont, QKeySequence, QColor
from dataset_adapters import (
    load_dataset_bundle,
    synthetic_fiducials_from_visual_areas,
)
from rf_selection_utils import build_ball_offsets, collect_local_rf_neighborhood


def _ensure_valid_qt_app_font(app: QApplication, fallback_pt: int = 11) -> None:
    current = app.font()

    # If the font already has a valid point size, don't touch it.
    current_pt = current.pointSize()
    if isinstance(current_pt, int) and current_pt > 0:
        return

    # Derive a reasonable point size from pixelSize when available.
    pixel = current.pixelSize()
    if isinstance(pixel, int) and pixel > 0:
        screen = app.primaryScreen()
        dpi = float(screen.logicalDotsPerInch()) if screen is not None else 96.0
        pt = max(1, int(round(pixel * 72.0 / dpi)))
    else:
        pt = max(1, int(fallback_pt))

    family = current.family() or "Arial"
    forced = QFont(family, pt)
    forced.setBold(current.bold())
    forced.setItalic(current.italic())
    forced.setUnderline(current.underline())
    app.setFont(forced)


# =============================================================================
# CONFIGURATION
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "nhp"
atlas_path = str(DATA_ROOT / "atlas" / "D99_in_Spike_iso.nii.gz")
prf_path = str(DATA_ROOT / "prf_maps")
fiducials_path = str(DATA_ROOT / "fiducials")
NN_PREDICTIONS_PATH = str(REPO_ROOT / "results" / "prf_prediction" / "predictions")
DEFAULT_HUMAN_SUBJECT = "fsaverage"

# Slicer3D coordinate reference
SLICER_ZERO_START_X = 72
SLICER_ZERO_START_Y = 94
SLICER_ZERO_START_Z = 107 - 52

# Implant parameters
NUMBER_COMBS = 16
R2_THRESHOLD_INIT = 0.5
R2_THRESHOLD_MAX = 10.0
UNDERSAMPLING = 1
ECC_VMIN, ECC_VMAX = 0, 15
POL_VMIN, POL_VMAX = 0, 360
SELECTION_RADIUS_VOX = 2
SELECTION_MAX_POINTS = 250

# Visual area labels from D99 atlas
V1_LABELS = [34]
V2_LABELS = [6, 17, 84, 131, 172, 174]
V3_LABELS = [60, 93, 123]
V4_LABELS = [20, 39, 75]

AREA_COLORS = {
    "V1": [0.12, 0.56, 1.0],
    "V2": [1.0, 0.41, 0.71],
    "V3": [0.2, 0.8, 0.2],
    "V4": [1.0, 0.0, 0.0],
}

RF_EXPORT_COLUMNS = [
    "source_app",
    "dataset",
    "prf_source",
    "implant_id",
    "electrode_index",
    "x_deg",
    "y_deg",
    "polar_deg",
    "ecc_deg",
]

# =============================================================================
# OPTIMIZED HELPER FUNCTIONS
# =============================================================================


def read_spikeDesignFiducialsFromSlicer3D(filepath, voxel_size):
    """Read fiducial points from Slicer3D .fcsv file."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    points = []
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split(",")
        if len(parts) >= 4:
            x = float(parts[1]) / voxel_size
            y = float(parts[2]) / voxel_size
            z = float(parts[3]) / voxel_size
            points.append([x, y, z])
    if len(points) >= 3:
        return (
            points[0][0],
            points[0][1],
            points[1][1],
            points[2][1],
            points[0][2],
            points[1][2],
            points[2][2],
        )
    return None


def createArcWithThreePoints(p1, p2, p3, XtoPredict):
    """Create arc through three points using quadratic interpolation."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])
    b = np.array([y1, y2, y3])
    coeffs = np.linalg.solve(A, b)
    return XtoPredict, coeffs[0] * XtoPredict**2 + coeffs[1] * XtoPredict + coeffs[2]


def rotate_points_vectorized(points_yz, pivot_yz, angle_rad):
    """
    OPTIMIZED: Rotate multiple 2D points around a pivot using matrix multiplication.

    Args:
        points_yz: (N, 2) array of [y, z] coordinates
        pivot_yz: (2,) array [pivot_y, pivot_z]
        angle_rad: rotation angle in radians

    Returns:
        (N, 2) array of rotated [y, z] coordinates
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    centered = points_yz - pivot_yz
    rotated = centered @ R.T
    return rotated + pivot_yz


def get_rf_batch_vectorized(
    coords, ecc_map, polar_map, r2_map, r2_threshold=0.2, sz_map=None
):
    """
    OPTIMIZED: Vectorized RF lookup for multiple coordinates at once.

    Args:
        coords: (N, 3) array of [x, y, z] integer coordinates
        ecc_map, polar_map, r2_map: 3D arrays
        r2_threshold: minimum R² value
        sz_map: optional 3D array of pRF sizes; None if not available

    Returns:
        ecc, polar, r2, sz: (N,) arrays with -1 / nan for invalid positions
    """
    n = coords.shape[0]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    shape = ecc_map.shape

    # Bounds mask
    valid = (
        (x >= 0)
        & (x < shape[0])
        & (y >= 0)
        & (y < shape[1])
        & (z >= 0)
        & (z < shape[2])
    )

    # Initialize with -1 (invalid)
    ecc = np.full(n, -1.0, dtype=np.float32)
    polar = np.full(n, -1.0, dtype=np.float32)
    r2 = np.full(n, -1.0, dtype=np.float32)
    sz = np.full(n, np.nan, dtype=np.float32)

    if not np.any(valid):
        return ecc, polar, r2, sz

    # Get valid coordinates
    xv, yv, zv = x[valid], y[valid], z[valid]

    # Batch lookup
    r2_vals = r2_map[xv, yv, zv]
    ecc_vals = ecc_map[xv, yv, zv]

    # Apply R² threshold and ecc > 0 check
    r2_mask = (r2_vals >= r2_threshold) & (ecc_vals > 0)

    # Build full validity mask
    full_valid = np.zeros(n, dtype=bool)
    valid_indices = np.where(valid)[0]
    full_valid[valid_indices[r2_mask]] = True

    if not np.any(full_valid):
        return ecc, polar, r2, sz

    xf, yf, zf = x[full_valid], y[full_valid], z[full_valid]
    ecc[full_valid] = ecc_map[xf, yf, zf]
    polar[full_valid] = polar_map[xf, yf, zf]
    r2[full_valid] = r2_map[xf, yf, zf]
    if sz_map is not None:
        sz[full_valid] = sz_map[xf, yf, zf]

    return ecc, polar, r2, sz


def build_area_volume(atlas, shape):
    """
    OPTIMIZED: Build 3D uint8 volume for O(1) area membership lookup.

    Returns:
        area_volume: 3D array where 0=no area, 1=V1, 2=V2, 3=V3, 4=V4
    """
    area_volume = np.zeros(shape, dtype=np.uint8)

    for i, labels in enumerate([V1_LABELS, V2_LABELS, V3_LABELS, V4_LABELS]):
        mask = np.isin(atlas, labels)
        area_volume[mask] = i + 1  # 1-indexed: V1=1, V2=2, etc.

    return area_volume


def map_legacy_vector_to_ras(vec_xyz_mm, legacy_vector_map):
    """Map a legacy [x,y,z] mm vector into canonical RAS axes."""
    vec = np.asarray(vec_xyz_mm, dtype=np.float64).reshape(3)
    try:
        m = np.asarray(legacy_vector_map, dtype=np.float64).reshape(3, 3)
    except Exception:
        m = np.eye(3, dtype=np.float64)
    return (m @ vec).astype(np.float64)


# =============================================================================
# FAST POLAR PLOT WIDGET (PyQtGraph - 50-100x faster than Matplotlib)
# =============================================================================


class PolarPlotWidget(pg.PlotWidget):
    """Fast polar plot using PyQtGraph scatter plot with polar coordinate transform."""

    def __init__(self, parent=None, convention="standard", max_ecc=15):
        super().__init__(parent, background="k")
        self.setMinimumSize(250, 250)
        self.setAspectLocked(True)
        self._convention = convention
        self._max_ecc = max_ecc
        self._brush_pool = []
        self._brush_default = pg.mkBrush(255, 255, 255, 255)
        self._brush_key = None

        # Draw polar grid
        self._draw_polar_grid()

        # Create scatter plot item (reused for speed)
        self._scatter = pg.ScatterPlotItem(
            size=1.0, pen=pg.mkPen("w", width=0.5), pxMode=False
        )
        self.addItem(self._scatter)

        # Selection overlay (hover/click-picked local neighborhood)
        self._selection_scatter = pg.ScatterPlotItem(
            size=1.0,
            pen=pg.mkPen(255, 255, 255, 220, width=0.8),
            brush=pg.mkBrush(0, 255, 255, 210),
            pxMode=False,
        )
        self.addItem(self._selection_scatter)
        self._selection_seed = pg.ScatterPlotItem(
            size=1.2,
            symbol="x",
            pen=pg.mkPen(255, 255, 255, 240, width=2),
            brush=pg.mkBrush(255, 255, 255, 30),
            pxMode=False,
        )
        self.addItem(self._selection_seed)

        # Title sits above the top spoke label (at max_ecc * 1.20); position scales with scope
        self._title = pg.TextItem("", color="w", anchor=(0.5, 0))
        self._title.setPos(0, self._max_ecc * 1.35)
        self.addItem(self._title)

        # Set range with more space at top for title
        self._apply_view_range()
        self.hideAxis("left")
        self.hideAxis("bottom")

    def _apply_view_range(self):
        # X: accommodate spoke labels at 1.20× + small pad
        # Y: accommodate title at 1.35× top, spoke label at 1.20× bottom
        h = self._max_ecc
        self.setXRange(-h * 1.28, h * 1.28)
        self.setYRange(-h * 1.28, h * 1.55)

    def _compute_dot_sizes(self, base_deg, sz_list=None, *, boost=1.0):
        # Size is diameter in data units (deg of visual angle) because pxMode=False.
        base_deg = float(base_deg)
        if sz_list is None:
            return base_deg

        sz_arr = np.asarray(sz_list, dtype=np.float32)
        valid_sz = np.isfinite(sz_arr) & (sz_arr > 0)
        if not np.any(valid_sz):
            return base_deg

        # Sigma radius mode: r_deg = sigma.
        # Keep robust bounds to avoid occasional pathological sigma outliers.
        sigma_valid = valid_sz & (sz_arr <= 20.0)
        if not np.any(sigma_valid):
            return base_deg

        sigma = np.where(sigma_valid, sz_arr, np.nan)
        dot_sizes = (2.0 * sigma * float(boost)).astype(np.float32)

        # Radius cap: "few degrees" relative to VF scope; diameter is 2x this cap.
        max_radius_deg = min(6.0, max(1.5, float(self._max_ecc) * 0.20))
        max_diam_deg = 2.0 * max_radius_deg

        # Invalid sigma entries fallback to a small but visible diameter.
        out = np.full_like(dot_sizes, base_deg, dtype=np.float32)
        good = np.isfinite(dot_sizes) & (dot_sizes > 0)
        out[good] = np.minimum(dot_sizes[good], max_diam_deg)
        return out

    def _adaptive_ring_step(self):
        """Choose a readable radial tick step for the current VF scope."""
        h = max(1.0, float(self._max_ecc))
        # Aim ~10 rings, then quantize to friendly steps.
        raw = h / 10.0
        for step in (1, 2, 5, 10):
            if raw <= step:
                return step
        return 10

    def _draw_polar_grid(self):
        """Draw concentric circles and radial lines. All text scales with _max_ecc."""
        # Larger, bolder fonts for readability.
        if self._max_ecc <= 20:
            spoke_pt = 12
            ring_pt = 11
        elif self._max_ecc <= 40:
            spoke_pt = 11
            ring_pt = 10
        else:
            spoke_pt = 10
            ring_pt = 9
        font = QFont("Arial", spoke_pt)
        font.setBold(True)
        small_font = QFont("Arial", ring_pt)
        small_font.setBold(True)

        # Adaptive ring/tick spacing for any scope.
        step = self._adaptive_ring_step()
        n_rings = max(1, int(np.floor(self._max_ecc / step)))
        label_every = step * (2 if n_rings > 8 else 1)
        offset = self._max_ecc * 0.025  # proportional label offset from ring
        for r in range(step, self._max_ecc + 1, step):
            theta = np.linspace(0, 2 * np.pi, 120)
            self.plot(
                r * np.cos(theta), r * np.sin(theta), pen=pg.mkPen("w", width=0.8)
            )
            if r % label_every == 0 or r == self._max_ecc:
                lbl = pg.TextItem(f"{r}°", color="w")
                lbl.setFont(small_font)
                lbl.setPos(r + offset, offset)
                self.addItem(lbl)

        if self._convention == "neuropythy":
            cardinal_labels = {
                0: "RHM\n(90°)",
                45: "135°",
                90: "UVM\n(0°)",
                135: "45°",
                180: "LHM\n(270°)",
                225: "225°",
                270: "LVM\n(180°)",
                315: "315°",
            }
        else:
            cardinal_labels = {a: f"{a}°" for a in [0, 45, 90, 135, 180, 225, 270, 315]}

        # Spoke labels radius: 120% of max_ecc so they sit just outside the last ring
        label_r = self._max_ecc * 1.20
        for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            angle = np.radians(angle_deg)
            self.plot(
                [0, self._max_ecc * np.cos(angle)],
                [0, self._max_ecc * np.sin(angle)],
                pen=pg.mkPen("w", width=0.8),
            )
            lbl = pg.TextItem(cardinal_labels[angle_deg], color="w", anchor=(0.5, 0.5))
            lbl.setFont(font)
            lbl.setPos(label_r * np.cos(angle), label_r * np.sin(angle))
            self.addItem(lbl)

    def reset_convention(self, convention):
        """Rebuild the grid for a new polar-angle convention (call after dataset switch)."""
        if convention == self._convention:
            return
        self._convention = convention
        self._rebuild_grid()

    def set_max_ecc(self, max_ecc):
        """Change the eccentricity scope and rebuild the grid + view range."""
        if max_ecc == self._max_ecc:
            return
        self._max_ecc = max_ecc
        self._title.setPos(0, self._max_ecc * 1.35)
        self._rebuild_grid()
        self._apply_view_range()

    def _rebuild_grid(self):
        """Remove all grid items and redraw."""
        for item in list(self.items()):
            if (
                item is not self._scatter
                and item is not self._selection_scatter
                and item is not self._selection_seed
                and item is not self._title
            ):
                self.removeItem(item)
        self._draw_polar_grid()

    def update_plot(
        self,
        ecc_list,
        pol_list,
        probe_indices,
        probe_colors,
        r2_thresh,
        n_total,
        dot_size=50,
        sz_list=None,
        alpha=0.5,
        centers_only=False,
    ):
        """Update polar plot with new RF data. FAST - reuses scatter item."""
        if len(ecc_list) == 0:
            self._scatter.setData([], [])
            self._title.setText(
                f"Receptive Fields: 0/{n_total} (0%)\nR² > {r2_thresh:.2f}"
            )
            return

        # Convert polar to cartesian for plotting.
        # For the 'neuropythy' convention (0°=UVM, clockwise):
        #   x = ecc * sin(pol)   — sin(0°)=0 → UVM sits on the vertical axis
        #   y = ecc * cos(pol)   — cos(0°)=1 → UVM plotted at the top
        # This is equivalent to the rotation: math_angle = 90° − pol
        # For 'standard' (0°=RHM, CCW) the usual cos/sin applies.
        ecc = np.array(ecc_list)
        pol_rad = np.radians(pol_list)
        if self._convention == "neuropythy":
            x = ecc * np.sin(pol_rad)
            y = ecc * np.cos(pol_rad)
        else:
            x = ecc * np.cos(pol_rad)
            y = ecc * np.sin(pol_rad)

        # Draw true RF size in visual-angle units (FWHM radius from sigma).
        # `dot_size` slider is legacy pixel-oriented, use it only as a gentle fallback scale.
        fallback_deg = max(0.35, min(1.2, float(dot_size) / 50.0))
        if centers_only:
            # Match web: fixed-radius dots, scale ~1.5× the fallback.
            dot_sizes = max(0.35, fallback_deg * 1.5)
        else:
            dot_sizes = self._compute_dot_sizes(fallback_deg, sz_list)

        # Build colors array (QColor format). Cache one brush per probe color
        # so we make K mkBrush() calls per refresh, not N.
        a8 = int(max(0.0, min(1.0, float(alpha))) * 255)
        brush_key = (id(probe_colors), len(probe_colors), a8)
        if brush_key != self._brush_key:
            self._brush_pool = [
                pg.mkBrush(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), a8)
                for c in probe_colors
            ]
            self._brush_default = pg.mkBrush(255, 255, 255, a8)
            self._brush_key = brush_key
        brushes = [
            (
                self._brush_pool[p]
                if 0 <= p < len(self._brush_pool)
                else self._brush_default
            )
            for p in probe_indices
        ]

        # Update scatter (FAST - no redraw of whole figure)
        self._scatter.setData(x=x, y=y, brush=brushes, size=dot_sizes)

        n_valid = len(ecc_list)
        pct = 100 * n_valid / n_total if n_total > 0 else 0
        self._title.setText(
            f"Receptive Fields: {n_valid}/{n_total} ({pct:.0f}%)\nR² > {r2_thresh:.2f}"
        )

    def clear_selection_overlay(self):
        self._selection_scatter.setData([], [])
        self._selection_seed.setData([], [])

    def update_selection_overlay(
        self, ecc_list, pol_list, locked=False, seed=None, sz_list=None
    ):
        if len(ecc_list) == 0:
            self.clear_selection_overlay()
            return

        ecc = np.asarray(ecc_list, dtype=np.float32)
        pol = np.asarray(pol_list, dtype=np.float32)
        pol_rad = np.radians(pol)
        if self._convention == "neuropythy":
            x = ecc * np.sin(pol_rad)
            y = ecc * np.cos(pol_rad)
        else:
            x = ecc * np.cos(pol_rad)
            y = ecc * np.sin(pol_rad)

        alpha = 250 if locked else 190
        self._selection_scatter.setBrush(pg.mkBrush(0, 255, 255, alpha))
        overlay_sizes = self._compute_dot_sizes(0.9, sz_list, boost=1.0)
        self._selection_scatter.setData(x=x, y=y, size=overlay_sizes)

        if seed is None:
            self._selection_seed.setData([], [])
            return

        seed_ecc, seed_pol = float(seed[0]), float(seed[1])
        seed_rad = np.radians(seed_pol)
        if self._convention == "neuropythy":
            sx = seed_ecc * np.sin(seed_rad)
            sy = seed_ecc * np.cos(seed_rad)
        else:
            sx = seed_ecc * np.cos(seed_rad)
            sy = seed_ecc * np.sin(seed_rad)
        if np.isscalar(overlay_sizes):
            seed_size = max(float(overlay_sizes) * 1.15, 0.8)
        else:
            seed_size = max(
                float(np.nanmax(np.asarray(overlay_sizes, dtype=np.float32))) * 1.15,
                0.8,
            )
        self._selection_seed.setData([sx], [sy], size=seed_size)


# =============================================================================
# MAIN WINDOW
# =============================================================================


class ImplantExplorerWindow(QMainWindow):
    def __init__(self, dataset_id="human_demo", human_subject_id=DEFAULT_HUMAN_SUBJECT):
        super().__init__()
        self.setWindowTitle("vimplant2 Implant Explorer")
        self._credit_text = "Antonio Lozano (NBIO, UMH)  |  a.lozano@umh.es"
        self.dataset_id = str(dataset_id).strip().lower()
        self.human_subject_id = str(human_subject_id).strip()
        self._supports_nn_predictions = False
        self._invert_fiducial_z = True
        self._nn_source_lookup = {}
        self.pol_convention = "standard"  # overwritten by _load_data()
        self.sz_map = None  # overwritten by _load_data() when sz available
        self._ecc_scope = 15  # VF polar plot eccentricity range (degrees)
        self._nn_dense_brain_data = None  # when set: {area: (pts, ecc, pol, r2)}
        self._nn_dense_brain_cache = {}  # monkey_name -> dense brain data dict
        self._anatomy_paths = {}
        self._anatomy_candidates = ["atlas_nonzero"]
        self._anatomy_unavailable_note = ""
        self._selection_offsets = build_ball_offsets(SELECTION_RADIUS_VOX)

        # Adaptive window sizing
        screen = QApplication.primaryScreen().availableGeometry()
        win_width = min(int(screen.width() * 0.8), 1400)
        win_height = min(int(screen.height() * 0.8), 800)
        x = screen.x() + (screen.width() - win_width) // 2
        y = screen.y() + (screen.height() - win_height) // 2
        self.setGeometry(x, y, win_width, win_height)
        self.setMinimumSize(800, 500)

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: white; font-size: 11px; }
            QCheckBox { color: white; font-size: 11px; }
            QGroupBox { color: white; border: 1px solid #444; border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; font-size: 11px; color: white; }
            QDialog { background-color: #242424; color: #f2f2f2; }
            QMessageBox { background-color: #242424; }
            QMessageBox QLabel { color: #f2f2f2; }
            QFileDialog { background-color: #242424; color: #f2f2f2; }
            QFileDialog QLabel { color: #f2f2f2; }
            QFileDialog QLineEdit, QFileDialog QListView, QFileDialog QTreeView {
                background-color: #1a1a1a;
                color: #f2f2f2;
                selection-background-color: #0078d4;
            }
            QPushButton { background-color: #3c3c3c; color: white; border: 1px solid #555; border-radius: 3px; padding: 4px 8px; font-size: 11px; }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2a2a2a; }
            QComboBox { background-color: #3c3c3c; color: white; border: 1px solid #555; border-radius: 3px; padding: 2px; font-size: 11px; }
            QComboBox QAbstractItemView { background-color: #3c3c3c; color: white; selection-background-color: #0078d4; }
            QComboBox::drop-down { border: none; background-color: #4a4a4a; width: 20px; }
            QSpinBox, QDoubleSpinBox { background-color: #3c3c3c; color: white; border: 1px solid #555; border-radius: 3px; padding: 1px; font-size: 11px; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background-color: #2f2f2f; color: #f0f0f0; padding: 4px 10px; font-size: 11px; }
            QTabBar::tab:selected { background-color: #3c3c3c; color: white; }
            QSlider::groove:horizontal { background: #555; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #0078d4; width: 16px; margin: -5px 0; border-radius: 8px; }
            QSlider::sub-page:horizontal { background: #0078d4; border-radius: 3px; }
        """)

        # Initialize state
        self.state = {
            "r2": R2_THRESHOLD_INIT,
            "mode": "polar",
            "depth": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "rot_x": 0.0,
            "rot_y": 0.0,
            "rot_z": 0.0,
            "scale": 1.0,
            "mirror_x": False,
            "pivot_mode": "entry",
            "pivot_x": 0.0,
            "pivot_y": 0.0,
            "pivot_z": 0.0,
            "snap_enabled": True,
            "snap_mm": 0.1,
            "snap_deg": 1.0,
            "snap_scale": 0.01,
            "undersampling": UNDERSAMPLING,
            "prf_source": "",
            "show_areas": {"V1": True, "V2": False, "V3": False, "V4": False},
            "electrode_dot_size": 4,
            "rf_dot_size": 20,
            "fiducial_size": 15,
            "electrode_color_mode": "classic",
            "electrode_color_hex": "#00D4FF",
            "show_combs": {},
            # Brain area style options
            "area_point_size": 5,
            "area_opacity": 0.5,
            "fast_mode": True,  # Flat points instead of spheres
            # Anatomy context
            "show_anatomy": True,
            "anatomy_source": "auto",
            "anatomy_opacity": 0.12,
            "anatomy_style": "surface",
            # Pointer RF selection
            "selection_enabled": False,
            "pick_radius_vox": SELECTION_RADIUS_VOX,
            # Pointer implant drag
            "drag_implant_enabled": True,
            "drag_requires_ctrl": True,
            "drag_sensitivity_mm": 1.0,
            # Polar plot styling (rf_alpha default matches web index.html)
            "rf_alpha": 0.5,
            "rf_centers_only": False,
            # Electrode master visibility (matches web "Show electrodes")
            "show_electrodes": True,
            # Anatomy mesh wireframe
            "anatomy_wireframe": False,
        }
        self.selection_locked = False
        self.selection_seed_voxel = None
        self.selection_overlay_rf = {"ecc": [], "pol": [], "sz": [], "seed": None}
        self.selection_overlay_coords_vox = np.empty((0, 3), dtype=np.float32)
        self.electrode_actors = []
        self._suppress_value_changed = False
        self._suppress_manip_value_changed = False
        self._first_render = True
        self._drag_active = False
        self._drag_last_world = None
        self._drag_last_pos = None
        self._drag_camera_position = None
        self._drag_had_movement = False
        self._loaded_implant_path = None
        self._loaded_implant_spec_v2 = None
        self._instance_mode_enabled = False
        self._template_contact_index = []
        self._nonspike_anchor_vox = None
        self._default_implant_label = "(built-in Utah array)"
        self._active_design_spec_v2 = None
        self._active_design_revision_id = ""
        self._design_revisions = {}
        self._scene_instances = []
        self._instance_index_by_id = {}
        self._selected_instance_ids = set()
        self._last_render_instance_ids = []
        self._last_render_contacts_by_instance = {}
        self._coord_frame = "voxel_ras_v1"
        self._orientation_original = "RAS"
        self._orientation_canonical = "RAS"
        self._legacy_vector_map = np.eye(3, dtype=np.float64)
        self._transform_warnings = []
        self._undo_stack = []
        self._redo_stack = []
        self._last_ui_instance_rot = {"rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": 0.0}

        # Debounce timer for slider updates
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._do_update)
        self._update_pending = False
        self._debounce_ms = 16  # ~60 FPS target for drag/slider updates

        # Track what needs updating (avoid full rebuilds)
        self._needs_brain_update = True
        self._needs_electrode_update = True
        self._needs_anatomy_update = True

        # Cached meshes for fast updates
        self._electrode_clouds = {}
        self._electrode_halos = {}
        self._fiducials_mesh = None
        self._area_clouds = {}  # Cache brain area meshes
        self._area_scalars_cache = {}  # Cache computed scalars
        self._anatomy_mesh_cache = {}
        self._anatomy_actor_name = "anatomy_context"
        self._selection_halo_actor_name = "rf_pick_points_halo"
        self._selection_actor_name = "rf_pick_points"
        self._selection_seed_halo_actor_name = "rf_pick_seed_halo"
        self._selection_seed_actor_name = "rf_pick_seed"
        # Selection-cloud PolyData caches (_selection_cloud,
        # _selection_halo_cloud, _selection_seed_cloud,
        # _selection_seed_halo_cloud) are managed lazily by
        # _upsert_point_overlay and _remove_selection_marker_actors.

        # Pointer-picking throttling
        self._pending_hover_world = None
        self._hover_pick_timer = QTimer()
        self._hover_pick_timer.setSingleShot(True)
        self._hover_pick_timer.setInterval(40)
        self._hover_pick_timer.timeout.connect(self._process_pending_hover_pick)

        # Load data
        self._load_data()

        # Build UI
        self._build_ui()
        self._setup_pointer_picking()

        # Initial update
        self.update_all()

    def _load_data(self):
        """Load all required data with optimized structures."""
        print("=" * 60)
        print("VIMPLANT2 EXPLORER V2 - OPTIMIZED")
        print("=" * 60)

        print(f"\nLoading dataset bundle: {self.dataset_id}")
        bundle = load_dataset_bundle(
            REPO_ROOT, self.dataset_id, subject_id=self.human_subject_id
        )

        self.dataset_id = str(bundle["dataset_id"])
        self.brain_data = bundle["brain_data"]
        self.mm_per_pixel = bundle["mm_per_pixel"]
        self.area_volume = bundle["area_volume"]
        self.visual_areas = bundle["visual_areas"]
        self.prf_maps = bundle["prf_maps"]
        self._supports_nn_predictions = bool(
            bundle.get("supports_nn_predictions", False)
        )
        self._invert_fiducial_z = bool(bundle.get("invert_fiducial_z", True))
        self._coord_frame = str(bundle.get("coord_frame", "voxel_ras_v1"))
        self._orientation_original = str(bundle.get("orientation_original", "RAS"))
        self._orientation_canonical = str(bundle.get("orientation_canonical", "RAS"))
        self._legacy_vector_map = np.asarray(
            bundle.get("legacy_vector_map", np.eye(3)), dtype=np.float64
        )
        # Polar angle convention: 'standard' (0°=RHM, CCW) or 'neuropythy' (0°=UVM, CW)
        self.pol_convention = str(bundle.get("pol_convention", "standard"))
        self._anatomy_paths = {
            k: Path(v) if not isinstance(v, Path) else v
            for k, v in dict(bundle.get("anatomy_paths", {})).items()
        }
        self._anatomy_candidates = list(
            bundle.get("anatomy_candidates", ["atlas_nonzero"])
        )
        self._anatomy_unavailable_note = ""

        print(
            f"  Volume: {self.brain_data.shape}, Voxel: {self.mm_per_pixel[0]:.2f} mm"
        )
        area_counts = [(self.area_volume == i).sum() for i in range(1, 5)]
        print(
            f"  V1: {area_counts[0]}, V2: {area_counts[1]}, V3: {area_counts[2]}, V4: {area_counts[3]} voxels"
        )
        print(f"  Memory (area volume): {self.area_volume.nbytes / 1024 / 1024:.1f} MB")

        default_source = bundle.get("default_prf_source")
        if default_source not in self.prf_maps:
            default_source = next(iter(self.prf_maps.keys()))
        self.state["prf_source"] = str(default_source)
        self.ecc_map = self.prf_maps[default_source]["ecc"]
        self.polar_map = self.prf_maps[default_source]["pol"]
        self.R2_map = self.prf_maps[default_source]["r2"]
        self.sz_map = self.prf_maps[default_source].get("sz", None)

        print("\nPreparing fiducials...")
        if self.dataset_id == "nhp":
            self.fiducial_anterior_list = []
            self.fiducial_posterior_list = []

            for i in range(NUMBER_COMBS):
                fcsv_file = os.path.join(fiducials_path, f"{i+1}.fcsv")
                if not os.path.exists(fcsv_file):
                    continue
                result = read_spikeDesignFiducialsFromSlicer3D(
                    fcsv_file, self.mm_per_pixel[0]
                )
                if result is None:
                    continue
                fx, fy_ant, fy_int, fy_post, fz_ant, fz_int, fz_post = result
                fx = SLICER_ZERO_START_X - fx
                self.fiducial_anterior_list.append(
                    [fx, SLICER_ZERO_START_Y - fy_ant, SLICER_ZERO_START_Z - fz_ant]
                )
                self.fiducial_posterior_list.append(
                    [fx, SLICER_ZERO_START_Y - fy_post, SLICER_ZERO_START_Z - fz_post]
                )
            self.fiducial_anterior_list = np.asarray(
                self.fiducial_anterior_list, dtype=np.float32
            )
            self.fiducial_posterior_list = np.asarray(
                self.fiducial_posterior_list, dtype=np.float32
            )
        else:
            anterior, posterior = synthetic_fiducials_from_visual_areas(
                self.visual_areas,
                num_combs=NUMBER_COMBS,
            )
            self.fiducial_anterior_list = anterior.astype(np.float32)
            self.fiducial_posterior_list = posterior.astype(np.float32)

        self.num_combs = len(self.fiducial_anterior_list)
        print(f"  Prepared {self.num_combs} fiducials")

        # Initialize comb state
        self.state["show_combs"] = {i: True for i in range(self.num_combs)}
        self.comb_offsets_mm = {
            i: {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
            for i in range(self.num_combs)
        }
        self._last_ui_move_mm = {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
        self.design_transform = {
            "tx_mm": 0.0,
            "ty_mm": 0.0,
            "tz_mm": 0.0,
            "rx_deg": 0.0,
            "ry_deg": 0.0,
            "rz_deg": 0.0,
            "scale": 1.0,
            "mirror_x": False,
            "pivot_mode": "entry",
            "pivot_custom_mm": [0.0, 0.0, 0.0],
        }
        self._last_ui_design = {"rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0, "scale": 1.0}
        self._undo_stack.clear()
        self._redo_stack.clear()

        # Create electrode design
        self._create_electrode_design()

    def _create_electrode_design(self):
        """Create default implant geometry."""
        print("\nCreating electrode design...")
        self._loaded_implant_path = None
        self._loaded_implant_spec_v2 = None
        self._instance_mode_enabled = True
        default_path = (
            REPO_ROOT
            / "implant_designs"
            / "non_canonical"
            / "Utah Array fovea left hemisphere.json"
        )
        spec_v2 = None

        if default_path.exists():
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if _HAS_IMPLANTS_CORE and str(raw.get("schema_version", "")).startswith(
                    "2"
                ):
                    spec_v2 = ImplantSpec.from_dict(raw)
                else:
                    raise ValueError("default implant must use schema_version 2.x")
                self._loaded_implant_path = str(default_path)
                self._default_implant_label = (
                    "(default: Utah Array fovea left hemisphere)"
                )
                print(f"  Loaded default implant file: {default_path}")
            except Exception as exc:
                print(f"  Failed to load default implant file ({default_path}): {exc}")

        if spec_v2 is None:
            spec_v2 = generate_utah(
                rows=10,
                cols=10,
                pitch_mm=0.4,
                shank_length_mm=1.5,
                tip_angle_deg=25.0,
                contact_diameter_um=20.0,
                name="Utah Array",
            )
            self._default_implant_label = "(built-in Utah array)"
            print("  Falling back to built-in Utah array.")

        self._loaded_implant_spec_v2 = spec_v2
        self._active_design_spec_v2 = spec_v2
        pts_mm, mapping = self._spec_contacts_with_index(spec_v2)
        px = float(self.mm_per_pixel[0])
        self.template_contacts_vox = np.asarray(pts_mm, dtype=np.float64) / px
        self._template_contact_index = list(mapping)
        self.electrodes_per_comb = int(self.template_contacts_vox.shape[0])
        self.entry_electrode_idx = 0
        self._nonspike_anchor_vox = self._default_nonspike_anchor_vox()
        self._apply_default_design_transform_for_spec(spec_v2)
        print(f"  Default implant contacts: {self.electrodes_per_comb}")

        # Mean brain angle across fiducials, surfaced in the startup log.
        brainAngles = []
        for i in range(self.num_combs):
            dY = self.fiducial_posterior_list[i, 1] - self.fiducial_anterior_list[i, 1]
            dZ = self.fiducial_posterior_list[i, 2] - self.fiducial_anterior_list[i, 2]
            if abs(dY) > 0:
                brainAngles.append(np.arctan(abs(dZ) / abs(dY)))
        self.mean_brain_angle = np.mean(brainAngles) if brainAngles else 0.785
        print(f"  Brain angle: {np.degrees(self.mean_brain_angle):.1f} deg")

        # Compatibility aliases used by older computation paths.
        if self.template_contacts_vox.shape[1] >= 3:
            self.base_comb_xy = np.column_stack(
                [self.template_contacts_vox[:, 1], self.template_contacts_vox[:, 2]]
            )
        else:
            self.base_comb_xy = np.empty((0, 2), dtype=np.float64)
        self.conn_pt = np.array([0.0, 0.0], dtype=np.float64)
        self.rotated_base_comb = self.base_comb_xy.copy()
        self._design_revisions.clear()
        self._scene_instances = []
        self._active_design_revision_id = self._register_design_revision(
            spec_v2,
            source_path=self._loaded_implant_path or "",
            design_transform=self.design_transform,
        )
        self._activate_design_revision(self._active_design_revision_id)
        self._update_template_transform()
        if not self._scene_instances:
            self._create_instance(
                revision_id=self._active_design_revision_id, selected=True
            )
        self._apply_loaded_placement_snapshot(spec_v2)

        self._rebuild_probe_colors()

    def _default_nonspike_anchor_vox(self):
        return np.array(
            [
                float(self.fiducial_anterior_list[0, 0]),
                float(self.fiducial_anterior_list[0, 1]),
                float(self._fiducial_plot_z(self.fiducial_anterior_list[0, 2])),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _spec_contacts_with_index(spec):
        pts = np.asarray(spec.contacts_local, dtype=np.float64)
        mapping = []
        for si, sd in enumerate(list(spec.shanks or [])):
            shank_id = int(sd.get("id", si))
            contacts = sd.get("contacts", []) or []
            # Tolerate compact form where ``contacts`` is an integer count
            # instead of a list of contact dicts (some saved designs use this).
            if isinstance(contacts, (int, np.integer)):
                for i in range(int(contacts)):
                    mapping.append((shank_id, i))
                continue
            try:
                contact_iter = list(contacts)
            except TypeError:
                continue
            for ci, c in enumerate(contact_iter):
                if isinstance(c, dict):
                    contact_id = int(c.get("id", ci))
                else:
                    contact_id = ci
                mapping.append((shank_id, contact_id))
        if len(mapping) != int(pts.shape[0]):
            mapping = [(0, i) for i in range(int(pts.shape[0]))]
        return pts, mapping

    @staticmethod
    def _hex_to_rgb01(hex_color):
        hc = str(hex_color).strip().lstrip("#")
        if len(hc) != 6:
            return [0.0, 0.831, 1.0]  # #00D4FF fallback
        try:
            r = int(hc[0:2], 16) / 255.0
            g = int(hc[2:4], 16) / 255.0
            b = int(hc[4:6], 16) / 255.0
            return [r, g, b]
        except Exception:
            return [0.0, 0.831, 1.0]

    @staticmethod
    def _rgb01_to_qcolor(color, alpha=1.0):
        arr = np.asarray(color, dtype=np.float64).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return QColor.fromRgbF(0.0, 0.831, 1.0, float(alpha))
        r = float(np.clip(arr[0], 0.0, 1.0))
        g = float(np.clip(arr[1], 0.0, 1.0))
        b = float(np.clip(arr[2], 0.0, 1.0))
        a = float(np.clip(alpha, 0.0, 1.0))
        return QColor.fromRgbF(r, g, b, a)

    def _rebuild_probe_colors(self):
        mode = str(self.state.get("electrode_color_mode", "classic"))
        n = max(1, int(getattr(self, "num_combs", 16)))

        if mode == "single":
            rgb = self._hex_to_rgb01(self.state.get("electrode_color_hex", "#00D4FF"))
            self.probe_colors = [list(rgb) for _ in range(n)]
            return

        if mode == "high_contrast":
            palette_hex = [
                "#4FC3F7",
                "#FFD54F",
                "#81C784",
                "#FF8A65",
                "#CE93D8",
                "#90A4AE",
                "#AED581",
                "#FFB74D",
            ]
            base = [self._hex_to_rgb01(h) for h in palette_hex]
            self.probe_colors = [list(base[i % len(base)]) for i in range(n)]
            return

        # Classic rainbow
        self.probe_colors = []
        for i in range(n):
            hue = i / float(max(1, n))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            self.probe_colors.append([r, g, b])

    def _refresh_comb_checkbox_colors(self):
        if not hasattr(self, "comb_checkboxes"):
            return
        for i, cb in self.comb_checkboxes.items():
            color = (
                self.probe_colors[i % len(self.probe_colors)]
                if self.probe_colors
                else [1.0, 1.0, 1.0]
            )
            cb.setStyleSheet(
                f"color: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}); "
                "font-weight: bold;"
            )

    def _load_nn_predmaps(self, monkey_name):
        """Load NN predicted maps from NPZ, plus dense maps if available."""
        if not hasattr(self, "nn_predmaps"):
            self.nn_predmaps = {}
        if monkey_name in self.nn_predmaps:
            return

        predmaps_file = os.path.join(
            NN_PREDICTIONS_PATH, f"pRF_NN_predmaps_{monkey_name}.npz"
        )
        if not os.path.exists(predmaps_file):
            raise FileNotFoundError(f"NN predicted maps not found: {predmaps_file}")

        data = np.load(predmaps_file)
        entry = {
            "ecc": data["ecc"].astype("float32"),
            "pol": data["pol"].astype("float32"),
            "r2": data["r2"].astype("float32"),
        }
        if "sz" in data:
            entry["sz"] = data["sz"].astype("float32")
        self.nn_predmaps[monkey_name] = entry

        # --- Dense maps: precompute per-area (pts, ecc, pol, r2) for rendering ---
        if monkey_name in self._nn_dense_brain_cache:
            return  # already precomputed
        import glob

        dense_pattern = os.path.join(
            NN_PREDICTIONS_PATH, f"pRF_NN_predmaps_{monkey_name}_dense_*x.npz"
        )
        dense_files = sorted(glob.glob(dense_pattern))
        if not dense_files:
            return  # no dense file, nothing to precompute

        try:
            from scipy.ndimage import zoom as ndimage_zoom
        except ImportError:
            print("  scipy not available – dense brain rendering disabled")
            return

        dense_data = np.load(dense_files[-1])  # highest density factor
        density_factor = int(dense_data.get("density_factor", 2))
        dense_ecc = dense_data["ecc"].astype("float32")
        dense_pol = dense_data["pol"].astype("float32")
        dense_r2 = dense_data["r2"].astype("float32")
        print(f"  Loading dense NN maps ({density_factor}×) for {monkey_name} …")

        # Zoom original area_volume to dense resolution (nearest-neighbour)
        dense_area_vol = ndimage_zoom(
            self.area_volume.astype(np.uint8), density_factor, order=0
        )
        # Trim/pad to match dense map shape
        ds = dense_ecc.shape
        dense_area_vol = dense_area_vol[: ds[0], : ds[1], : ds[2]]

        area_dense_data = {}
        for area_idx, area_name in enumerate(["V1", "V2", "V3", "V4"], start=1):
            mask = dense_area_vol == area_idx
            if not mask.any():
                continue
            coords = np.where(mask)
            # pRF values from dense maps
            ecc_v = dense_ecc[coords]
            pol_v = dense_pol[coords]
            r2_v = dense_r2[coords]
            # Coordinates in original voxel space (divide by density factor)
            pts = np.column_stack(
                [
                    coords[0].astype(np.float32) / density_factor,
                    coords[1].astype(np.float32) / density_factor,
                    coords[2].astype(np.float32) / density_factor,
                ]
            )
            area_dense_data[area_name] = (pts, ecc_v, pol_v, r2_v)

        total = sum(len(v[0]) for v in area_dense_data.values())
        print(f"  Dense brain data: {total} points across {len(area_dense_data)} areas")
        self._nn_dense_brain_cache[monkey_name] = area_dense_data

    def _nn_monkey_for_source_label(self, source_label):
        mapping = {
            "Monkey D": "Danny",
            "Monkey E": "Eddy",
            "Danny": "Danny",
            "Eddy": "Eddy",
        }
        name = mapping.get(source_label)
        if name is not None:
            return name
        # Human demo sources: "Human Demo (100610)" → fsaverage
        if source_label.startswith("Human Demo"):
            return "fsaverage"
        return source_label

    def _fiducial_plot_z(self, z_value):
        return 107 - z_value if self._invert_fiducial_z else z_value

    def _current_visible_area_indices(self):
        visible = set()
        for idx, name in enumerate(["V1", "V2", "V3", "V4"], start=1):
            if self.state["show_areas"].get(name, False):
                visible.add(idx)
        return visible

    def _resolve_anatomy_source(self, requested_key):
        requested = (requested_key or "auto").strip().lower()
        # Auto priority: ribbon first (if available), then brainmask/brain.
        ordered = [
            k
            for k in ("ribbon", "brainmask", "brain", "atlas_nonzero")
            if k in self._anatomy_candidates
        ]
        if not ordered:
            ordered = ["atlas_nonzero"]

        if requested == "auto":
            return ordered[0], ""
        if requested in ordered:
            return requested, ""
        return ordered[0], f"Anatomy source '{requested_key}' unavailable, using auto."

    def _load_anatomy_mask(self, source_key):
        if source_key == "atlas_nonzero":
            if self.dataset_id == "nhp":
                return np.asarray(self.brain_data > 0, dtype=bool)
            return np.asarray(self.area_volume > 0, dtype=bool)

        path = self._anatomy_paths.get(source_key)
        if path is None or not Path(path).exists():
            return None

        vol_img = nib.as_closest_canonical(nib.load(str(path)))
        vol = vol_img.get_fdata()
        if source_key == "ribbon":
            return np.asarray(vol > 0, dtype=bool)
        return np.asarray(vol > 0, dtype=bool)

    def _build_anatomy_mesh(self, source_key, style_key):
        cache_key = (self.dataset_id, self.human_subject_id, source_key, style_key)
        if cache_key in self._anatomy_mesh_cache:
            return self._anatomy_mesh_cache[cache_key]

        mask = self._load_anatomy_mask(source_key)
        if mask is None or not np.any(mask):
            self._anatomy_mesh_cache[cache_key] = None
            return None

        mesh = None
        if style_key == "surface":
            try:
                img = pv.wrap(mask.astype(np.uint8))
                mesh = img.contour(isosurfaces=[0.5])
                if mesh is not None and mesh.n_points == 0:
                    mesh = None
            except Exception:
                mesh = None
        else:
            coords = np.column_stack(np.where(mask)).astype(np.float32)
            if coords.shape[0] > 300000:
                coords = coords[::2]
            if coords.shape[0] > 300000:
                coords = coords[::2]
            if coords.shape[0] > 0:
                mesh = pv.PolyData(coords)

        self._anatomy_mesh_cache[cache_key] = mesh
        return mesh

    def _refresh_anatomy_source_combo(self):
        if not hasattr(self, "anatomy_source_combo"):
            return
        labels = [
            ("Auto", "auto"),
            ("Brainmask", "brainmask"),
            ("Ribbon", "ribbon"),
            ("Brain", "brain"),
            ("Atlas Fallback", "atlas_nonzero"),
        ]
        self.anatomy_source_combo.blockSignals(True)
        self.anatomy_source_combo.clear()
        for label, key in labels:
            self.anatomy_source_combo.addItem(label, key)
            model_item = self.anatomy_source_combo.model().item(
                self.anatomy_source_combo.count() - 1
            )
            if (
                key not in ("auto", "atlas_nonzero")
                and key not in self._anatomy_candidates
            ):
                model_item.setEnabled(False)
        wanted = self.state.get("anatomy_source", "auto")
        idx = self.anatomy_source_combo.findData(wanted)
        self.anatomy_source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.anatomy_source_combo.blockSignals(False)

    def _update_anatomy_context(self):
        self._safe_remove_actor(self._anatomy_actor_name)

        if not self.state.get("show_anatomy", True):
            self._anatomy_unavailable_note = ""
            return

        source_key, note = self._resolve_anatomy_source(
            self.state.get("anatomy_source", "auto")
        )
        self._anatomy_unavailable_note = note
        style_key = self.state.get("anatomy_style", "surface")
        mesh = self._build_anatomy_mesh(source_key, style_key)
        if mesh is None:
            self._anatomy_unavailable_note = (
                "No anatomy context available for this subject."
            )
            return

        opacity = float(self.state.get("anatomy_opacity", 0.12))
        wireframe = bool(self.state.get("anatomy_wireframe", False))
        if style_key == "surface":
            self.plotter.add_mesh(
                mesh,
                color=(0.75, 0.75, 0.78),
                opacity=opacity,
                smooth_shading=not wireframe,
                style="wireframe" if wireframe else "surface",
                name=self._anatomy_actor_name,
            )
        else:
            self.plotter.add_mesh(
                mesh,
                color=(0.7, 0.7, 0.74),
                point_size=2,
                opacity=opacity,
                render_points_as_spheres=False,
                name=self._anatomy_actor_name,
            )

    def _on_anatomy_controls_changed(self):
        self.state["show_anatomy"] = self.anatomy_show_cb.isChecked()
        self.state["anatomy_source"] = str(
            self.anatomy_source_combo.currentData() or "auto"
        )
        self.state["anatomy_opacity"] = self.anatomy_opacity_slider.value() / 100.0
        self.state["anatomy_style"] = str(
            self.anatomy_style_combo.currentData() or "surface"
        )
        self.state["anatomy_wireframe"] = self.anatomy_wireframe_cb.isChecked()
        # Wireframe is meaningless for the Points style; grey it out so the
        # user isn't surprised by a no-op toggle.
        self.anatomy_wireframe_cb.setEnabled(self.state["anatomy_style"] == "surface")
        self._needs_anatomy_update = True
        self.update_all()

    def _set_selection_overlay(self, selection, locked=False):
        ecc = selection["ecc"]
        pol = selection["pol"]
        if ecc.size == 0:
            self.selection_overlay_rf = {"ecc": [], "pol": [], "sz": [], "seed": None}
            self.selection_seed_voxel = None
            self.selection_overlay_coords_vox = np.empty((0, 3), dtype=np.float32)
            self.polar_plot.clear_selection_overlay()
            self._update_selection_markers()
            return

        seed = None
        if (
            selection.get("seed_ecc") is not None
            and selection.get("seed_pol") is not None
        ):
            seed = (selection["seed_ecc"], selection["seed_pol"])
        self.selection_overlay_rf = {
            "ecc": ecc.tolist(),
            "pol": pol.tolist(),
            "sz": selection["sz"].tolist(),
            "seed": seed,
        }
        self.selection_seed_voxel = selection.get("seed_coord")
        self.selection_overlay_coords_vox = np.asarray(
            selection.get("coords", np.empty((0, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        self.polar_plot.update_selection_overlay(
            self.selection_overlay_rf["ecc"],
            self.selection_overlay_rf["pol"],
            locked=locked,
            seed=self.selection_overlay_rf["seed"],
            sz_list=self.selection_overlay_rf["sz"],
        )
        self._update_selection_markers()

    def _refresh_locked_selection_overlay(self):
        if not self.state.get("selection_enabled", True):
            return
        if not self.selection_locked or self.selection_seed_voxel is None:
            return
        selection = self._selection_from_world(self.selection_seed_voxel)
        if selection is None:
            self._clear_selection()
            return
        self._set_selection_overlay(selection, locked=True)

    def _remove_selection_marker_actors(self):
        if not hasattr(self, "plotter"):
            return
        for name in (
            self._selection_halo_actor_name,
            self._selection_actor_name,
            self._selection_seed_halo_actor_name,
            self._selection_seed_actor_name,
        ):
            if name in self.plotter.actors:
                self._safe_remove_actor(name)
        self._selection_halo_cloud = None
        self._selection_cloud = None
        self._selection_seed_halo_cloud = None
        self._selection_seed_cloud = None

    def _upsert_point_overlay(
        self, cloud_attr, actor_name, points, *, color, point_size, opacity
    ):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            if actor_name in self.plotter.actors:
                self._safe_remove_actor(actor_name)
            setattr(self, cloud_attr, None)
            return

        cloud = getattr(self, cloud_attr, None)
        needs_rebuild = (
            not isinstance(cloud, pv.PolyData)
            or int(cloud.n_points) != int(pts.shape[0])
            or actor_name not in self.plotter.actors
        )
        if needs_rebuild:
            if actor_name in self.plotter.actors:
                self._safe_remove_actor(actor_name)
            cloud = pv.PolyData(pts)
            setattr(self, cloud_attr, cloud)
            self.plotter.add_mesh(
                cloud,
                color=color,
                point_size=float(point_size),
                render_points_as_spheres=False,
                opacity=float(opacity),
                name=actor_name,
                reset_camera=False,
                pickable=False,
            )
            return

        cloud.points = pts
        actor = self.plotter.actors.get(actor_name)
        if actor is not None:
            prop = actor.GetProperty()
            prop.SetPointSize(float(point_size))
            prop.SetColor(*self._hex_to_rgb01(color))
            prop.SetOpacity(float(opacity))

    def _update_selection_markers(self):
        if not hasattr(self, "plotter"):
            return

        coords = np.asarray(self.selection_overlay_coords_vox, dtype=np.float32)
        enabled = bool(self.state.get("selection_enabled", True))
        if (
            coords.ndim != 2
            or coords.shape[1] != 3
            or coords.shape[0] == 0
            or not enabled
        ):
            self._remove_selection_marker_actors()
            return

        neighborhood_size = max(7.0, float(self.state.get("rf_dot_size", 20)) * 0.34)
        halo_size = neighborhood_size + 4.0
        neighborhood_opacity = 0.95 if self.selection_locked else 0.7
        self._upsert_point_overlay(
            "_selection_halo_cloud",
            self._selection_halo_actor_name,
            coords,
            color="#ffffff",
            point_size=halo_size,
            opacity=min(1.0, neighborhood_opacity + 0.12),
        )
        self._upsert_point_overlay(
            "_selection_cloud",
            self._selection_actor_name,
            coords,
            color="#00ffff",
            point_size=neighborhood_size,
            opacity=neighborhood_opacity,
        )

        if (
            self.selection_seed_voxel is not None
            and len(self.selection_seed_voxel) >= 3
        ):
            seed = np.asarray(self.selection_seed_voxel[:3], dtype=np.float32).reshape(
                1, 3
            )
            seed_size = max(neighborhood_size + 3.0, 12.0)
            self._upsert_point_overlay(
                "_selection_seed_halo_cloud",
                self._selection_seed_halo_actor_name,
                seed,
                color="#ffffff",
                point_size=seed_size + 5.0,
                opacity=1.0,
            )
            self._upsert_point_overlay(
                "_selection_seed_cloud",
                self._selection_seed_actor_name,
                seed,
                color="#ffd54f",
                point_size=seed_size,
                opacity=1.0,
            )
        else:
            self._upsert_point_overlay(
                "_selection_seed_halo_cloud",
                self._selection_seed_halo_actor_name,
                np.empty((0, 3), dtype=np.float32),
                color="#ffffff",
                point_size=1.0,
                opacity=1.0,
            )
            self._upsert_point_overlay(
                "_selection_seed_cloud",
                self._selection_seed_actor_name,
                np.empty((0, 3), dtype=np.float32),
                color="#ffd54f",
                point_size=1.0,
                opacity=1.0,
            )

    def _clear_selection(self):
        self.selection_locked = False
        self.selection_seed_voxel = None
        self.selection_overlay_rf = {"ecc": [], "pol": [], "sz": [], "seed": None}
        self.selection_overlay_coords_vox = np.empty((0, 3), dtype=np.float32)
        self.polar_plot.clear_selection_overlay()
        self._update_selection_markers()

    def _on_clear_selection_clicked(self):
        self._clear_selection()

    def _on_selection_enabled_changed(self):
        self.state["selection_enabled"] = bool(self.selection_enabled_cb.isChecked())
        if not self.state["selection_enabled"]:
            self._pending_hover_world = None
            self._hover_pick_timer.stop()
            self._clear_selection()
            return
        self._refresh_locked_selection_overlay()

    def _on_drag_implant_enabled_changed(self):
        self.state["drag_implant_enabled"] = bool(self.drag_implant_cb.isChecked())
        self._drag_active = False
        self._drag_last_world = None
        self._drag_last_pos = None
        self._drag_had_movement = False

    def _on_drag_sensitivity_changed(self):
        self.state["drag_sensitivity_mm"] = float(self.drag_sensitivity_spin.value())

    def _on_drag_requires_ctrl_changed(self):
        self.state["drag_requires_ctrl"] = bool(self.drag_requires_ctrl_cb.isChecked())

    def _on_pick_radius_changed(self, val):
        radius = max(0, int(val))
        self.state["pick_radius_vox"] = radius
        self._selection_offsets = build_ball_offsets(radius)
        # Recompute the locked overlay against the new ball so the user
        # doesn't lose their selection when nudging the radius. Hover
        # previews refresh naturally on the next mouse move.
        self._refresh_locked_selection_overlay()

    def _apply_drag_delta_world(self, world_delta):
        if world_delta is None or len(world_delta) < 3:
            return
        dxyz = np.asarray(world_delta[:3], dtype=np.float64)
        if not np.all(np.isfinite(dxyz)):
            return
        # World coordinates are voxel-space; convert to mm for offsets.
        mm_per_vox = float(self.mm_per_pixel[0])
        sens = float(self.state.get("drag_sensitivity_mm", 1.0))
        dx_mm = float(dxyz[0]) * mm_per_vox * sens
        dy_mm = float(dxyz[1]) * mm_per_vox * sens
        dz_mm = float(dxyz[2]) * mm_per_vox * sens
        if abs(dx_mm) < 1e-9 and abs(dy_mm) < 1e-9 and abs(dz_mm) < 1e-9:
            return

        self._drag_had_movement = True
        if self._instance_mode_active():
            if not self._apply_instance_transform_delta(
                d_tx=dx_mm, d_ty=dy_mm, d_tz=dz_mm
            ):
                return
            first = self._selected_instances()[0]
            place = normalize_instance_placement(first.placement)
            self.state["x"] = float(place["tx_mm"])
            self.state["y"] = float(place["ty_mm"])
            self.state["z"] = float(place["tz_mm"])
            self._last_ui_move_mm = {
                "depth": 0.0,
                "x": self.state["x"],
                "y": self.state["y"],
                "z": self.state["z"],
            }
            self._suppress_value_changed = True
            self.x_spin.setValue(float(self.state["x"]))
            self.y_spin.setValue(float(self.state["y"]))
            self.z_spin.setValue(float(self.state["z"]))
            self._suppress_value_changed = False
        else:
            selected = self._selected_comb_indices()
            if not selected:
                return
            for idx in selected:
                offs = self.comb_offsets_mm.setdefault(
                    idx, {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
                )
                offs["x"] += dx_mm
                offs["y"] += dy_mm
                offs["z"] += dz_mm

            self.state["x"] += dx_mm
            self.state["y"] += dy_mm
            self.state["z"] += dz_mm
            self._last_ui_move_mm = {
                "depth": float(self.state.get("depth", 0.0)),
                "x": float(self.state.get("x", 0.0)),
                "y": float(self.state.get("y", 0.0)),
                "z": float(self.state.get("z", 0.0)),
            }
            self._suppress_value_changed = True
            self.x_spin.setValue(float(self.state["x"]))
            self.y_spin.setValue(float(self.state["y"]))
            self.z_spin.setValue(float(self.state["z"]))
            self._suppress_value_changed = False

        self._needs_electrode_update = True
        self._update_pending = True
        self._update_timer.start(self._debounce_ms)

    def _setup_pointer_picking(self):
        if not hasattr(self, "plotter"):
            return
        if getattr(self, "_pick_observers_added", False):
            return
        try:
            self.plotter.track_mouse_position()
        except Exception:
            pass
        try:
            self.plotter.iren.add_observer("MouseMoveEvent", self._on_mouse_move_pick)
            self.plotter.iren.add_observer(
                "LeftButtonPressEvent", self._on_left_click_pick
            )
            self.plotter.iren.add_observer(
                "LeftButtonReleaseEvent", self._on_left_release_pick
            )
            self._pick_observers_added = True
        except Exception:
            self._pick_observers_added = False

    def _drag_world_delta_from_pixels(self, dx_px, dy_px):
        """
        Approximate world-space (voxel) delta from screen pixel delta using
        current camera right/up vectors.
        """
        try:
            cam = self.plotter.camera
            pos = np.asarray(cam.position, dtype=np.float64)
            focal = np.asarray(cam.focal_point, dtype=np.float64)
            up = np.asarray(cam.up, dtype=np.float64)
            fwd = focal - pos
            fwd_n = np.linalg.norm(fwd)
            up_n = np.linalg.norm(up)
            if fwd_n < 1e-12 or up_n < 1e-12:
                return np.array([dx_px * 0.2, -dy_px * 0.2, 0.0], dtype=np.float64)
            fwd = fwd / fwd_n
            up = up / up_n
            right = np.cross(fwd, up)
            r_n = np.linalg.norm(right)
            if r_n < 1e-12:
                return np.array([dx_px * 0.2, -dy_px * 0.2, 0.0], dtype=np.float64)
            right = right / r_n
            up_ortho = np.cross(right, fwd)
            u_n = np.linalg.norm(up_ortho)
            if u_n < 1e-12:
                return np.array([dx_px * 0.2, -dy_px * 0.2, 0.0], dtype=np.float64)
            up_ortho = up_ortho / u_n
            # 0.2 vox/px baseline, user sensitivity scales in mm later.
            scale_vox_per_px = 0.2
            return (right * float(dx_px) + up_ortho * float(-dy_px)) * scale_vox_per_px
        except Exception:
            return np.array([dx_px * 0.2, -dy_px * 0.2, 0.0], dtype=np.float64)

    def _event_pos(self, event):
        try:
            return event.pos()
        except Exception:
            try:
                return event.position().toPoint()
            except Exception:
                return None

    def _event_modifiers(self, event):
        try:
            modifiers = event.modifiers
        except Exception:
            modifiers = None
        if callable(modifiers):
            try:
                return modifiers()
            except Exception:
                pass
        try:
            return QApplication.keyboardModifiers()
        except Exception:
            return Qt.NoModifier

    def eventFilter(self, obj, event):
        try:
            interactor = getattr(self.plotter, "interactor", None)
        except Exception:
            interactor = None
        if obj is not interactor:
            return super().eventFilter(obj, event)

        if not self.state.get("drag_implant_enabled", False):
            return super().eventFilter(obj, event)

        et = event.type()
        ctrl_required = bool(self.state.get("drag_requires_ctrl", True))
        ctrl_held = bool(self._event_modifiers(event) & Qt.ControlModifier)
        if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if ctrl_required and not ctrl_held:
                return super().eventFilter(obj, event)
            if self._instance_mode_active():
                try:
                    world = self.plotter.pick_click_position()
                except Exception:
                    world = None
                hit_id = self._hit_test_instance(world)
                if hit_id is None:
                    return super().eventFilter(obj, event)
                if hit_id not in self._selected_instance_ids:
                    self._set_instance_selection([hit_id], additive=False, toggle=False)
                self._sync_instance_controls_from_selection()
                self._needs_electrode_update = True
                self.update_all()
            elif not self._selected_comb_indices():
                return super().eventFilter(obj, event)
            self._push_undo_state()
            self._drag_active = True
            self._drag_had_movement = False
            self._drag_camera_position = self.plotter.camera_position
            self._drag_last_world = None
            self._drag_last_pos = self._event_pos(event)
            return True

        if et == QEvent.MouseMove and self._drag_active:
            if ctrl_required and not ctrl_held:
                self._drag_active = False
                self._drag_last_pos = None
                self._drag_last_world = None
                self._drag_camera_position = None
                return super().eventFilter(obj, event)
            pos = self._event_pos(event)
            if pos is not None and self._drag_last_pos is not None:
                dx = pos.x() - self._drag_last_pos.x()
                dy = pos.y() - self._drag_last_pos.y()
                if dx != 0 or dy != 0:
                    world_delta = self._drag_world_delta_from_pixels(dx, dy)
                    self._apply_drag_delta_world(world_delta)
            self._drag_last_pos = pos
            if self._drag_camera_position is not None:
                try:
                    self.plotter.camera_position = self._drag_camera_position
                except Exception:
                    pass
            return True

        if (
            et == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
            and self._drag_active
        ):
            self._drag_active = False
            self._drag_last_pos = None
            self._drag_last_world = None
            self._drag_camera_position = None
            return True

        return super().eventFilter(obj, event)

    def _is_camera_interacting(self):
        try:
            style = self.plotter.interactor.GetInteractorStyle()
            return bool(
                style is not None
                and hasattr(style, "GetState")
                and style.GetState() != 0
            )
        except Exception:
            return False

    def _selection_from_world(self, world_xyz):
        if world_xyz is None:
            return None
        if len(world_xyz) < 3:
            return None
        world_arr = np.asarray(world_xyz[:3], dtype=np.float32)
        if not np.all(np.isfinite(world_arr)):
            return None
        seed = tuple(int(np.round(v)) for v in world_arr)
        selection = collect_local_rf_neighborhood(
            seed_coord=seed,
            offsets_xyz=self._selection_offsets,
            ecc_map=self.ecc_map,
            polar_map=self.polar_map,
            r2_map=self.R2_map,
            area_volume=self.area_volume,
            visible_area_indices=self._current_visible_area_indices(),
            r2_threshold=float(self.state["r2"]),
            max_points=SELECTION_MAX_POINTS,
            sz_map=self.sz_map,
        )
        if selection["ecc"].size == 0:
            return None
        return selection

    def _on_mouse_move_pick(self, _obj=None, _event=None):
        if self.state.get("drag_implant_enabled", False):
            return
        if self._is_camera_interacting():
            return
        if not self.state.get("selection_enabled", True):
            return
        if self.selection_locked:
            return
        try:
            world = self.plotter.pick_mouse_position()
        except Exception:
            world = None
        self._pending_hover_world = world
        if not self._hover_pick_timer.isActive():
            self._hover_pick_timer.start()

    def _process_pending_hover_pick(self):
        if not self.state.get("selection_enabled", True):
            return
        if self.selection_locked:
            return
        selection = self._selection_from_world(self._pending_hover_world)
        if selection is None:
            self.selection_overlay_rf = {"ecc": [], "pol": [], "sz": [], "seed": None}
            self.selection_seed_voxel = None
            self.selection_overlay_coords_vox = np.empty((0, 3), dtype=np.float32)
            self.polar_plot.clear_selection_overlay()
            self._update_selection_markers()
            return
        self._set_selection_overlay(selection, locked=False)

    def _on_left_click_pick(self, _obj=None, _event=None):
        try:
            world = self.plotter.pick_click_position()
        except Exception:
            world = None
        mods = QApplication.keyboardModifiers()
        additive = bool(mods & Qt.ControlModifier)
        if self._instance_mode_active():
            hit_id = self._hit_test_instance(world)
            if hit_id is not None:
                self._set_instance_selection(
                    [hit_id], additive=additive, toggle=additive
                )
                self._sync_instance_controls_from_selection()
                self._needs_electrode_update = True
                self.update_all()
                return
            if not self.state.get("selection_enabled", True):
                if not additive:
                    self._clear_instance_selection()
                    self._needs_electrode_update = True
                    self.update_all()
                return
        if self.state.get("drag_implant_enabled", False):
            return
        if not self.state.get("selection_enabled", True):
            return
        selection = self._selection_from_world(world)
        if selection is None:
            if self.selection_locked:
                self._clear_selection()
            return
        self.selection_locked = True
        self._set_selection_overlay(selection, locked=True)

    def _on_left_release_pick(self, _obj=None, _event=None):
        if not self._drag_active:
            return
        self._drag_active = False
        self._drag_last_world = None
        self._drag_last_pos = None
        self._drag_camera_position = None

    def _apply_r2_control_state(self):
        source = self.state.get("prf_source", "")
        is_nn_source = source.startswith("NN Prediction (")
        controls_enabled = (self.dataset_id == "nhp") and (not is_nn_source)
        self.r2_slider.setEnabled(controls_enabled)
        self.r2_spin.setEnabled(controls_enabled)

        if self.dataset_id == "human_demo":
            hint = "Disabled in human demo: R2 is synthetic in inferred maps."
        elif is_nn_source:
            hint = "Disabled for NN prediction maps."
        else:
            hint = "Threshold for measured pRF maps."
        self.r2_slider.setToolTip(hint)
        self.r2_spin.setToolTip(hint)

    def _discover_human_subjects(self):
        subjects_root = REPO_ROOT / "data" / "human" / "demo_subject" / "subjects"
        if not subjects_root.exists():
            return [DEFAULT_HUMAN_SUBJECT]

        all_subjects = sorted([p.name for p in subjects_root.iterdir() if p.is_dir()])
        subjects = all_subjects.copy()
        if not subjects:
            subjects = [DEFAULT_HUMAN_SUBJECT]
        return subjects

    def _clear_dataset_memory(self):
        # Drop large arrays and caches so old dataset can be reclaimed.
        attrs_to_clear = [
            "brain_data",
            "area_volume",
            "visual_areas",
            "prf_maps",
            "ecc_map",
            "polar_map",
            "R2_map",
            "fiducial_anterior_list",
            "fiducial_posterior_list",
            "nn_predmaps",
        ]
        for name in attrs_to_clear:
            if hasattr(self, name):
                delattr(self, name)

        self._nn_source_lookup = {}
        self._nn_dense_brain_data = None
        self._nn_dense_brain_cache = {}
        self._electrode_clouds.clear()
        self._fiducials_mesh = None
        self._area_clouds.clear()
        self._area_scalars_cache.clear()
        self._anatomy_mesh_cache.clear()
        self._anatomy_paths = {}
        self._anatomy_candidates = ["atlas_nonzero"]
        self.electrode_actors = []
        self.selection_locked = False
        self.selection_seed_voxel = None
        self.selection_overlay_rf = {"ecc": [], "pol": [], "sz": [], "seed": None}
        self.selection_overlay_coords_vox = np.empty((0, 3), dtype=np.float32)
        self._drag_active = False
        self._drag_last_world = None
        self._anatomy_unavailable_note = ""
        self._nonspike_anchor_vox = None
        self._default_implant_label = "(built-in Utah array)"
        self._coord_frame = "voxel_ras_v1"
        self._orientation_original = "RAS"
        self._orientation_canonical = "RAS"
        self._legacy_vector_map = np.eye(3, dtype=np.float64)
        self._needs_anatomy_update = True

        if hasattr(self, "polar_plot"):
            self.polar_plot.clear_selection_overlay()

        if hasattr(self, "plotter"):
            try:
                self.plotter.clear()
                self._ensure_credit_overlay()
            except Exception:
                pass

        gc.collect()

    def _refresh_prf_source_combo(self, apply_source=True):
        prf_sources = list(self.prf_maps.keys())
        self._nn_source_lookup = {}
        if self._supports_nn_predictions:
            for source_key in self.prf_maps.keys():
                monkey_name = self._nn_monkey_for_source_label(source_key)
                predmaps_file = os.path.join(
                    NN_PREDICTIONS_PATH, f"pRF_NN_predmaps_{monkey_name}.npz"
                )
                if os.path.exists(predmaps_file):
                    nn_source = f"NN Prediction ({source_key})"
                    prf_sources.append(nn_source)
                    self._nn_source_lookup[nn_source] = monkey_name

        self.prf_combo.blockSignals(True)
        self.prf_combo.clear()
        self.prf_combo.addItems(prf_sources)
        default_source = self.state.get("prf_source", "")
        default_idx = (
            prf_sources.index(default_source) if default_source in prf_sources else 0
        )
        self.prf_combo.setCurrentIndex(default_idx)
        self.prf_combo.blockSignals(False)

        if prf_sources:
            selected = self.prf_combo.currentText()
            self.state["prf_source"] = selected
            if apply_source:
                self._on_prf_source_changed(selected)
            else:
                self._apply_r2_control_state()
        else:
            self._apply_r2_control_state()

        if hasattr(self, "_r2_hint_label"):
            self._r2_hint_label.setVisible(self.dataset_id == "human_demo")

    def _on_dataset_mode_changed(self, _idx):
        is_human = self.dataset_combo.currentData() == "human_demo"
        self.human_subject_combo.setEnabled(is_human)

    def _on_reload_dataset_clicked(self):
        new_dataset = str(self.dataset_combo.currentData())
        new_subject = (
            self.human_subject_combo.currentText().strip() or DEFAULT_HUMAN_SUBJECT
        )
        self._switch_dataset(new_dataset, new_subject)

    def _refresh_comb_checkbox_state(self):
        if self._instance_mode_active():
            if hasattr(self, "combs_group"):
                self.combs_group.setEnabled(False)
                self.combs_group.setVisible(False)
            return
        if hasattr(self, "combs_group"):
            self.combs_group.setEnabled(True)
            self.combs_group.setVisible(True)
        for i, cb in self.comb_checkboxes.items():
            enabled = i == 0
            cb.blockSignals(True)
            cb.setVisible(enabled)
            cb.setEnabled(enabled)
            cb.setChecked(enabled)
            cb.blockSignals(False)
            self.state["show_combs"][i] = bool(enabled)
        self._refresh_comb_checkbox_colors()

    def _switch_dataset(self, new_dataset_id, new_human_subject_id):
        new_dataset_id = str(new_dataset_id).strip().lower()
        new_human_subject_id = (
            str(new_human_subject_id).strip() or DEFAULT_HUMAN_SUBJECT
        )

        if new_dataset_id == self.dataset_id and (
            new_dataset_id != "human_demo"
            or new_human_subject_id == self.human_subject_id
        ):
            return

        previous_dataset = self.dataset_id
        previous_subject = self.human_subject_id
        self.dataset_id = new_dataset_id
        self.human_subject_id = new_human_subject_id

        self._update_timer.stop()
        self._update_pending = False

        try:
            self._clear_dataset_memory()
            self._load_data()
        except Exception as exc:
            self.dataset_id = previous_dataset
            self.human_subject_id = previous_subject
            self._clear_dataset_memory()
            self._load_data()
            self._show_warning_message("Dataset Load Failed", str(exc))
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.setCurrentIndex(
                self.dataset_combo.findData(self.dataset_id)
            )
            self.dataset_combo.blockSignals(False)
            self.human_subject_combo.setCurrentText(self.human_subject_id)
            return

        # Update dataset controls after successful load.
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.setCurrentIndex(self.dataset_combo.findData(self.dataset_id))
        self.dataset_combo.blockSignals(False)

        available_subjects = self._discover_human_subjects()
        self.human_subject_combo.blockSignals(True)
        self.human_subject_combo.clear()
        self.human_subject_combo.addItems(available_subjects)
        if self.human_subject_id not in available_subjects:
            self.human_subject_combo.addItem(self.human_subject_id)
        self.human_subject_combo.setCurrentText(self.human_subject_id)
        self.human_subject_combo.blockSignals(False)
        self._on_dataset_mode_changed(0)

        self._refresh_comb_checkbox_state()
        self._refresh_prf_source_combo()
        self._refresh_anatomy_source_combo()
        self._clear_selection()
        self._reset_position()
        self._first_render = True
        self._needs_brain_update = True
        self._needs_electrode_update = True
        self._needs_anatomy_update = True
        # Apply the new dataset's polar-angle convention to the RF coverage widget.
        self.polar_plot.reset_convention(self.pol_convention)
        self.update_all()

    def _build_ui(self):
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # LEFT: Control Panel
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)

        # CENTER: 3D View
        vtk_frame = QFrame()
        vtk_layout = QVBoxLayout(vtk_frame)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(vtk_frame)
        self.plotter.set_background("black")
        self.plotter.show_axes()
        self.plotter.add_axes(
            xlabel="X", ylabel="Y", zlabel="Z", color="white", line_width=4
        )
        self._ensure_credit_overlay()
        self.plotter.interactor.installEventFilter(self)
        vtk_layout.addWidget(self.plotter.interactor)
        splitter.addWidget(vtk_frame)

        # RIGHT: Polar Plot
        polar_frame = QFrame()
        polar_frame.setStyleSheet("background-color: black;")
        polar_layout = QVBoxLayout(polar_frame)
        polar_layout.setContentsMargins(10, 10, 10, 10)
        polar_label = QLabel("RF Coverage")
        polar_label.setAlignment(Qt.AlignCenter)
        polar_label.setFont(QFont("Arial", 14, QFont.Bold))
        polar_label.setStyleSheet("color: white; background-color: black;")
        polar_layout.addWidget(polar_label)
        self.polar_plot = PolarPlotWidget(
            convention=getattr(self, "pol_convention", "standard"),
            max_ecc=getattr(self, "_ecc_scope", 15),
        )
        polar_layout.addWidget(self.polar_plot, stretch=1)
        splitter.addWidget(polar_frame)

        splitter.setSizes([280, 850, 450])
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self._on_undo_clicked)
        self._redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self._redo_shortcut.activated.connect(self._on_redo_clicked)

    def _create_control_panel(self):
        """Create the left control panel."""
        scroll_container = QWidget()
        scroll_container.setMinimumWidth(200)
        container_layout = QVBoxLayout(scroll_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #1e1e1e; }")

        panel = QWidget()
        panel.setStyleSheet("background-color: #1e1e1e;")
        layout = QVBoxLayout(panel)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title
        title = QLabel("vimplant2 Implant Explorer")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        setup_page = QWidget()
        analysis_page = QWidget()
        view_page = QWidget()
        setup_layout = QVBoxLayout(setup_page)
        analysis_layout_root = QVBoxLayout(analysis_page)
        view_layout = QVBoxLayout(view_page)
        for tl in (setup_layout, analysis_layout_root, view_layout):
            tl.setContentsMargins(4, 4, 4, 4)
            tl.setSpacing(6)
        tabs.addTab(setup_page, "Setup")
        tabs.addTab(analysis_page, "Analysis")
        tabs.addTab(view_page, "View")
        layout.addWidget(tabs, stretch=1)

        # Dataset Controls
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_group)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("Type:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("NHP", "nhp")
        self.dataset_combo.addItem("Human Demo", "human_demo")
        initial_dataset_idx = self.dataset_combo.findData(self.dataset_id)
        self.dataset_combo.setCurrentIndex(
            initial_dataset_idx if initial_dataset_idx >= 0 else 0
        )
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_mode_changed)
        dataset_row.addWidget(self.dataset_combo)
        dataset_layout.addLayout(dataset_row)

        subject_row = QHBoxLayout()
        subject_row.addWidget(QLabel("Subject:"))
        self.human_subject_combo = QComboBox()
        self.human_subject_combo.setEditable(True)
        self.human_subject_combo.addItems(self._discover_human_subjects())
        self.human_subject_combo.setCurrentText(self.human_subject_id)
        subject_row.addWidget(self.human_subject_combo)
        dataset_layout.addLayout(subject_row)

        load_dataset_btn = QPushButton("Load Dataset")
        load_dataset_btn.clicked.connect(self._on_reload_dataset_clicked)
        dataset_layout.addWidget(load_dataset_btn)
        setup_layout.addWidget(dataset_group)
        self._on_dataset_mode_changed(0)

        # Placement Controls
        pos_group = QGroupBox("Placement")
        pos_layout = QVBoxLayout(pos_group)
        self.depth_slider, self.depth_spin = self._create_slider_with_spin(
            "Depth", -50, 50, 0, pos_layout
        )
        self.x_slider, self.x_spin = self._create_slider_with_spin(
            "X Offset", -50, 50, 0, pos_layout
        )
        self.y_slider, self.y_spin = self._create_slider_with_spin(
            "Y Offset", -50, 50, 0, pos_layout
        )
        self.z_slider, self.z_spin = self._create_slider_with_spin(
            "Z Offset", -50, 50, 0, pos_layout
        )
        self.place_rot_x_slider, self.place_rot_x_spin = self._create_slider_with_spin(
            "Rot X",
            -180,
            180,
            0,
            pos_layout,
            decimals=1,
            step=0.5,
            callback=self._on_instance_rotation_changed,
        )
        self.place_rot_y_slider, self.place_rot_y_spin = self._create_slider_with_spin(
            "Rot Y",
            -180,
            180,
            0,
            pos_layout,
            decimals=1,
            step=0.5,
            callback=self._on_instance_rotation_changed,
        )
        self.place_rot_z_slider, self.place_rot_z_spin = self._create_slider_with_spin(
            "Rot Z",
            -180,
            180,
            0,
            pos_layout,
            decimals=1,
            step=0.5,
            callback=self._on_instance_rotation_changed,
        )
        reset_place_btn = QPushButton("Reset Placement")
        reset_place_btn.clicked.connect(self._reset_position)
        pos_layout.addWidget(reset_place_btn)
        setup_layout.addWidget(pos_group)

        # Manipulation Controls (template geometry)
        manip_group = QGroupBox("Manipulation")
        manip_layout = QVBoxLayout(manip_group)
        self.rot_x_slider, self.rot_x_spin = self._create_slider_with_spin(
            "Rot X",
            -180,
            180,
            0,
            manip_layout,
            decimals=1,
            step=0.5,
            callback=self._on_manipulation_changed,
        )
        self.rot_y_slider, self.rot_y_spin = self._create_slider_with_spin(
            "Rot Y",
            -180,
            180,
            0,
            manip_layout,
            decimals=1,
            step=0.5,
            callback=self._on_manipulation_changed,
        )
        self.rot_z_slider, self.rot_z_spin = self._create_slider_with_spin(
            "Rot Z",
            -180,
            180,
            0,
            manip_layout,
            decimals=1,
            step=0.5,
            callback=self._on_manipulation_changed,
        )
        self.scale_slider, self.scale_spin = self._create_slider_with_spin(
            "Scale",
            0.25,
            3.0,
            1.0,
            manip_layout,
            decimals=2,
            step=0.01,
            callback=self._on_manipulation_changed,
        )

        pivot_row = QHBoxLayout()
        pivot_row.addWidget(QLabel("Pivot:"))
        self.pivot_combo = QComboBox()
        self.pivot_combo.addItem("Entry", "entry")
        self.pivot_combo.addItem("Centroid", "centroid")
        self.pivot_combo.addItem("Custom", "custom")
        self.pivot_combo.currentIndexChanged.connect(self._on_manipulation_changed)
        pivot_row.addWidget(self.pivot_combo)
        manip_layout.addLayout(pivot_row)

        self.pivot_x_spin = QDoubleSpinBox()
        self.pivot_y_spin = QDoubleSpinBox()
        self.pivot_z_spin = QDoubleSpinBox()
        for spin, label in [
            (self.pivot_x_spin, "Pivot X"),
            (self.pivot_y_spin, "Pivot Y"),
            (self.pivot_z_spin, "Pivot Z"),
        ]:
            spin.setRange(-50.0, 50.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            row = QHBoxLayout()
            row.addWidget(QLabel(label + " (mm):"))
            row.addWidget(spin)
            manip_layout.addLayout(row)
            spin.valueChanged.connect(self._on_manipulation_changed)

        self.mirror_btn = QPushButton("Mirror X: OFF")
        self.mirror_btn.clicked.connect(self._toggle_mirror_x)
        manip_layout.addWidget(self.mirror_btn)

        snap_row = QHBoxLayout()
        self.snap_cb = QCheckBox("Snap")
        self.snap_cb.setChecked(self.state.get("snap_enabled", True))
        self.snap_cb.stateChanged.connect(self._on_manipulation_changed)
        snap_row.addWidget(self.snap_cb)
        manip_layout.addLayout(snap_row)

        self.snap_mm_spin = self._add_double_spin_row(
            manip_layout,
            "Snap mm:",
            self.state.get("snap_mm", 0.1),
            lo=0.01,
            hi=10.0,
            decimals=2,
            step=0.01,
            callback=self._on_manipulation_changed,
        )
        self.snap_deg_spin = self._add_double_spin_row(
            manip_layout,
            "Snap deg:",
            self.state.get("snap_deg", 1.0),
            lo=0.1,
            hi=45.0,
            decimals=1,
            step=0.1,
            callback=self._on_manipulation_changed,
        )
        self.snap_scale_spin = self._add_double_spin_row(
            manip_layout,
            "Snap scale:",
            self.state.get("snap_scale", 0.01),
            lo=0.001,
            hi=1.0,
            decimals=3,
            step=0.001,
            callback=self._on_manipulation_changed,
        )

        hist_row = QHBoxLayout()
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self._on_undo_clicked)
        redo_btn = QPushButton("Redo")
        redo_btn.clicked.connect(self._on_redo_clicked)
        hist_row.addWidget(undo_btn)
        hist_row.addWidget(redo_btn)
        manip_layout.addLayout(hist_row)
        reset_manip_btn = QPushButton("Reset Manipulation")
        reset_manip_btn.clicked.connect(self._reset_manipulation)
        manip_layout.addWidget(reset_manip_btn)
        setup_layout.addWidget(manip_group)

        # Analysis Controls
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        self.r2_slider, self.r2_spin = self._create_slider_with_spin(
            "R² Threshold",
            0,
            R2_THRESHOLD_MAX,
            R2_THRESHOLD_INIT,
            analysis_layout,
            decimals=1,
            step=0.1,
        )

        # Undersampling
        undersample_layout = QHBoxLayout()
        undersample_layout.addWidget(QLabel("Undersampling:"))
        self.undersample_spin = QSpinBox()
        self.undersample_spin.setRange(1, 20)
        self.undersample_spin.setValue(UNDERSAMPLING)
        self.undersample_spin.valueChanged.connect(self._on_undersampling_changed)
        undersample_layout.addWidget(self.undersample_spin)
        analysis_layout.addLayout(undersample_layout)

        # Mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Color Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Anatomy", "Eccentricity", "Polar Angle"])
        self.mode_combo.setCurrentIndex(2)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        analysis_layout.addLayout(mode_layout)

        # pRF Source
        prf_layout = QHBoxLayout()
        prf_layout.addWidget(QLabel("pRF Source:"))
        self.prf_combo = QComboBox()
        self.prf_combo.currentTextChanged.connect(self._on_prf_source_changed)
        prf_layout.addWidget(self.prf_combo)
        analysis_layout.addLayout(prf_layout)
        self._r2_hint_label = QLabel(
            "R2 threshold disabled for human demo (synthetic map)."
        )
        self._r2_hint_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        analysis_layout.addWidget(self._r2_hint_label)
        self._refresh_prf_source_combo(apply_source=False)
        analysis_layout_root.addWidget(analysis_group)

        # Visual Areas
        areas_group = QGroupBox("Visual Areas")
        areas_layout = QVBoxLayout(areas_group)
        self.area_checkboxes = {}
        for name, color in AREA_COLORS.items():
            cb = QCheckBox(name)
            cb.setChecked(self.state["show_areas"].get(name, False))
            cb.setStyleSheet(
                f"color: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}); font-weight: bold;"
            )
            cb.stateChanged.connect(self._on_area_visibility_changed)
            areas_layout.addWidget(cb)
            self.area_checkboxes[name] = cb
        analysis_layout_root.addWidget(areas_group)

        # Brain Area Style Settings (points only)
        style_group = QGroupBox("Brain Area Style")
        style_layout = QVBoxLayout(style_group)

        # Point size
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Point Size:"))
        self.area_size_spin = QSpinBox()
        self.area_size_spin.setRange(1, 20)
        self.area_size_spin.setValue(5)
        self.area_size_spin.valueChanged.connect(self._on_area_style_changed)
        size_row.addWidget(self.area_size_spin)
        style_layout.addLayout(size_row)

        # Opacity slider
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self.area_opacity_slider = QSlider(Qt.Horizontal)
        self.area_opacity_slider.setRange(10, 100)
        self.area_opacity_slider.setValue(50)
        self.area_opacity_slider.valueChanged.connect(self._on_area_style_changed)
        opacity_row.addWidget(self.area_opacity_slider)
        style_layout.addLayout(opacity_row)

        analysis_layout_root.addWidget(style_group)

        # Anatomy context
        anatomy_group = QGroupBox("Anatomy Context")
        anatomy_layout = QVBoxLayout(anatomy_group)

        self.anatomy_show_cb = QCheckBox("Show anatomy")
        self.anatomy_show_cb.setChecked(self.state.get("show_anatomy", True))
        self.anatomy_show_cb.stateChanged.connect(self._on_anatomy_controls_changed)
        anatomy_layout.addWidget(self.anatomy_show_cb)

        anatomy_source_row = QHBoxLayout()
        anatomy_source_row.addWidget(QLabel("Source:"))
        self.anatomy_source_combo = QComboBox()
        self.anatomy_source_combo.currentIndexChanged.connect(
            self._on_anatomy_controls_changed
        )
        anatomy_source_row.addWidget(self.anatomy_source_combo)
        anatomy_layout.addLayout(anatomy_source_row)
        self._refresh_anatomy_source_combo()

        anatomy_style_row = QHBoxLayout()
        anatomy_style_row.addWidget(QLabel("Style:"))
        self.anatomy_style_combo = QComboBox()
        self.anatomy_style_combo.addItem("Surface", "surface")
        self.anatomy_style_combo.addItem("Points", "points")
        style_idx = self.anatomy_style_combo.findData(
            self.state.get("anatomy_style", "surface")
        )
        self.anatomy_style_combo.setCurrentIndex(style_idx if style_idx >= 0 else 0)
        self.anatomy_style_combo.currentIndexChanged.connect(
            self._on_anatomy_controls_changed
        )
        anatomy_style_row.addWidget(self.anatomy_style_combo)
        anatomy_layout.addLayout(anatomy_style_row)

        anatomy_opacity_row = QHBoxLayout()
        anatomy_opacity_row.addWidget(QLabel("Opacity:"))
        self.anatomy_opacity_slider = QSlider(Qt.Horizontal)
        self.anatomy_opacity_slider.setRange(5, 100)
        self.anatomy_opacity_slider.setValue(
            int(self.state.get("anatomy_opacity", 0.12) * 100)
        )
        self.anatomy_opacity_slider.valueChanged.connect(
            self._on_anatomy_controls_changed
        )
        anatomy_opacity_row.addWidget(self.anatomy_opacity_slider)
        anatomy_layout.addLayout(anatomy_opacity_row)

        self.anatomy_wireframe_cb = QCheckBox("Wireframe surface")
        self.anatomy_wireframe_cb.setChecked(self.state.get("anatomy_wireframe", False))
        # Wireframe only applies when the anatomy is rendered as a surface mesh.
        self.anatomy_wireframe_cb.setEnabled(
            str(self.state.get("anatomy_style", "surface")) == "surface"
        )
        self.anatomy_wireframe_cb.setToolTip(
            "Render the anatomy surface mesh as wireframe (no effect when style is Points)."
        )
        self.anatomy_wireframe_cb.stateChanged.connect(
            self._on_anatomy_controls_changed
        )
        anatomy_layout.addWidget(self.anatomy_wireframe_cb)

        analysis_layout_root.addWidget(anatomy_group)

        # Pointer RF selection
        picking_group = QGroupBox("RF Picking")
        picking_layout = QVBoxLayout(picking_group)
        self.selection_enabled_cb = QCheckBox("Enable RF picking")
        self.selection_enabled_cb.setChecked(self.state.get("selection_enabled", True))
        self.selection_enabled_cb.stateChanged.connect(
            self._on_selection_enabled_changed
        )
        picking_layout.addWidget(self.selection_enabled_cb)

        pick_radius_row = QHBoxLayout()
        pick_radius_row.addWidget(QLabel("Radius (vox):"))
        self.pick_radius_spin = QSpinBox()
        self.pick_radius_spin.setRange(0, 20)
        self.pick_radius_spin.setValue(
            int(self.state.get("pick_radius_vox", SELECTION_RADIUS_VOX))
        )
        self.pick_radius_spin.valueChanged.connect(self._on_pick_radius_changed)
        pick_radius_row.addWidget(self.pick_radius_spin)
        picking_layout.addLayout(pick_radius_row)

        picking_hint = QLabel(
            "Hover previews RF neighborhood. Click locks. Clear to reset."
        )
        picking_hint.setWordWrap(True)
        picking_hint.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        picking_layout.addWidget(picking_hint)
        clear_sel_btn = QPushButton("Clear Selection")
        clear_sel_btn.clicked.connect(self._on_clear_selection_clicked)
        picking_layout.addWidget(clear_sel_btn)
        analysis_layout_root.addWidget(picking_group)

        export_group = QGroupBox("RF Export")
        export_layout = QVBoxLayout(export_group)
        export_csv_btn = QPushButton("Export RFs CSV...")
        export_csv_btn.clicked.connect(self._export_rfs_csv)
        export_layout.addWidget(export_csv_btn)
        export_json_btn = QPushButton("Export RFs JSON...")
        export_json_btn.clicked.connect(self._export_rfs_json)
        export_layout.addWidget(export_json_btn)
        export_hint = QLabel("Exports only RFs currently visible in the polar plot.")
        export_hint.setWordWrap(True)
        export_hint.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        export_layout.addWidget(export_hint)
        analysis_layout_root.addWidget(export_group)

        # Display Settings
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self.show_electrodes_cb = QCheckBox("Show electrodes")
        self.show_electrodes_cb.setChecked(self.state.get("show_electrodes", True))
        self.show_electrodes_cb.stateChanged.connect(self._on_display_changed)
        display_layout.addWidget(self.show_electrodes_cb)

        for label_text, attr, default, range_vals in [
            ("Electrode Size:", "elec_size_spin", 4, (2, 30)),
            ("RF Dot Size:", "rf_size_spin", 20, (5, 150)),
            ("Fiducial Size:", "fid_size_spin", 15, (5, 50)),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            spin = QSpinBox()
            spin.setRange(*range_vals)
            spin.setValue(default)
            spin.valueChanged.connect(self._on_display_changed)
            row.addWidget(spin)
            display_layout.addLayout(row)
            setattr(self, attr, spin)

        self.rf_centers_only_cb = QCheckBox("RF centers only")
        self.rf_centers_only_cb.setChecked(self.state.get("rf_centers_only", False))
        self.rf_centers_only_cb.stateChanged.connect(self._on_display_changed)
        display_layout.addWidget(self.rf_centers_only_cb)

        rf_alpha_row = QHBoxLayout()
        rf_alpha_row.addWidget(QLabel("RF Alpha:"))
        self.rf_alpha_slider = QSlider(Qt.Horizontal)
        self.rf_alpha_slider.setRange(10, 100)
        self.rf_alpha_slider.setValue(int(self.state.get("rf_alpha", 0.5) * 100))
        self.rf_alpha_slider.valueChanged.connect(self._on_display_changed)
        rf_alpha_row.addWidget(self.rf_alpha_slider)
        display_layout.addLayout(rf_alpha_row)

        color_mode_row = QHBoxLayout()
        color_mode_row.addWidget(QLabel("Electrode colors:"))
        self.electrode_color_combo = QComboBox()
        self.electrode_color_combo.addItem("Classic rainbow", "classic")
        self.electrode_color_combo.addItem("High contrast", "high_contrast")
        self.electrode_color_combo.addItem("Single color", "single")
        mode_idx = self.electrode_color_combo.findData(
            self.state.get("electrode_color_mode", "classic")
        )
        self.electrode_color_combo.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
        self.electrode_color_combo.currentIndexChanged.connect(
            self._on_electrode_color_mode_changed
        )
        color_mode_row.addWidget(self.electrode_color_combo)
        display_layout.addLayout(color_mode_row)

        color_pick_row = QHBoxLayout()
        color_pick_row.addWidget(QLabel("Color:"))
        self.electrode_color_btn = QPushButton("Pick color…")
        self.electrode_color_btn.clicked.connect(self._on_pick_electrode_color)
        self.electrode_color_btn.setEnabled(
            self.electrode_color_combo.currentData() == "single"
        )
        color_pick_row.addWidget(self.electrode_color_btn)
        display_layout.addLayout(color_pick_row)

        # VF plot eccentricity scope
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("VF Scope:"))
        self.ecc_scope_combo = QComboBox()
        for deg in [10, 15, 30]:
            self.ecc_scope_combo.addItem(f"{deg}°", deg)
        self.ecc_scope_combo.setCurrentIndex(1)  # default 15°
        self.ecc_scope_combo.currentIndexChanged.connect(self._on_ecc_scope_changed)
        scope_row.addWidget(self.ecc_scope_combo)
        display_layout.addLayout(scope_row)

        scope_slider_row = QHBoxLayout()
        scope_slider_row.addWidget(QLabel("VF Scope Bar:"))
        self.ecc_scope_slider = QSlider(Qt.Horizontal)
        self.ecc_scope_slider.setRange(5, 60)
        self.ecc_scope_slider.setValue(int(self._ecc_scope))
        self.ecc_scope_slider.setSingleStep(1)
        self.ecc_scope_slider.valueChanged.connect(self._on_ecc_scope_slider_changed)
        scope_slider_row.addWidget(self.ecc_scope_slider)
        self.ecc_scope_value_label = QLabel(f"{int(self._ecc_scope)}°")
        self.ecc_scope_value_label.setMinimumWidth(36)
        self.ecc_scope_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        scope_slider_row.addWidget(self.ecc_scope_value_label)
        display_layout.addLayout(scope_slider_row)

        view_layout.addWidget(display_group)

        instances_group = QGroupBox("Placed Implants")
        instances_layout = QVBoxLayout(instances_group)
        instances_layout.setSpacing(4)
        inst_row = QHBoxLayout()
        duplicate_btn = QPushButton("Duplicate Current Design")
        duplicate_btn.clicked.connect(self._duplicate_current_design)
        remove_inst_btn = QPushButton("Remove Selected")
        remove_inst_btn.clicked.connect(self._remove_selected_instances)
        inst_row.addWidget(duplicate_btn)
        inst_row.addWidget(remove_inst_btn)
        instances_layout.addLayout(inst_row)

        inst_row2 = QHBoxLayout()
        select_all_inst_btn = QPushButton("Select All")
        select_all_inst_btn.clicked.connect(
            lambda: self._set_instance_selection(
                [inst.instance_id for inst in self._scene_instances], additive=False
            )
        )
        hide_selected_btn = QPushButton("Hide Selected")
        hide_selected_btn.clicked.connect(self._hide_selected_instances)
        show_all_btn = QPushButton("Show All")
        show_all_btn.clicked.connect(self._show_all_instances)
        inst_row2.addWidget(select_all_inst_btn)
        inst_row2.addWidget(hide_selected_btn)
        inst_row2.addWidget(show_all_btn)
        instances_layout.addLayout(inst_row2)

        self.instance_list = QListWidget()
        self.instance_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.instance_list.itemSelectionChanged.connect(
            self._sync_selection_from_instance_list
        )
        self.instance_list.itemChanged.connect(self._sync_selection_from_instance_list)
        self.instance_list.setMinimumHeight(60)
        self.instance_list.setMaximumHeight(90)
        instances_layout.addWidget(self.instance_list)
        instance_hint = QLabel(
            "Tip: select implants here or in 3D. Use Ctrl + left-drag in the 3D view to move the selected implants."
        )
        instance_hint.setWordWrap(True)
        instance_hint.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        instances_layout.addWidget(instance_hint)

        scene_row = QHBoxLayout()
        load_scene_btn = QPushButton("Load Layout...")
        load_scene_btn.clicked.connect(self._load_scene)
        load_scene_btn.setToolTip("Load a saved Explorer multi-implant layout")
        save_scene_btn = QPushButton("Save Layout...")
        save_scene_btn.clicked.connect(self._save_scene)
        save_scene_btn.setToolTip("Save the current Explorer multi-implant layout")
        scene_row.addWidget(load_scene_btn)
        scene_row.addWidget(save_scene_btn)
        instances_layout.addLayout(scene_row)
        copy_scene_btn = QPushButton("Copy Layout to Clipboard")
        copy_scene_btn.clicked.connect(self._copy_scene_to_clipboard)
        copy_scene_btn.setToolTip("Copy the current layout JSON to clipboard")
        instances_layout.addWidget(copy_scene_btn)
        setup_layout.addWidget(instances_group)

        drag_group = QGroupBox("Implant Drag")
        drag_layout = QVBoxLayout(drag_group)
        self.drag_implant_cb = QCheckBox("Enable drag")
        self.drag_implant_cb.setChecked(self.state.get("drag_implant_enabled", False))
        self.drag_implant_cb.stateChanged.connect(self._on_drag_implant_enabled_changed)
        drag_layout.addWidget(self.drag_implant_cb)
        self.drag_requires_ctrl_cb = QCheckBox("Require Ctrl modifier")
        self.drag_requires_ctrl_cb.setChecked(
            self.state.get("drag_requires_ctrl", True)
        )
        self.drag_requires_ctrl_cb.stateChanged.connect(
            self._on_drag_requires_ctrl_changed
        )
        drag_layout.addWidget(self.drag_requires_ctrl_cb)
        drag_hint = QLabel(
            "With Ctrl required (default): plain left-drag rotates the camera, "
            "Ctrl + left-drag moves the selected implant. With Ctrl off: a left-drag "
            "starting on a selected implant moves it; a left-drag starting on empty "
            "space still rotates the camera."
        )
        drag_hint.setWordWrap(True)
        drag_hint.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        drag_layout.addWidget(drag_hint)
        self.drag_sensitivity_spin = self._add_double_spin_row(
            drag_layout,
            "Sensitivity:",
            float(self.state.get("drag_sensitivity_mm", 1.0)),
            lo=0.1,
            hi=5.0,
            decimals=2,
            step=0.05,
            callback=self._on_drag_sensitivity_changed,
        )
        setup_layout.addWidget(drag_group)

        # Comb Selection
        combs_group = QGroupBox("Probe Combs")
        combs_layout = QVBoxLayout(combs_group)

        btn_layout = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(self._select_all_combs)
        none_btn = QPushButton("None")
        none_btn.clicked.connect(self._select_no_combs)
        btn_layout.addWidget(all_btn)
        btn_layout.addWidget(none_btn)
        combs_layout.addLayout(btn_layout)

        grid = QGridLayout()
        grid.setSpacing(2)
        self.comb_checkboxes = {}
        for i in range(self.num_combs):
            cb = QCheckBox(str(i + 1))
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_comb_visibility_changed)
            grid.addWidget(cb, i // 4, i % 4)
            self.comb_checkboxes[i] = cb
        self._refresh_comb_checkbox_state()
        combs_layout.addLayout(grid)
        setup_layout.addWidget(combs_group)
        self.combs_group = combs_group

        # Implant Geometry
        implant_group = QGroupBox("Implant Geometry")
        implant_layout = QVBoxLayout(implant_group)
        load_implant_btn = QPushButton("Load Implant...")
        load_implant_btn.setToolTip("Load a JSON config from the Implant Designer")
        load_implant_btn.clicked.connect(self._load_implant_config)
        implant_layout.addWidget(load_implant_btn)
        save_template_btn = QPushButton("Save Design Template...")
        save_template_btn.setToolTip(
            "Save transformed implant template (ImplantSpec v2)"
        )
        save_template_btn.clicked.connect(self._save_design_template)
        implant_layout.addWidget(save_template_btn)
        save_snapshot_btn = QPushButton("Save Placement Snapshot...")
        save_snapshot_btn.setToolTip("Save template plus comb placement transforms")
        save_snapshot_btn.clicked.connect(self._save_placement_snapshot)
        self.save_snapshot_btn = save_snapshot_btn
        implant_layout.addWidget(save_snapshot_btn)
        self._implant_label = QLabel(self._default_implant_label)
        self._implant_label.setWordWrap(True)
        self._implant_label.setStyleSheet("color: #aaa; font-size: 10px;")
        implant_layout.addWidget(self._implant_label)
        setup_layout.addWidget(implant_group)

        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        reset_view_btn = QPushButton("Reset View")
        reset_view_btn.clicked.connect(lambda: self.plotter.reset_camera())
        action_layout.addWidget(reset_view_btn)
        reset_pos_btn = QPushButton("Reset Position")
        reset_pos_btn.clicked.connect(self._reset_position)
        action_layout.addWidget(reset_pos_btn)
        setup_layout.addWidget(action_group)

        # Info
        self.info_label = QLabel("Ready")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "background-color: #2d2d2d; padding: 10px; border-radius: 5px;"
        )
        self.info_label.setMaximumHeight(180)
        view_layout.addWidget(self.info_label)

        setup_layout.addStretch()
        analysis_layout_root.addStretch()
        view_layout.addStretch()
        self._sync_manip_controls_from_state()
        self._sync_instance_list_from_state()
        self._sync_instance_controls_from_selection()
        scroll.setWidget(panel)
        container_layout.addWidget(scroll)
        return scroll_container

    def _ensure_credit_overlay(self):
        if not hasattr(self, "plotter"):
            return
        try:
            self.plotter.add_text(
                self._credit_text,
                position="lower_right",
                color="white",
                font_size=9,
                shadow=False,
                name="credit_overlay",
            )
        except Exception:
            pass

    def _load_implant_config(self, path=None):
        """Load an implant JSON file into Explorer's v2 runtime."""
        # QPushButton.clicked passes a bool; treat it as "no explicit path".
        if isinstance(path, bool):
            path = None
        if path is not None:
            path = str(path).strip()
            if not path:
                path = None

        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Implant", "", "JSON Files (*.json)"
            )
            if not path:
                return

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            self._show_warning_message("Load failed", str(exc))
            return

        if str(raw.get("schema_version", "")) == "explorer_scene_v1":
            self._load_scene(path)
            return

        self._load_implant_spec_v2(path, raw)

    def _load_implant_spec_v2(self, path: str, raw_dict: dict | None = None):
        """Load an ImplantSpec JSON as the current active design revision."""
        spec = (
            ImplantSpec.from_dict(raw_dict)
            if raw_dict is not None
            else load_spec_json(path)
        )
        self._push_undo_state()
        self._loaded_implant_path = path
        self._loaded_implant_spec_v2 = spec
        self._active_design_spec_v2 = spec
        self._instance_mode_enabled = True

        pts_mm, mapping = self._spec_contacts_with_index(spec)
        if pts_mm.size == 0:
            return

        px = float(self.mm_per_pixel[0])
        self.template_contacts_vox = np.asarray(pts_mm, dtype=np.float64) / px
        self.electrodes_per_comb = self.template_contacts_vox.shape[0]
        self._template_contact_index = list(mapping)
        self.entry_electrode_idx = 0
        self._nonspike_anchor_vox = self._default_nonspike_anchor_vox()

        # Compatibility aliases
        self.base_comb_xy = np.column_stack(
            [
                self.template_contacts_vox[:, 1],
                self.template_contacts_vox[:, 2],
            ]
        )
        self.rotated_base_comb = self.base_comb_xy.copy()

        self._apply_default_design_transform_for_spec(spec)
        self._apply_loaded_transform_meta(spec)
        revision_id = self._register_design_revision(spec, source_path=path)
        self._activate_design_revision(revision_id)
        self._apply_loaded_revision_to_current_instances(revision_id, spec.name)
        self._apply_loaded_placement_snapshot(spec)
        self._update_template_transform()

        self._clear_all_electrode_actors()
        self._refresh_comb_checkbox_state()
        QTimer.singleShot(0, self._apply_implant_config_visual_update)

        family_label = spec.family.replace("multishank", "multi-shank")
        self._implant_label.setText(
            f"{spec.name} [{family_label}]\n"
            f"{spec.total_contacts()} contacts, schema v{spec.schema_version}"
        )
        print(
            f"[Explorer] Loaded v2 spec: {spec.name} "
            f"(family={spec.family}, {spec.total_contacts()} contacts)"
        )

    def _safe_remove_actor(self, name):
        try:
            self.plotter.remove_actor(name)
        except Exception:
            pass

    def _clear_all_electrode_actors(self):
        """Remove all electrode VTK actors and cached clouds.

        Called when switching implant designs so stale actors from a previous
        design (which may have had a different number of combs) are cleaned up.
        """
        for name in list(self.electrode_actors):
            self._safe_remove_actor(name)
        self.electrode_actors.clear()
        self._electrode_clouds.clear()
        self._electrode_halos.clear()
        self._last_render_contacts_by_instance = {}
        self._last_render_instance_ids = []

    def _apply_implant_config_visual_update(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Implant geometry changes only affect electrodes/RF sampling.
            self._needs_brain_update = False
            self._needs_anatomy_update = False
            self._needs_electrode_update = True
            self._sync_instance_list_from_state()
            self._sync_instance_controls_from_selection()
            self.update_all()
        finally:
            QApplication.restoreOverrideCursor()

    def _snap_value(self, val, step):
        if not self.state.get("snap_enabled", False):
            return float(val)
        try:
            st = float(step)
        except Exception:
            return float(val)
        if st <= 0:
            return float(val)
        return round(float(val) / st) * st

    def _selected_comb_indices(self):
        return [i for i, selected in self.state["show_combs"].items() if selected]

    def _scene_anchor_mm(self):
        if self._nonspike_anchor_vox is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self._nonspike_anchor_vox, dtype=np.float64) * float(
            self.mm_per_pixel[0]
        )

    def _instance_mode_active(self):
        return bool(self._instance_mode_enabled)

    def _current_design_name(self):
        rev = self._design_revisions.get(self._active_design_revision_id, {})
        return str(rev.get("label", "Implant"))

    def _rebuild_instance_index(self):
        self._instance_index_by_id = {
            inst.instance_id: inst for inst in self._scene_instances
        }
        self._selected_instance_ids = {
            inst.instance_id for inst in self._scene_instances if inst.selected
        }

    def _selected_instance_ids_list(self):
        return [inst.instance_id for inst in self._scene_instances if inst.selected]

    def _selected_instances(self):
        return [inst for inst in self._scene_instances if inst.selected]

    def _instance_for_id(self, instance_id):
        return self._instance_index_by_id.get(str(instance_id))

    def _set_instance_selection(self, instance_ids, *, additive=False, toggle=False):
        ids = {str(v) for v in instance_ids}
        if not additive and not toggle:
            for inst in self._scene_instances:
                inst.selected = inst.instance_id in ids
        elif toggle:
            for inst in self._scene_instances:
                if inst.instance_id in ids:
                    inst.selected = not bool(inst.selected)
        else:
            for inst in self._scene_instances:
                if inst.instance_id in ids:
                    inst.selected = True
        self._rebuild_instance_index()
        self._sync_instance_list_from_state()
        self._sync_instance_controls_from_selection()

    def _clear_instance_selection(self):
        for inst in self._scene_instances:
            inst.selected = False
        self._rebuild_instance_index()
        self._sync_instance_list_from_state()
        self._sync_instance_controls_from_selection()

    def _instance_color(self, idx):
        if not hasattr(self, "probe_colors") or not self.probe_colors:
            return "#00D4FF"
        return self.probe_colors[idx % len(self.probe_colors)]

    def _pivot_from_contacts_mm(self, contacts_mm, design_transform, entry_index=0):
        pts = np.asarray(contacts_mm, dtype=np.float64)
        if pts.size == 0:
            return np.zeros(3, dtype=np.float64)
        mode = str(dict(design_transform or {}).get("pivot_mode", "entry"))
        if mode == "centroid":
            return pts.mean(axis=0)
        if mode == "custom":
            piv = list(
                dict(design_transform or {}).get("pivot_custom_mm", [0.0, 0.0, 0.0])
            )
            return np.asarray((piv + [0.0, 0.0, 0.0])[:3], dtype=np.float64)
        idx = int(min(max(0, entry_index), pts.shape[0] - 1))
        return pts[idx]

    def _clone_design_transform(self, transform=None):
        src = dict(transform or self.design_transform or {})
        piv = list(src.get("pivot_custom_mm", [0.0, 0.0, 0.0]))
        return {
            "tx_mm": float(src.get("tx_mm", 0.0)),
            "ty_mm": float(src.get("ty_mm", 0.0)),
            "tz_mm": float(src.get("tz_mm", 0.0)),
            "rx_deg": float(src.get("rx_deg", 0.0)),
            "ry_deg": float(src.get("ry_deg", 0.0)),
            "rz_deg": float(src.get("rz_deg", 0.0)),
            "scale": float(src.get("scale", 1.0)),
            "mirror_x": bool(src.get("mirror_x", False)),
            "pivot_mode": str(src.get("pivot_mode", "entry")),
            "pivot_custom_mm": [float(v) for v in (piv + [0.0, 0.0, 0.0])[:3]],
        }

    def _default_design_transform_for_spec(self, spec=None):
        transform = self._clone_design_transform({})
        family = str(getattr(spec, "family", "") or "").strip().lower()
        if family == "utah":
            # Stand Utah arrays in the X/Z wall plane by default.
            transform["rx_deg"] = -90.0
        return transform

    def _apply_default_design_transform_for_spec(self, spec=None):
        self.design_transform = self._default_design_transform_for_spec(spec)
        self._last_ui_design = {
            "rot_x": float(self.design_transform.get("rx_deg", 0.0)),
            "rot_y": float(self.design_transform.get("ry_deg", 0.0)),
            "rot_z": float(self.design_transform.get("rz_deg", 0.0)),
            "scale": float(self.design_transform.get("scale", 1.0)),
        }

    def _build_transform_meta(self, design_transform, *, source_path=""):
        dtx = self._clone_design_transform(design_transform)
        return {
            "model": "hybrid_v1",
            "pivot_mode": str(dtx.get("pivot_mode", "entry")),
            "snap": {
                "mm": float(self.state.get("snap_mm", 0.1)),
                "deg": float(self.state.get("snap_deg", 1.0)),
                "scale": float(self.state.get("snap_scale", 0.01)),
                "enabled": bool(self.state.get("snap_enabled", False)),
            },
            "current_design": {
                "tx_mm": float(dtx.get("tx_mm", 0.0)),
                "ty_mm": float(dtx.get("ty_mm", 0.0)),
                "tz_mm": float(dtx.get("tz_mm", 0.0)),
                "rx_deg": float(dtx.get("rx_deg", 0.0)),
                "ry_deg": float(dtx.get("ry_deg", 0.0)),
                "rz_deg": float(dtx.get("rz_deg", 0.0)),
                "scale": float(dtx.get("scale", 1.0)),
                "mirror_x": bool(dtx.get("mirror_x", False)),
            },
            "history": [],
            "source": {
                "app": "explorer_v2",
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "parent_path": str(source_path or ""),
            },
        }

    def _design_transform_from_spec(self, spec):
        tm = dict(getattr(spec, "transform_meta", {}) or {})
        cd = dict(tm.get("current_design", {}) or {})
        out = self._clone_design_transform({})
        if cd:
            out.update(
                {
                    "tx_mm": float(cd.get("tx_mm", 0.0)),
                    "ty_mm": float(cd.get("ty_mm", 0.0)),
                    "tz_mm": float(cd.get("tz_mm", 0.0)),
                    "rx_deg": float(cd.get("rx_deg", 0.0)),
                    "ry_deg": float(cd.get("ry_deg", 0.0)),
                    "rz_deg": float(cd.get("rz_deg", 0.0)),
                    "scale": float(cd.get("scale", 1.0)),
                    "mirror_x": bool(cd.get("mirror_x", False)),
                    "pivot_mode": str(tm.get("pivot_mode", "entry")),
                }
            )
        return out

    def _revision_export_spec(self, revision_id):
        rev = self._design_revisions.get(str(revision_id))
        if rev is None:
            return None
        base_spec = ImplantSpec.from_dict(dict(rev["spec_dict"]))
        template_mm = np.asarray(
            rev["template_contacts_transformed_vox"], dtype=np.float64
        ) * float(self.mm_per_pixel[0])
        base_spec.contacts_local = np.asarray(template_mm, dtype=np.float64)
        base_spec.transform_meta = self._build_transform_meta(
            rev["design_transform"], source_path=rev.get("source_path", "")
        )
        base_spec.placement_snapshot = {}
        return base_spec

    def _register_design_revision(self, spec, *, source_path="", design_transform=None):
        spec_dict = spec.to_dict()
        pts_mm, mapping = self._spec_contacts_with_index(spec)
        px = float(self.mm_per_pixel[0])
        base_vox = np.asarray(pts_mm, dtype=np.float64) / px
        transform = self._clone_design_transform(
            design_transform
            if design_transform is not None
            else getattr(self, "design_transform", {})
        )
        pivot_mm = self._pivot_from_contacts_mm(
            np.asarray(pts_mm, dtype=np.float64), transform, entry_index=0
        )
        transformed_mm = apply_transform_to_contacts(
            np.asarray(pts_mm, dtype=np.float64),
            tx_mm=float(transform.get("tx_mm", 0.0)),
            ty_mm=float(transform.get("ty_mm", 0.0)),
            tz_mm=float(transform.get("tz_mm", 0.0)),
            rx_deg=float(transform.get("rx_deg", 0.0)),
            ry_deg=float(transform.get("ry_deg", 0.0)),
            rz_deg=float(transform.get("rz_deg", 0.0)),
            scale=float(transform.get("scale", 1.0)),
            mirror_x=bool(transform.get("mirror_x", False)),
            pivot_mm=pivot_mm,
        )
        revision_id = uuid.uuid4().hex[:12]
        self._design_revisions[revision_id] = {
            "revision_id": revision_id,
            "label": str(spec.name),
            "source_path": str(source_path or ""),
            "spec_dict": spec_dict,
            "base_contacts_vox": base_vox,
            "template_contacts_transformed_vox": np.asarray(
                transformed_mm, dtype=np.float64
            )
            / px,
            "template_contact_index": list(mapping),
            "entry_electrode_idx": 0,
            "design_transform": transform,
        }
        return revision_id

    def _activate_design_revision(self, revision_id):
        rev = self._design_revisions.get(str(revision_id))
        if rev is None:
            return
        self._active_design_revision_id = str(revision_id)
        self._instance_mode_enabled = True
        self._loaded_implant_path = rev.get("source_path", "")
        spec_dict = dict(rev.get("spec_dict", {}) or {})
        self._loaded_implant_spec_v2 = (
            ImplantSpec.from_dict(spec_dict) if spec_dict else None
        )
        self._active_design_spec_v2 = self._loaded_implant_spec_v2
        self.template_contacts_vox = np.asarray(
            rev["base_contacts_vox"], dtype=np.float64
        ).copy()
        self.template_contacts_transformed_vox = np.asarray(
            rev["template_contacts_transformed_vox"], dtype=np.float64
        ).copy()
        self._template_contact_index = list(rev["template_contact_index"])
        self.electrodes_per_comb = int(self.template_contacts_transformed_vox.shape[0])
        self.entry_electrode_idx = int(rev.get("entry_electrode_idx", 0))
        self.design_transform = self._clone_design_transform(rev["design_transform"])
        self._last_ui_design = {
            "rot_x": float(self.design_transform.get("rx_deg", 0.0)),
            "rot_y": float(self.design_transform.get("ry_deg", 0.0)),
            "rot_z": float(self.design_transform.get("rz_deg", 0.0)),
            "scale": float(self.design_transform.get("scale", 1.0)),
        }
        self._sync_manip_controls_from_state()
        if hasattr(self, "_implant_label"):
            self._implant_label.setText(
                f"{rev.get('label', 'Implant')}\n{self.electrodes_per_comb} contacts/template"
            )

    def _create_instance(
        self,
        *,
        revision_id=None,
        label=None,
        placement=None,
        visible=True,
        selected=True,
    ):
        rev_id = str(revision_id or self._active_design_revision_id)
        if not rev_id:
            return None
        place = normalize_instance_placement(placement)
        if selected:
            for other in self._scene_instances:
                other.selected = False
        inst = ExplorerImplantInstance(
            instance_id=uuid.uuid4().hex[:12],
            label=str(
                label
                or f"{self._design_revisions[rev_id]['label']} #{len(self._scene_instances) + 1}"
            ),
            design_revision_id=rev_id,
            visible=bool(visible),
            selected=bool(selected),
            placement=place,
        )
        self._scene_instances.append(inst)
        self._rebuild_instance_index()
        self._sync_instance_list_from_state()
        self._sync_instance_controls_from_selection()
        return inst

    def _apply_loaded_revision_to_current_instances(self, revision_id, design_name):
        rev_id = str(revision_id or "")
        if not rev_id:
            return
        targets = self._selected_instances()
        if not targets and self._scene_instances:
            targets = [self._scene_instances[0]]
        if not targets:
            self._create_instance(revision_id=rev_id, selected=True)
            return
        for inst in targets:
            inst.design_revision_id = rev_id
            try:
                slot_index = self._scene_instances.index(inst) + 1
            except ValueError:
                slot_index = 1
            inst.label = f"{design_name} #{slot_index}"
        self._rebuild_instance_index()
        self._sync_instance_list_from_state()
        self._sync_instance_controls_from_selection()

    def _duplicate_current_design(self):
        if not self._active_design_revision_id:
            return
        self._push_undo_state()
        selected = self._selected_instances()
        if selected:
            base = normalize_instance_placement(selected[0].placement)
            base["tx_mm"] += 1.0
            base["ty_mm"] += 1.0
        else:
            n = len(self._scene_instances)
            base = normalize_instance_placement(
                {
                    "tx_mm": float(n) * 0.5,
                    "ty_mm": float(n) * 0.5,
                    "tz_mm": 0.0,
                    "rx_deg": 0.0,
                    "ry_deg": 0.0,
                    "rz_deg": 0.0,
                }
            )
        for other in self._scene_instances:
            other.selected = False
        self._create_instance(
            revision_id=self._active_design_revision_id, placement=base, selected=True
        )
        self._needs_electrode_update = True
        self.update_all()

    def _remove_selected_instances(self):
        selected_ids = set(self._selected_instance_ids_list())
        if not selected_ids:
            return
        self._push_undo_state()
        self._scene_instances = [
            inst
            for inst in self._scene_instances
            if inst.instance_id not in selected_ids
        ]
        if self._scene_instances:
            self._scene_instances[0].selected = True
        self._rebuild_instance_index()
        self._sync_instance_list_from_state()
        self._clear_all_electrode_actors()
        self._needs_electrode_update = True
        self.update_all()

    def _show_all_instances(self):
        for inst in self._scene_instances:
            inst.visible = True
        self._sync_instance_list_from_state()
        self._needs_electrode_update = True
        self.update_all()

    def _hide_selected_instances(self):
        changed = False
        for inst in self._selected_instances():
            inst.visible = False
            changed = True
        if not changed:
            return
        self._sync_instance_list_from_state()
        self._needs_electrode_update = True
        self.update_all()

    def _apply_instance_transform_delta(
        self, *, d_tx=0.0, d_ty=0.0, d_tz=0.0, d_rx=0.0, d_ry=0.0, d_rz=0.0
    ):
        selected = self._selected_instances()
        if not selected:
            return False
        for inst in selected:
            place = normalize_instance_placement(inst.placement)
            place["tx_mm"] += float(d_tx)
            place["ty_mm"] += float(d_ty)
            place["tz_mm"] += float(d_tz)
            place["rx_deg"] += float(d_rx)
            place["ry_deg"] += float(d_ry)
            place["rz_deg"] += float(d_rz)
            inst.placement = place
        self._rebuild_instance_index()
        return True

    def _sync_instance_list_from_state(self):
        if not hasattr(self, "instance_list"):
            return
        self.instance_list.blockSignals(True)
        self.instance_list.clear()
        selected_row = -1
        for idx, inst in enumerate(self._scene_instances):
            item = QListWidgetItem(inst.label)
            item.setData(Qt.UserRole, inst.instance_id)
            item.setFlags(
                item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsSelectable
                | Qt.ItemIsEnabled
            )
            item.setCheckState(Qt.Checked if inst.visible else Qt.Unchecked)
            item.setForeground(QColor("white") if inst.selected else QColor("#d0d0d0"))
            if inst.selected:
                item.setSelected(True)
                selected_row = idx
            color = self._instance_color(idx)
            item.setBackground(
                self._rgb01_to_qcolor(color, alpha=0.32)
                if inst.selected
                else QColor(0, 0, 0, 0)
            )
            self.instance_list.addItem(item)
        if self.instance_list.count() > 0:
            if selected_row < 0:
                selected_row = 0
                self._scene_instances[0].selected = True
                self._rebuild_instance_index()
            self.instance_list.setCurrentRow(selected_row)
            item = self.instance_list.item(selected_row)
            if item is not None:
                item.setSelected(True)
        self.instance_list.blockSignals(False)
        if hasattr(self, "save_snapshot_btn"):
            allow_snapshot = (not self._instance_mode_active()) or (
                len(self._scene_instances) <= 1
            )
            self.save_snapshot_btn.setEnabled(allow_snapshot)

    def _sync_selection_from_instance_list(self):
        if not hasattr(self, "instance_list"):
            return
        selected_ids = set()
        for item in self.instance_list.selectedItems():
            selected_ids.add(str(item.data(Qt.UserRole)))
        for row in range(self.instance_list.count()):
            item = self.instance_list.item(row)
            inst = self._instance_for_id(item.data(Qt.UserRole))
            if inst is None:
                continue
            inst.selected = inst.instance_id in selected_ids
            inst.visible = item.checkState() == Qt.Checked
        self._rebuild_instance_index()
        self._needs_electrode_update = True
        self._sync_instance_controls_from_selection()
        self.update_all()

    def _instance_contacts_vox(self, inst):
        rev = self._design_revisions.get(inst.design_revision_id)
        if rev is None:
            return np.empty((0, 3), dtype=np.float64)
        template_mm = np.asarray(
            rev["template_contacts_transformed_vox"], dtype=np.float64
        ) * float(self.mm_per_pixel[0])
        placed_mm = transform_instance_contacts_mm(
            template_mm,
            inst.placement,
            self._scene_anchor_mm(),
            entry_index=int(rev.get("entry_electrode_idx", 0)),
        )
        return placed_mm / float(self.mm_per_pixel[0])

    def _instance_hit_threshold_vox(self):
        return max(1.5, float(self.state.get("electrode_dot_size", 4)) * 0.25)

    def _hit_test_instance(self, world_xyz):
        return hit_test_instance_contacts(
            world_xyz,
            self._last_render_contacts_by_instance,
            self._instance_hit_threshold_vox(),
        )

    def _capture_transform_state(self):
        return {
            "comb_offsets_mm": copy.deepcopy(self.comb_offsets_mm),
            "_last_ui_move_mm": copy.deepcopy(self._last_ui_move_mm),
            "design_transform": copy.deepcopy(self.design_transform),
            "_last_ui_design": copy.deepcopy(self._last_ui_design),
            "_last_ui_instance_rot": copy.deepcopy(self._last_ui_instance_rot),
            "_scene_instances": copy.deepcopy(self._scene_instances),
            "_design_revisions": copy.deepcopy(self._design_revisions),
            "_active_design_revision_id": str(self._active_design_revision_id),
            "ui": {
                "depth": float(self.depth_spin.value()),
                "x": float(self.x_spin.value()),
                "y": float(self.y_spin.value()),
                "z": float(self.z_spin.value()),
                "place_rot_x": (
                    float(self.place_rot_x_spin.value())
                    if hasattr(self, "place_rot_x_spin")
                    else 0.0
                ),
                "place_rot_y": (
                    float(self.place_rot_y_spin.value())
                    if hasattr(self, "place_rot_y_spin")
                    else 0.0
                ),
                "place_rot_z": (
                    float(self.place_rot_z_spin.value())
                    if hasattr(self, "place_rot_z_spin")
                    else 0.0
                ),
                "rot_x": (
                    float(self.rot_x_spin.value())
                    if hasattr(self, "rot_x_spin")
                    else 0.0
                ),
                "rot_y": (
                    float(self.rot_y_spin.value())
                    if hasattr(self, "rot_y_spin")
                    else 0.0
                ),
                "rot_z": (
                    float(self.rot_z_spin.value())
                    if hasattr(self, "rot_z_spin")
                    else 0.0
                ),
                "scale": (
                    float(self.scale_spin.value())
                    if hasattr(self, "scale_spin")
                    else 1.0
                ),
                "mirror_x": bool(self.state.get("mirror_x", False)),
            },
        }

    def _restore_transform_state(self, snap):
        self.comb_offsets_mm = copy.deepcopy(snap["comb_offsets_mm"])
        self._last_ui_move_mm = copy.deepcopy(snap["_last_ui_move_mm"])
        self.design_transform = copy.deepcopy(snap["design_transform"])
        self._last_ui_design = copy.deepcopy(snap["_last_ui_design"])
        self._last_ui_instance_rot = copy.deepcopy(
            snap.get("_last_ui_instance_rot", self._last_ui_instance_rot)
        )
        self._scene_instances = copy.deepcopy(
            snap.get("_scene_instances", self._scene_instances)
        )
        self._design_revisions = copy.deepcopy(
            snap.get("_design_revisions", self._design_revisions)
        )
        self._rebuild_instance_index()
        active_rev = str(
            snap.get("_active_design_revision_id", self._active_design_revision_id)
        )
        if active_rev:
            self._activate_design_revision(active_rev)

        self._suppress_value_changed = True
        self._suppress_manip_value_changed = True
        ui = snap.get("ui", {})
        self.depth_spin.setValue(float(ui.get("depth", 0.0)))
        self.x_spin.setValue(float(ui.get("x", 0.0)))
        self.y_spin.setValue(float(ui.get("y", 0.0)))
        self.z_spin.setValue(float(ui.get("z", 0.0)))
        if hasattr(self, "place_rot_x_spin"):
            self.place_rot_x_spin.setValue(float(ui.get("place_rot_x", 0.0)))
            self.place_rot_y_spin.setValue(float(ui.get("place_rot_y", 0.0)))
            self.place_rot_z_spin.setValue(float(ui.get("place_rot_z", 0.0)))
        if hasattr(self, "rot_x_spin"):
            self.rot_x_spin.setValue(float(ui.get("rot_x", 0.0)))
            self.rot_y_spin.setValue(float(ui.get("rot_y", 0.0)))
            self.rot_z_spin.setValue(float(ui.get("rot_z", 0.0)))
            self.scale_spin.setValue(float(ui.get("scale", 1.0)))
        self._suppress_value_changed = False
        self._suppress_manip_value_changed = False
        self._sync_manip_controls_from_state()
        self._update_template_transform()
        self._sync_instance_list_from_state()

        self._needs_electrode_update = True
        self.update_all()

    def _push_undo_state(self):
        self._undo_stack.append(self._capture_transform_state())
        if len(self._undo_stack) > 120:
            self._undo_stack = self._undo_stack[-120:]
        self._redo_stack.clear()

    def _on_undo_clicked(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._capture_transform_state())
        snap = self._undo_stack.pop()
        self._restore_transform_state(snap)

    def _on_redo_clicked(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._capture_transform_state())
        snap = self._redo_stack.pop()
        self._restore_transform_state(snap)

    def _pivot_from_template_mm(self, template_mm):
        if template_mm.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        mode = self.design_transform.get("pivot_mode", "entry")
        if mode == "centroid":
            return template_mm.mean(axis=0)
        if mode == "custom":
            return np.asarray(
                self.design_transform.get("pivot_custom_mm", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )

        idx = int(min(max(0, self.entry_electrode_idx), template_mm.shape[0] - 1))
        return template_mm[idx]

    def _update_template_transform(self):
        if not hasattr(self, "template_contacts_vox"):
            return
        px = float(self.mm_per_pixel[0])
        base_mm = self.template_contacts_vox * px
        pivot_mm = self._pivot_from_template_mm(base_mm)
        transformed_mm = apply_transform_to_contacts(
            base_mm,
            tx_mm=float(self.design_transform.get("tx_mm", 0.0)),
            ty_mm=float(self.design_transform.get("ty_mm", 0.0)),
            tz_mm=float(self.design_transform.get("tz_mm", 0.0)),
            rx_deg=float(self.design_transform.get("rx_deg", 0.0)),
            ry_deg=float(self.design_transform.get("ry_deg", 0.0)),
            rz_deg=float(self.design_transform.get("rz_deg", 0.0)),
            scale=float(self.design_transform.get("scale", 1.0)),
            mirror_x=bool(self.design_transform.get("mirror_x", False)),
            pivot_mm=pivot_mm,
        )
        self.template_contacts_transformed_vox = transformed_mm / px
        rev = self._design_revisions.get(self._active_design_revision_id)
        if rev is not None:
            rev["base_contacts_vox"] = np.asarray(
                self.template_contacts_vox, dtype=np.float64
            ).copy()
            rev["template_contacts_transformed_vox"] = np.asarray(
                self.template_contacts_transformed_vox, dtype=np.float64
            ).copy()
            rev["template_contact_index"] = list(self._template_contact_index)
            rev["entry_electrode_idx"] = int(self.entry_electrode_idx)
            rev["design_transform"] = self._clone_design_transform(
                self.design_transform
            )

    def _sync_manip_controls_from_state(self):
        self.state["rot_x"] = float(self.design_transform.get("rx_deg", 0.0))
        self.state["rot_y"] = float(self.design_transform.get("ry_deg", 0.0))
        self.state["rot_z"] = float(self.design_transform.get("rz_deg", 0.0))
        self.state["scale"] = float(self.design_transform.get("scale", 1.0))
        self.state["mirror_x"] = bool(self.design_transform.get("mirror_x", False))
        self.state["pivot_mode"] = str(self.design_transform.get("pivot_mode", "entry"))
        piv = self.design_transform.get("pivot_custom_mm", [0.0, 0.0, 0.0])
        self.state["pivot_x"] = float(piv[0]) if len(piv) > 0 else 0.0
        self.state["pivot_y"] = float(piv[1]) if len(piv) > 1 else 0.0
        self.state["pivot_z"] = float(piv[2]) if len(piv) > 2 else 0.0
        if hasattr(self, "rot_x_spin"):
            self._suppress_manip_value_changed = True
            self.rot_x_spin.setValue(self.state["rot_x"])
            self.rot_y_spin.setValue(self.state["rot_y"])
            self.rot_z_spin.setValue(self.state["rot_z"])
            self.scale_spin.setValue(self.state["scale"])
            idx = self.pivot_combo.findData(self.state["pivot_mode"])
            self.pivot_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.pivot_x_spin.setValue(self.state["pivot_x"])
            self.pivot_y_spin.setValue(self.state["pivot_y"])
            self.pivot_z_spin.setValue(self.state["pivot_z"])
            if hasattr(self, "snap_cb"):
                self.snap_cb.setChecked(bool(self.state.get("snap_enabled", True)))
                self.snap_mm_spin.setValue(float(self.state.get("snap_mm", 0.1)))
                self.snap_deg_spin.setValue(float(self.state.get("snap_deg", 1.0)))
                self.snap_scale_spin.setValue(float(self.state.get("snap_scale", 0.01)))
            self._suppress_manip_value_changed = False
        if hasattr(self, "mirror_btn"):
            self.mirror_btn.setText(
                "Mirror X: ON" if self.state["mirror_x"] else "Mirror X: OFF"
            )

    def _toggle_mirror_x(self):
        self._push_undo_state()
        self.design_transform["mirror_x"] = not bool(
            self.design_transform.get("mirror_x", False)
        )
        self._sync_manip_controls_from_state()
        self._update_template_transform()
        self._needs_electrode_update = True
        self.update_all()

    def _reset_manipulation(self):
        self._push_undo_state()
        self._apply_default_design_transform_for_spec(
            getattr(self, "_active_design_spec_v2", None)
        )
        self._suppress_manip_value_changed = True
        if hasattr(self, "rot_x_spin"):
            self.rot_x_spin.setValue(float(self.design_transform.get("rx_deg", 0.0)))
            self.rot_y_spin.setValue(float(self.design_transform.get("ry_deg", 0.0)))
            self.rot_z_spin.setValue(float(self.design_transform.get("rz_deg", 0.0)))
            self.scale_spin.setValue(float(self.design_transform.get("scale", 1.0)))
            self.pivot_combo.setCurrentIndex(self.pivot_combo.findData("entry"))
            self.pivot_x_spin.setValue(0.0)
            self.pivot_y_spin.setValue(0.0)
            self.pivot_z_spin.setValue(0.0)
        self._suppress_manip_value_changed = False
        self._sync_manip_controls_from_state()
        self._update_template_transform()
        self._needs_electrode_update = True
        self.update_all()

    def _build_export_spec(self, include_snapshot=False):
        spec = self._revision_export_spec(self._active_design_revision_id)
        if spec is None:
            spec = ImplantSpec(name="Explorer Implant", family="multishank")
        if include_snapshot:
            comb_transforms = []
            if self._instance_mode_active():
                if len(self._scene_instances) > 1:
                    raise ValueError("Use Save Scene for multi-instance layouts.")
                if self._scene_instances:
                    inst = self._scene_instances[0]
                    place = normalize_instance_placement(inst.placement)
                    comb_transforms.append(
                        {
                            "comb_index": 0,
                            "tx_mm": float(place.get("tx_mm", 0.0)),
                            "ty_mm": float(place.get("ty_mm", 0.0)),
                            "tz_mm": float(place.get("tz_mm", 0.0)),
                            "rx_deg": float(place.get("rx_deg", 0.0)),
                            "ry_deg": float(place.get("ry_deg", 0.0)),
                            "rz_deg": float(place.get("rz_deg", 0.0)),
                            "scale": 1.0,
                            "depth_mm": 0.0,
                            "x_mm": float(place.get("tx_mm", 0.0)),
                            "y_mm": float(place.get("ty_mm", 0.0)),
                            "z_mm": float(place.get("tz_mm", 0.0)),
                        }
                    )
            else:
                for i in range(self.num_combs):
                    offs = self.comb_offsets_mm.get(
                        i, {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
                    )
                    comb_transforms.append(
                        {
                            "comb_index": int(i),
                            "tx_mm": float(offs.get("x", 0.0)),
                            "ty_mm": float(offs.get("y", 0.0)),
                            "tz_mm": float(offs.get("z", 0.0)),
                            "rx_deg": 0.0,
                            "ry_deg": 0.0,
                            "rz_deg": 0.0,
                            "scale": 1.0,
                            "depth_mm": float(offs.get("depth", 0.0)),
                            "x_mm": float(offs.get("x", 0.0)),
                            "y_mm": float(offs.get("y", 0.0)),
                            "z_mm": float(offs.get("z", 0.0)),
                        }
                    )
            spec.placement_snapshot = {
                "dataset_id": str(self.dataset_id),
                "subject_id": str(self.human_subject_id),
                "coordinate_frame": str(getattr(self, "_coord_frame", "voxel_ras_v1")),
                "orientation": str(getattr(self, "_orientation_canonical", "RAS")),
                "comb_transforms": comb_transforms,
            }
        else:
            spec.placement_snapshot = {}
        return spec

    def _save_design_template(self):
        spec = self._build_export_spec(include_snapshot=False)
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Design Template", "", "JSON Files (*.json)", options=opts
        )
        if not path:
            return
        save_spec_json(spec, path)
        self._show_info_message("Saved", f"Design template saved:\n{path}")

    def _save_placement_snapshot(self):
        try:
            spec = self._build_export_spec(include_snapshot=True)
        except ValueError as exc:
            self._show_warning_message("Save Placement Snapshot", str(exc))
            return
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Placement Snapshot", "", "JSON Files (*.json)", options=opts
        )
        if not path:
            return
        save_spec_json(spec, path)
        self._show_info_message("Saved", f"Placement snapshot saved:\n{path}")

    def _build_scene_config(self):
        revisions = []
        for revision_id, rev in self._design_revisions.items():
            spec = self._revision_export_spec(revision_id)
            if spec is None:
                continue
            revisions.append(
                ExplorerDesignRevision(
                    revision_id=str(revision_id),
                    label=str(rev.get("label", spec.name)),
                    design_payload=spec.to_dict(),
                    source_path=str(rev.get("source_path", "")),
                )
            )
        return ExplorerScene(
            dataset_context={
                "dataset_id": str(self.dataset_id),
                "subject_id": str(self.human_subject_id),
                "coordinate_frame": str(getattr(self, "_coord_frame", "voxel_ras_v1")),
                "orientation": str(getattr(self, "_orientation_canonical", "RAS")),
            },
            active_design_revision_id=str(self._active_design_revision_id),
            design_revisions=revisions,
            instances=copy.deepcopy(self._scene_instances),
        )

    def _save_scene(self):
        scene = self._build_scene_config()
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Explorer Layout", "", "JSON Files (*.json)", options=opts
        )
        if not path:
            return
        scene.save(path)
        self._show_info_message("Saved", f"Explorer layout saved:\n{path}")

    def _copy_scene_to_clipboard(self):
        """Desktop equivalent of the web 'Copy Share Link' button: dump the
        current layout JSON to the system clipboard."""
        try:
            scene = self._build_scene_config()
            payload = json.dumps(scene.to_dict(), indent=2)
        except Exception as exc:
            self._show_warning_message("Copy Layout", f"Could not build layout: {exc}")
            return
        try:
            QApplication.clipboard().setText(payload)
        except Exception as exc:
            self._show_warning_message("Copy Layout", f"Clipboard unavailable: {exc}")
            return
        self._show_info_message(
            "Copied",
            f"Layout JSON copied to clipboard ({len(payload):,} chars).",
        )

    def _load_scene(self, path=None):
        if isinstance(path, bool):
            path = None
        if not path:
            opts = QFileDialog.Options()
            opts |= QFileDialog.DontUseNativeDialog
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Explorer Layout", "", "JSON Files (*.json)", options=opts
            )
            if not path:
                return
        try:
            scene = ExplorerScene.load(path)
        except Exception as exc:
            self._load_implant_config(path)
            return

        if scene.schema_version != "explorer_scene_v1" or not scene.design_revisions:
            self._load_implant_config(path)
            return

        self._push_undo_state()
        self._design_revisions.clear()
        self._scene_instances = []
        for rev in scene.design_revisions:
            spec = ImplantSpec.from_dict(dict(rev.design_payload))
            revision_id = self._register_design_revision(
                spec,
                source_path=rev.source_path,
                design_transform=self._design_transform_from_spec(spec),
            )
            if revision_id != rev.revision_id:
                self._design_revisions[rev.revision_id] = self._design_revisions.pop(
                    revision_id
                )
                self._design_revisions[rev.revision_id]["revision_id"] = rev.revision_id
        self._scene_instances = [copy.deepcopy(inst) for inst in scene.instances]
        self._rebuild_instance_index()
        active = scene.active_design_revision_id or (
            scene.design_revisions[0].revision_id if scene.design_revisions else ""
        )
        if active:
            self._activate_design_revision(active)
        self._sync_instance_list_from_state()
        self._clear_all_electrode_actors()
        self._needs_electrode_update = True
        self.update_all()

    def _show_info_message(self, title, text):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(str(title))
        box.setText(str(text))
        box.setStandardButtons(QMessageBox.Ok)
        box.setStyleSheet(
            "QMessageBox { background-color: #242424; }"
            "QMessageBox QLabel { color: #f2f2f2; min-width: 360px; }"
            "QPushButton { color: #f2f2f2; }"
        )
        box.exec()

    def _show_warning_message(self, title, text):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(str(title))
        box.setText(str(text))
        box.setStandardButtons(QMessageBox.Ok)
        box.setStyleSheet(
            "QMessageBox { background-color: #242424; }"
            "QMessageBox QLabel { color: #f2f2f2; min-width: 360px; }"
            "QPushButton { color: #f2f2f2; }"
        )
        box.exec()

    def _apply_loaded_transform_meta(self, spec):
        tm = dict(getattr(spec, "transform_meta", {}) or {})
        cd = dict(tm.get("current_design", {}) or {})
        if not cd:
            return
        self.design_transform.update(
            {
                "tx_mm": float(cd.get("tx_mm", 0.0)),
                "ty_mm": float(cd.get("ty_mm", 0.0)),
                "tz_mm": float(cd.get("tz_mm", 0.0)),
                "rx_deg": float(cd.get("rx_deg", 0.0)),
                "ry_deg": float(cd.get("ry_deg", 0.0)),
                "rz_deg": float(cd.get("rz_deg", 0.0)),
                "scale": float(cd.get("scale", 1.0)),
                "mirror_x": bool(cd.get("mirror_x", False)),
                "pivot_mode": str(tm.get("pivot_mode", "entry")),
            }
        )
        snap = dict(tm.get("snap", {}) or {})
        if snap:
            self.state["snap_enabled"] = bool(
                snap.get("enabled", self.state.get("snap_enabled", True))
            )
            self.state["snap_mm"] = float(
                snap.get("mm", self.state.get("snap_mm", 0.1))
            )
            self.state["snap_deg"] = float(
                snap.get("deg", self.state.get("snap_deg", 1.0))
            )
            self.state["snap_scale"] = float(
                snap.get("scale", self.state.get("snap_scale", 0.01))
            )
        self._last_ui_design = {
            "rot_x": float(self.design_transform.get("rx_deg", 0.0)),
            "rot_y": float(self.design_transform.get("ry_deg", 0.0)),
            "rot_z": float(self.design_transform.get("rz_deg", 0.0)),
            "scale": float(self.design_transform.get("scale", 1.0)),
        }
        self._sync_manip_controls_from_state()

    def _apply_loaded_placement_snapshot(self, spec):
        snap = dict(getattr(spec, "placement_snapshot", {}) or {})
        transforms = list(snap.get("comb_transforms", []) or [])
        if not transforms:
            return

        if self._instance_mode_active():
            if not self._scene_instances:
                self._create_instance(
                    revision_id=self._active_design_revision_id, selected=True
                )
            tr = dict(transforms[0])
            coordinate_frame = str(snap.get("coordinate_frame", "") or "")
            is_legacy_snapshot = coordinate_frame == ""
            if is_legacy_snapshot and (self.dataset_id == "human_demo"):
                legacy_xyz = np.array(
                    [
                        float(tr.get("x_mm", tr.get("tx_mm", 0.0))),
                        float(tr.get("y_mm", tr.get("ty_mm", 0.0))),
                        float(tr.get("z_mm", tr.get("tz_mm", 0.0))),
                    ],
                    dtype=np.float64,
                )
                mapped_xyz = map_legacy_vector_to_ras(
                    legacy_xyz, self._legacy_vector_map
                )
                tr["x_mm"], tr["y_mm"], tr["z_mm"] = [float(v) for v in mapped_xyz]
                tr["tx_mm"], tr["ty_mm"], tr["tz_mm"] = [float(v) for v in mapped_xyz]
            inst = self._scene_instances[0]
            inst.placement = normalize_instance_placement(
                {
                    "tx_mm": float(tr.get("x_mm", tr.get("tx_mm", 0.0))),
                    "ty_mm": float(tr.get("y_mm", tr.get("ty_mm", 0.0))),
                    "tz_mm": float(tr.get("z_mm", tr.get("tz_mm", 0.0))),
                    "rx_deg": float(tr.get("rx_deg", 0.0)),
                    "ry_deg": float(tr.get("ry_deg", 0.0)),
                    "rz_deg": float(tr.get("rz_deg", 0.0)),
                }
            )
            self._rebuild_instance_index()
            self._last_ui_move_mm = {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
            self._sync_instance_list_from_state()
            self._sync_instance_controls_from_selection()
            return

        coordinate_frame = str(snap.get("coordinate_frame", "") or "")
        is_legacy_snapshot = coordinate_frame == ""
        should_migrate_legacy = is_legacy_snapshot and (self.dataset_id == "human_demo")
        legacy_map = np.asarray(
            getattr(self, "_legacy_vector_map", np.eye(3)), dtype=np.float64
        )

        for tr in transforms:
            idx = int(tr.get("comb_index", -1))
            if idx < 0 or idx >= self.num_combs:
                continue

            if should_migrate_legacy:
                legacy_xyz = np.array(
                    [
                        float(tr.get("x_mm", tr.get("tx_mm", 0.0))),
                        float(tr.get("y_mm", tr.get("ty_mm", 0.0))),
                        float(tr.get("z_mm", tr.get("tz_mm", 0.0))),
                    ],
                    dtype=np.float64,
                )
                mapped_xyz = map_legacy_vector_to_ras(legacy_xyz, legacy_map)
                tr["x_mm"] = float(mapped_xyz[0])
                tr["y_mm"] = float(mapped_xyz[1])
                tr["z_mm"] = float(mapped_xyz[2])
                tr["tx_mm"] = float(mapped_xyz[0])
                tr["ty_mm"] = float(mapped_xyz[1])
                tr["tz_mm"] = float(mapped_xyz[2])

            offs = self.comb_offsets_mm.setdefault(
                idx, {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
            )
            offs["depth"] = float(tr.get("depth_mm", offs.get("depth", 0.0)))
            offs["x"] = float(tr.get("x_mm", tr.get("tx_mm", offs.get("x", 0.0))))
            offs["y"] = float(tr.get("y_mm", tr.get("ty_mm", offs.get("y", 0.0))))
            offs["z"] = float(tr.get("z_mm", tr.get("tz_mm", offs.get("z", 0.0))))
        self._last_ui_move_mm = {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}

    def _add_double_spin_row(
        self,
        parent_layout,
        label,
        value,
        *,
        lo,
        hi,
        decimals=2,
        step=0.01,
        callback=None,
    ):
        """Append a "label: spinbox" row to ``parent_layout`` and return the spinbox.

        Replaces the repeated 8-line pattern used for snap mm/deg/scale and
        drag sensitivity rows.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(float(lo), float(hi))
        spin.setDecimals(int(decimals))
        spin.setSingleStep(float(step))
        spin.setValue(float(value))
        if callback is not None:
            spin.valueChanged.connect(callback)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return spin

    def _create_slider_with_spin(
        self,
        label,
        min_val,
        max_val,
        init_val,
        parent_layout,
        decimals=1,
        step=0.1,
        callback=None,
    ):
        """Create a labeled slider with spinbox."""
        row = QHBoxLayout()
        lbl = QLabel(label + ":")
        lbl.setMinimumWidth(80)
        row.addWidget(lbl)

        multiplier = int(1.0 / step) if step > 0 else 100

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * multiplier))
        slider.setMaximum(int(max_val * multiplier))
        slider.setValue(int(init_val * multiplier))
        row.addWidget(slider, stretch=1)

        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(init_val)
        spin.setDecimals(int(decimals))
        spin.setSingleStep(step)
        spin.setMinimumWidth(90)
        row.addWidget(spin)

        parent_layout.addLayout(row)

        def on_slider(val):
            spin.blockSignals(True)
            spin.setValue(val / multiplier)
            spin.blockSignals(False)
            if callback is None:
                self._on_value_changed()
            else:
                callback()

        def on_spin(val):
            slider.blockSignals(True)
            slider.setValue(int(val * multiplier))
            slider.blockSignals(False)
            if callback is None:
                self._on_value_changed()
            else:
                callback()

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

        return slider, spin

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def _on_value_changed(self):
        """Handle position/R2 value changes with debouncing."""
        new_depth = float(self.depth_spin.value())
        new_x = float(self.x_spin.value())
        new_y = float(self.y_spin.value())
        new_z = float(self.z_spin.value())

        if self._suppress_value_changed:
            return

        # Optional snapping for translation controls.
        snapped_depth = self._snap_value(new_depth, self.state.get("snap_mm", 0.1))
        snapped_x = self._snap_value(new_x, self.state.get("snap_mm", 0.1))
        snapped_y = self._snap_value(new_y, self.state.get("snap_mm", 0.1))
        snapped_z = self._snap_value(new_z, self.state.get("snap_mm", 0.1))
        if any(
            abs(a - b) > 1e-9
            for a, b in (
                (new_depth, snapped_depth),
                (new_x, snapped_x),
                (new_y, snapped_y),
                (new_z, snapped_z),
            )
        ):
            self._suppress_value_changed = True
            self.depth_spin.setValue(snapped_depth)
            self.x_spin.setValue(snapped_x)
            self.y_spin.setValue(snapped_y)
            self.z_spin.setValue(snapped_z)
            self._suppress_value_changed = False
            new_depth, new_x, new_y, new_z = (
                snapped_depth,
                snapped_x,
                snapped_y,
                snapped_z,
            )

        self.state["depth"] = new_depth
        self.state["x"] = new_x
        self.state["y"] = new_y
        self.state["z"] = new_z
        self.state["r2"] = float(self.r2_spin.value())

        if self._instance_mode_active():
            d_x = new_x - self._last_ui_move_mm["x"]
            d_y = new_y - self._last_ui_move_mm["y"]
            d_z = new_z - self._last_ui_move_mm["z"]
            self._last_ui_move_mm = {"depth": 0.0, "x": new_x, "y": new_y, "z": new_z}
            if any(abs(v) > 1e-9 for v in (d_x, d_y, d_z)):
                self._push_undo_state()
                if self._apply_instance_transform_delta(d_tx=d_x, d_ty=d_y, d_tz=d_z):
                    self._needs_electrode_update = True
            else:
                self._needs_brain_update = True
                self._needs_electrode_update = True
            self._update_pending = True
            self._update_timer.start(self._debounce_ms)
            return

        d_depth = new_depth - self._last_ui_move_mm["depth"]
        d_x = new_x - self._last_ui_move_mm["x"]
        d_y = new_y - self._last_ui_move_mm["y"]
        d_z = new_z - self._last_ui_move_mm["z"]
        self._last_ui_move_mm = {"depth": new_depth, "x": new_x, "y": new_y, "z": new_z}

        if any(abs(v) > 1e-9 for v in (d_depth, d_x, d_y, d_z)):
            self._push_undo_state()
            for idx, selected in self.state["show_combs"].items():
                if selected:
                    self.comb_offsets_mm[idx]["depth"] += d_depth
                    self.comb_offsets_mm[idx]["x"] += d_x
                    self.comb_offsets_mm[idx]["y"] += d_y
                    self.comb_offsets_mm[idx]["z"] += d_z
            # Position-only change: electrodes move, brain stays
            self._needs_electrode_update = True
        else:
            # R2 changed: need to update brain areas too
            self._needs_brain_update = True
            self._needs_electrode_update = True

        # Debounced update - restart timer on each change
        self._update_pending = True
        self._update_timer.start(self._debounce_ms)

    def _on_instance_rotation_changed(self):
        if not self._instance_mode_active() or getattr(
            self, "_suppress_value_changed", False
        ):
            return
        new_rx = self._snap_value(
            float(self.place_rot_x_spin.value()), self.state.get("snap_deg", 1.0)
        )
        new_ry = self._snap_value(
            float(self.place_rot_y_spin.value()), self.state.get("snap_deg", 1.0)
        )
        new_rz = self._snap_value(
            float(self.place_rot_z_spin.value()), self.state.get("snap_deg", 1.0)
        )
        old = dict(self._last_ui_instance_rot)
        d_rx = new_rx - float(old.get("rx_deg", 0.0))
        d_ry = new_ry - float(old.get("ry_deg", 0.0))
        d_rz = new_rz - float(old.get("rz_deg", 0.0))
        if abs(d_rx) <= 1e-9 and abs(d_ry) <= 1e-9 and abs(d_rz) <= 1e-9:
            return
        self._push_undo_state()
        if self._apply_instance_transform_delta(d_rx=d_rx, d_ry=d_ry, d_rz=d_rz):
            self._last_ui_instance_rot = {
                "rx_deg": new_rx,
                "ry_deg": new_ry,
                "rz_deg": new_rz,
            }
            self._needs_electrode_update = True
            self.update_all()

    def _sync_instance_controls_from_selection(self):
        if not hasattr(self, "place_rot_x_spin"):
            return
        selected = self._selected_instances()
        enabled = bool(selected)
        for widget in (
            self.x_spin,
            self.y_spin,
            self.z_spin,
            self.place_rot_x_spin,
            self.place_rot_y_spin,
            self.place_rot_z_spin,
        ):
            widget.setEnabled(enabled)
        self.depth_spin.setEnabled((not self._instance_mode_active()) and enabled)
        if not enabled:
            return
        first = selected[0]
        place = normalize_instance_placement(first.placement)
        self._suppress_value_changed = True
        self.x_spin.setValue(float(place["tx_mm"]))
        self.y_spin.setValue(float(place["ty_mm"]))
        self.z_spin.setValue(float(place["tz_mm"]))
        self.place_rot_x_spin.setValue(float(place["rx_deg"]))
        self.place_rot_y_spin.setValue(float(place["ry_deg"]))
        self.place_rot_z_spin.setValue(float(place["rz_deg"]))
        self._suppress_value_changed = False
        self._last_ui_move_mm = {
            "depth": 0.0,
            "x": float(place["tx_mm"]),
            "y": float(place["ty_mm"]),
            "z": float(place["tz_mm"]),
        }
        self._last_ui_instance_rot = {
            "rx_deg": float(place["rx_deg"]),
            "ry_deg": float(place["ry_deg"]),
            "rz_deg": float(place["rz_deg"]),
        }

    def _on_manipulation_changed(self):
        if self._suppress_manip_value_changed:
            return

        self.state["snap_enabled"] = (
            bool(self.snap_cb.isChecked())
            if hasattr(self, "snap_cb")
            else self.state.get("snap_enabled", True)
        )
        self.state["snap_mm"] = (
            float(self.snap_mm_spin.value())
            if hasattr(self, "snap_mm_spin")
            else self.state.get("snap_mm", 0.1)
        )
        self.state["snap_deg"] = (
            float(self.snap_deg_spin.value())
            if hasattr(self, "snap_deg_spin")
            else self.state.get("snap_deg", 1.0)
        )
        self.state["snap_scale"] = (
            float(self.snap_scale_spin.value())
            if hasattr(self, "snap_scale_spin")
            else self.state.get("snap_scale", 0.01)
        )

        new_rx = self._snap_value(
            float(self.rot_x_spin.value()), self.state.get("snap_deg", 1.0)
        )
        new_ry = self._snap_value(
            float(self.rot_y_spin.value()), self.state.get("snap_deg", 1.0)
        )
        new_rz = self._snap_value(
            float(self.rot_z_spin.value()), self.state.get("snap_deg", 1.0)
        )
        new_scale = self._snap_value(
            float(self.scale_spin.value()), self.state.get("snap_scale", 0.01)
        )
        new_scale = max(0.01, float(new_scale))

        mode = str(self.pivot_combo.currentData() or "entry")
        px = float(self.pivot_x_spin.value())
        py = float(self.pivot_y_spin.value())
        pz = float(self.pivot_z_spin.value())
        if mode == "custom":
            px = self._snap_value(px, self.state.get("snap_mm", 0.1))
            py = self._snap_value(py, self.state.get("snap_mm", 0.1))
            pz = self._snap_value(pz, self.state.get("snap_mm", 0.1))

        old = self._last_ui_design
        d_rx = new_rx - old["rot_x"]
        d_ry = new_ry - old["rot_y"]
        d_rz = new_rz - old["rot_z"]
        d_scale = 1.0 if abs(old["scale"]) < 1e-12 else (new_scale / old["scale"])
        piv_old = list(self.design_transform.get("pivot_custom_mm", [0.0, 0.0, 0.0]))
        if len(piv_old) < 3:
            piv_old = (piv_old + [0.0, 0.0, 0.0])[:3]
        pivot_changed = (
            mode != self.design_transform.get("pivot_mode", "entry")
            or abs(px - float(piv_old[0])) > 1e-9
            or abs(py - float(piv_old[1])) > 1e-9
            or abs(pz - float(piv_old[2])) > 1e-9
        )
        changed = (
            any(abs(v) > 1e-9 for v in (d_rx, d_ry, d_rz, d_scale - 1.0))
            or pivot_changed
        )
        if not changed:
            return

        self._push_undo_state()
        self.design_transform["rx_deg"] += d_rx
        self.design_transform["ry_deg"] += d_ry
        self.design_transform["rz_deg"] += d_rz
        self.design_transform["scale"] = max(
            0.01, float(self.design_transform.get("scale", 1.0) * d_scale)
        )
        self.design_transform["pivot_mode"] = mode
        self.design_transform["pivot_custom_mm"] = [px, py, pz]
        self._last_ui_design = {
            "rot_x": new_rx,
            "rot_y": new_ry,
            "rot_z": new_rz,
            "scale": new_scale,
        }
        self._sync_manip_controls_from_state()
        self._update_template_transform()
        self._needs_electrode_update = True
        self.update_all()

    def _do_update(self):
        """Execute the actual update after debounce delay."""
        if self._update_pending:
            self._update_pending = False
            self.update_all()

    def _on_mode_changed(self, idx):
        modes = ["anatomy", "ecc", "polar"]
        self.state["mode"] = modes[idx]
        self._needs_brain_update = True
        self._needs_electrode_update = True
        self.update_all()

    def _on_undersampling_changed(self, val):
        self.state["undersampling"] = val
        self._needs_brain_update = True
        self._needs_electrode_update = True
        self.update_all()

    def _on_prf_source_changed(self, source):
        self.state["prf_source"] = source
        if source.startswith("NN Prediction (") and source.endswith(")"):
            label = source[len("NN Prediction (") : -1]
            monkey = self._nn_source_lookup.get(
                source, self._nn_monkey_for_source_label(label)
            )
            self._load_nn_predmaps(monkey)
            self.ecc_map = self.nn_predmaps[monkey]["ecc"]
            self.polar_map = self.nn_predmaps[monkey]["pol"]
            self.R2_map = self.nn_predmaps[monkey]["r2"]
            self.sz_map = self.nn_predmaps[monkey].get("sz", None)
            # Activate dense brain rendering if precomputed data exists
            self._nn_dense_brain_data = self._nn_dense_brain_cache.get(monkey)
        else:
            if source not in self.prf_maps:
                source = next(iter(self.prf_maps.keys()))
                self.state["prf_source"] = source
            self.ecc_map = self.prf_maps[source]["ecc"]
            self.polar_map = self.prf_maps[source]["pol"]
            self.R2_map = self.prf_maps[source]["r2"]
            self.sz_map = self.prf_maps[source].get("sz", None)
            # Deactivate dense brain rendering
            self._nn_dense_brain_data = None
        self._apply_r2_control_state()
        self._needs_brain_update = True
        self._needs_electrode_update = True
        self.update_all()

    def _on_area_visibility_changed(self):
        for name, cb in self.area_checkboxes.items():
            self.state["show_areas"][name] = cb.isChecked()
        self._needs_brain_update = True
        self._needs_electrode_update = True
        self.update_all()

    def _on_area_style_changed(self):
        """Handle brain area style changes."""
        self.state["area_point_size"] = self.area_size_spin.value()
        self.state["area_opacity"] = self.area_opacity_slider.value() / 100.0
        self._needs_brain_update = True
        self.update_all()

    def _set_ecc_scope(self, scope_deg, *, sync_combo=False, sync_slider=False):
        v = int(np.clip(int(round(float(scope_deg))), 5, 60))
        self._ecc_scope = v
        self.polar_plot.set_max_ecc(v)
        if hasattr(self, "ecc_scope_value_label"):
            self.ecc_scope_value_label.setText(f"{v}°")
        if sync_slider and hasattr(self, "ecc_scope_slider"):
            self.ecc_scope_slider.blockSignals(True)
            self.ecc_scope_slider.setValue(v)
            self.ecc_scope_slider.blockSignals(False)
        if sync_combo and hasattr(self, "ecc_scope_combo"):
            idx = self.ecc_scope_combo.findData(v)
            if idx >= 0:
                self.ecc_scope_combo.blockSignals(True)
                self.ecc_scope_combo.setCurrentIndex(idx)
                self.ecc_scope_combo.blockSignals(False)
        self.update_all()

    def _on_ecc_scope_changed(self):
        """Update VF scope from combo presets."""
        val = self.ecc_scope_combo.currentData()
        if val is None:
            return
        self._set_ecc_scope(val, sync_slider=True, sync_combo=False)

    def _on_ecc_scope_slider_changed(self, value):
        """Update VF scope from continuous slider bar."""
        self._set_ecc_scope(value, sync_slider=False, sync_combo=True)

    def _on_electrode_color_mode_changed(self, _idx):
        mode = str(self.electrode_color_combo.currentData() or "classic")
        self.state["electrode_color_mode"] = mode
        if hasattr(self, "electrode_color_btn"):
            self.electrode_color_btn.setEnabled(mode == "single")
        self._rebuild_probe_colors()
        self._refresh_comb_checkbox_colors()
        self._electrode_clouds.clear()
        self._needs_electrode_update = True
        self.update_all()

    def _on_pick_electrode_color(self):
        current_hex = str(self.state.get("electrode_color_hex", "#00D4FF"))
        color = QColorDialog.getColor(
            QColor(current_hex), self, "Choose Electrode Color"
        )
        if not color.isValid():
            return
        self.state["electrode_color_hex"] = color.name().upper()
        if self.electrode_color_combo.currentData() != "single":
            idx = self.electrode_color_combo.findData("single")
            if idx >= 0:
                self.electrode_color_combo.setCurrentIndex(idx)
                return
        self._rebuild_probe_colors()
        self._refresh_comb_checkbox_colors()
        self._electrode_clouds.clear()
        self._needs_electrode_update = True
        self.update_all()

    def _on_display_changed(self):
        prev_elec_size = self.state.get("electrode_dot_size")
        prev_fid_size = self.state.get("fiducial_size")
        new_elec_size = self.elec_size_spin.value()
        new_fid_size = self.fid_size_spin.value()
        self.state["electrode_dot_size"] = new_elec_size
        self.state["rf_dot_size"] = self.rf_size_spin.value()
        self.state["fiducial_size"] = new_fid_size
        self.state["show_electrodes"] = self.show_electrodes_cb.isChecked()
        self.state["rf_centers_only"] = self.rf_centers_only_cb.isChecked()
        self.state["rf_alpha"] = self.rf_alpha_slider.value() / 100.0
        # Only rebuild caches when the inputs they depend on actually changed.
        # rf_dot_size / rf_centers_only / rf_alpha only affect the polar plot.
        if new_elec_size != prev_elec_size:
            self._electrode_clouds.clear()
        if new_fid_size != prev_fid_size:
            self._fiducials_mesh = None
        self._needs_electrode_update = True
        self.update_all()

    def _on_comb_visibility_changed(self):
        for i, cb in self.comb_checkboxes.items():
            self.state["show_combs"][i] = cb.isChecked()
        self._needs_electrode_update = True
        self.update_all()

    def _select_all_combs(self):
        for cb in self.comb_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        for i in self.state["show_combs"]:
            self.state["show_combs"][i] = True
        self._needs_electrode_update = True
        self.update_all()

    def _select_no_combs(self):
        for cb in self.comb_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        for i in self.state["show_combs"]:
            self.state["show_combs"][i] = False
        self._needs_electrode_update = True
        self.update_all()

    def _reset_position(self):
        self._push_undo_state()
        if self._instance_mode_active():
            for inst in self._selected_instances():
                inst.placement = normalize_instance_placement({})
            self._last_ui_move_mm = {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
            self._last_ui_instance_rot = {"rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": 0.0}
            self._suppress_value_changed = True
            self.depth_spin.setValue(0)
            self.x_spin.setValue(0)
            self.y_spin.setValue(0)
            self.z_spin.setValue(0)
            if hasattr(self, "place_rot_x_spin"):
                self.place_rot_x_spin.setValue(0)
                self.place_rot_y_spin.setValue(0)
                self.place_rot_z_spin.setValue(0)
            self._suppress_value_changed = False
            self._rebuild_instance_index()
            self._sync_instance_list_from_state()
            self._needs_electrode_update = True
            self.update_all()
            return
        for i in self.comb_offsets_mm:
            self.comb_offsets_mm[i] = {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
        self._last_ui_move_mm = {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}

        self._suppress_value_changed = True
        self.depth_spin.setValue(0)
        self.x_spin.setValue(0)
        self.y_spin.setValue(0)
        self.z_spin.setValue(0)
        self.r2_spin.setValue(R2_THRESHOLD_INIT)
        self._suppress_value_changed = False

        self.state.update(
            {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0, "r2": R2_THRESHOLD_INIT}
        )
        self._needs_brain_update = True
        self._needs_electrode_update = True
        self.update_all()

    # =========================================================================
    # OPTIMIZED COMPUTATION
    # =========================================================================

    def compute_electrode_positions(self):
        """Compute electrode positions for all visible placements.

        Returns a list of (4, N) homogeneous-coord arrays. In instance mode,
        one entry per visible instance; otherwise one entry placed at the
        non-instance anchor.
        """
        all_elec = []

        template = getattr(self, "template_contacts_transformed_vox", None)
        if template is None:
            # Fallback for very early startup paths.
            rotated = getattr(
                self, "rotated_base_comb", np.empty((0, 2), dtype=np.float64)
            )
            template = np.column_stack(
                [
                    np.zeros(len(rotated), dtype=np.float64),
                    rotated[:, 0] if len(rotated) else np.array([], dtype=np.float64),
                    rotated[:, 1] if len(rotated) else np.array([], dtype=np.float64),
                ]
            )

        pts = np.asarray(template, dtype=np.float64)

        if self._instance_mode_active():
            self._last_render_instance_ids = []
            self._last_render_contacts_by_instance = {}
            for inst in self._scene_instances:
                if not inst.visible:
                    continue
                inst_pts = self._instance_contacts_vox(inst)
                self._last_render_instance_ids.append(inst.instance_id)
                self._last_render_contacts_by_instance[inst.instance_id] = inst_pts
                n = inst_pts.shape[0]
                elec = np.zeros((4, n), dtype=np.float32)
                if n > 0:
                    elec[0, :] = inst_pts[:, 0]
                    elec[1, :] = inst_pts[:, 1]
                    elec[2, :] = inst_pts[:, 2]
                    elec[3, :] = 1
                all_elec.append(elec)
            return all_elec

        if pts.shape[0] == 0:
            return [np.zeros((4, 0), dtype=np.float32)]

        # Translate so entry contact sits at the configured anchor.
        idx = int(min(max(0, self.entry_electrode_idx), pts.shape[0] - 1))
        ref = pts[idx]
        anchor = getattr(self, "_nonspike_anchor_vox", None)
        if anchor is None or len(anchor) != 3:
            anchor = np.array(
                [
                    float(self.fiducial_anterior_list[0, 0]),
                    float(self.fiducial_anterior_list[0, 1]),
                    float(self._fiducial_plot_z(self.fiducial_anterior_list[0, 2])),
                ],
                dtype=np.float64,
            )
        tgt_x, tgt_y, tgt_z = float(anchor[0]), float(anchor[1]), float(anchor[2])

        # Apply global offset from depth/xyz sliders (comb 0 only)
        offs = self.comb_offsets_mm.get(0, {"depth": 0.0, "x": 0.0, "y": 0.0, "z": 0.0})
        px = float(self.mm_per_pixel[0])
        ox = offs["x"] / px + offs["depth"] / px  # depth treated as simple Z shift
        oy = offs["y"] / px
        oz = offs["z"] / px

        n = pts.shape[0]
        elec = np.zeros((4, n), dtype=np.float32)
        elec[0, :] = pts[:, 0] - ref[0] + tgt_x + ox
        elec[1, :] = pts[:, 1] - ref[1] + tgt_y + oy
        elec[2, :] = pts[:, 2] - ref[2] + tgt_z + oz
        elec[3, :] = 1
        return [elec]

    def _visible_show_area_indices(self):
        """1-based area indices toggled on in show_areas (defaulting to True)."""
        return {
            i + 1
            for i, name in enumerate(["V1", "V2", "V3", "V4"])
            if self.state["show_areas"].get(name, True)
        }

    def _filter_comb_to_visible_areas(self, probe_idx, comb, visible_areas):
        """Per-comb pipeline: skip + round + bounds + area-mask filter.

        Returns (coords_int, local_indices) restricted to visible-area voxels,
        or None if the comb is hidden, empty, out of bounds, or has nothing in
        a visible area.
        """
        if (not self._instance_mode_active()) and (
            not self.state["show_combs"].get(probe_idx, True)
        ):
            return None
        n_contacts = int(comb.shape[1]) if getattr(comb, "ndim", 0) >= 2 else 0
        if n_contacts == 0:
            return None
        coords = np.column_stack(
            [
                np.round(comb[0, :]).astype(int),
                np.round(comb[1, :]).astype(int),
                np.round(comb[2, :]).astype(int),
            ]
        )
        local_indices = np.arange(n_contacts, dtype=int)
        shape = self.area_volume.shape
        in_bounds = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < shape[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < shape[1])
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < shape[2])
        )
        if not np.any(in_bounds):
            return None
        coords = coords[in_bounds]
        local_indices = local_indices[in_bounds]
        areas = self.area_volume[coords[:, 0], coords[:, 1], coords[:, 2]]
        in_area = np.isin(areas, list(visible_areas))
        if not np.any(in_area):
            return None
        return coords[in_area], local_indices[in_area]

    def collect_rfs(self, elec_list):
        """Collect RF data for live polar plot via vectorized lookups."""
        visible_areas = self._visible_show_area_indices()
        r2_thresh = self.state["r2"]

        coord_chunks = []
        probe_chunks = []
        for pi, comb in enumerate(elec_list):
            filtered = self._filter_comb_to_visible_areas(pi, comb, visible_areas)
            if filtered is None:
                continue
            coords, _local = filtered
            coord_chunks.append(coords)
            probe_chunks.append(np.full(coords.shape[0], pi, dtype=int))

        if not coord_chunks:
            return [], [], [], []

        coords_area = np.vstack(coord_chunks)
        probes_area = np.concatenate(probe_chunks)

        ecc, polar, r2, sz = get_rf_batch_vectorized(
            coords_area,
            self.ecc_map,
            self.polar_map,
            self.R2_map,
            r2_thresh,
            sz_map=self.sz_map,
        )
        valid_rf = ecc > 0
        return (
            ecc[valid_rf].tolist(),
            polar[valid_rf].tolist(),
            probes_area[valid_rf].tolist(),
            sz[valid_rf].tolist(),
        )

    def _rf_to_xy_deg(self, ecc_deg, polar_deg):
        pol_rad = np.radians(float(polar_deg))
        if getattr(self, "pol_convention", "standard") == "neuropythy":
            return float(ecc_deg * np.sin(pol_rad)), float(ecc_deg * np.cos(pol_rad))
        return float(ecc_deg * np.cos(pol_rad)), float(ecc_deg * np.sin(pol_rad))

    def _iter_visible_rf_rows(self, elec_list):
        visible_areas = self._visible_show_area_indices()
        r2_thresh = float(self.state["r2"])
        ecc_scope = float(getattr(self, "_ecc_scope", 15))
        instance_ids = [
            inst.instance_id for inst in self._scene_instances if inst.visible
        ]

        for probe_idx, comb in enumerate(elec_list):
            filtered = self._filter_comb_to_visible_areas(
                probe_idx, comb, visible_areas
            )
            if filtered is None:
                continue
            coords, local_indices = filtered
            ecc, polar, _r2, _sz = get_rf_batch_vectorized(
                coords,
                self.ecc_map,
                self.polar_map,
                self.R2_map,
                r2_thresh,
                sz_map=self.sz_map,
            )
            valid_rf = (
                (ecc > 0) & np.isfinite(ecc) & np.isfinite(polar) & (ecc <= ecc_scope)
            )
            if not np.any(valid_rf):
                continue

            implant_id = (
                instance_ids[probe_idx]
                if self._instance_mode_active() and probe_idx < len(instance_ids)
                else f"comb_{probe_idx}"
            )
            for contact_idx, ecc_deg, polar_deg in zip(
                local_indices[valid_rf].tolist(),
                ecc[valid_rf].tolist(),
                polar[valid_rf].tolist(),
            ):
                x_deg, y_deg = self._rf_to_xy_deg(ecc_deg, polar_deg)
                yield {
                    "source_app": "implant_explorer",
                    "dataset": str(self.dataset_id),
                    "prf_source": str(self.state.get("prf_source", "")),
                    "implant_id": str(implant_id),
                    "electrode_index": int(contact_idx),
                    "x_deg": x_deg,
                    "y_deg": y_deg,
                    "polar_deg": float(polar_deg),
                    "ecc_deg": float(ecc_deg),
                }

    def _build_rf_export_rows(self):
        return list(self._iter_visible_rf_rows(self.compute_electrode_positions()))

    def _build_rf_export_payload(self, rows):
        return {
            "schema": "rf_export_v1",
            "source_app": "implant_explorer",
            "dataset": str(self.dataset_id),
            "prf_source": str(self.state.get("prf_source", "")),
            "row_count": int(len(rows)),
            "rows": rows,
        }

    def _write_rf_export_csv(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RF_EXPORT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "source_app": row["source_app"],
                        "dataset": row["dataset"],
                        "prf_source": row["prf_source"],
                        "implant_id": row["implant_id"],
                        "electrode_index": row["electrode_index"],
                        "x_deg": f"{row['x_deg']:.6f}",
                        "y_deg": f"{row['y_deg']:.6f}",
                        "polar_deg": f"{row['polar_deg']:.6f}",
                        "ecc_deg": f"{row['ecc_deg']:.6f}",
                    }
                )

    def _export_rfs_csv(self):
        rows = self._build_rf_export_rows()
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export RFs CSV", "", "CSV Files (*.csv)", options=opts
        )
        if not path:
            return
        self._write_rf_export_csv(path, rows)
        self._show_info_message("Saved", f"RF CSV saved:\n{path}\n\nRows: {len(rows)}")

    def _export_rfs_json(self):
        rows = self._build_rf_export_rows()
        payload = self._build_rf_export_payload(rows)
        opts = QFileDialog.Options()
        opts |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export RFs JSON", "", "JSON Files (*.json)", options=opts
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._show_info_message("Saved", f"RF JSON saved:\n{path}\n\nRows: {len(rows)}")

    def update_brain_areas(self):
        """OPTIMIZED: Update brain area visualization with flat points (10x faster than spheres)."""
        undersampling = self.state["undersampling"]
        mode = self.state["mode"]
        r2_thresh = self.state["r2"]
        dense = self._nn_dense_brain_data  # None when not in dense mode

        for name, xa, ya, za in self.visual_areas:
            actor_name = f"area_{name}"
            show = self.state["show_areas"].get(name, True)

            if not show:
                self._safe_remove_actor(actor_name)
                continue

            # --- Pick data source: dense precomputed or original maps ---
            if dense and name in dense:
                pts_all, ecc_v, pol_v, r2_v = dense[name]
                mask = (ecc_v > 0) & (r2_v > r2_thresh)
                idx = np.where(mask)[0][::undersampling]
                if len(idx) == 0:
                    self._safe_remove_actor(actor_name)
                    continue
                pts = pts_all[idx]
                ecc_sel = ecc_v[idx]
                pol_sel = pol_v[idx]
            else:
                # Original path: index maps with visual_areas coords
                ecc_v = self.ecc_map[xa.astype(int), ya.astype(int), za.astype(int)]
                r2_v = self.R2_map[xa.astype(int), ya.astype(int), za.astype(int)]
                pol_v = self.polar_map[xa.astype(int), ya.astype(int), za.astype(int)]
                mask = (ecc_v > 0) & (r2_v > r2_thresh)
                idx = np.where(mask)[0][::undersampling]
                if len(idx) == 0:
                    self._safe_remove_actor(actor_name)
                    continue
                pts = np.column_stack([xa[idx], ya[idx], za[idx]]).astype(np.float32)
                ecc_sel = ecc_v[idx]
                pol_sel = pol_v[idx]

            # Remove old actor and create new
            self._safe_remove_actor(actor_name)

            cloud = pv.PolyData(pts)

            # Get style settings (points only now)
            point_size = self.state["area_point_size"]
            opacity = self.state["area_opacity"]

            if mode == "anatomy":
                self.plotter.add_mesh(
                    cloud,
                    color=AREA_COLORS[name],
                    point_size=point_size,
                    opacity=opacity,
                    render_points_as_spheres=False,
                    name=actor_name,
                )
            elif mode == "ecc":
                cloud["ecc"] = ecc_sel
                self.plotter.add_mesh(
                    cloud,
                    scalars="ecc",
                    cmap="inferno",
                    clim=[ECC_VMIN, ECC_VMAX],
                    point_size=point_size,
                    opacity=opacity,
                    render_points_as_spheres=False,
                    name=actor_name,
                    scalar_bar_args={"title": "Ecc°", "color": "white"},
                )
            elif mode == "polar":
                cloud["pol"] = pol_sel
                self.plotter.add_mesh(
                    cloud,
                    scalars="pol",
                    cmap="hsv",
                    clim=[POL_VMIN, POL_VMAX],
                    point_size=point_size,
                    opacity=opacity,
                    render_points_as_spheres=False,
                    name=actor_name,
                    scalar_bar_args={"title": "Pol°", "color": "white"},
                )

    def update_electrodes(self, elec_list):
        """OPTIMIZED: Update electrode visualization with in-place point updates."""
        if not self.state.get("show_electrodes", True):
            self._clear_all_electrode_actors()
            return
        dot_size = self.state["electrode_dot_size"]
        fast_mode = self.state["fast_mode"]

        if self._instance_mode_active():
            active_names = set()
            for i, comb in enumerate(elec_list):
                if i >= len(self._last_render_instance_ids):
                    continue
                instance_id = self._last_render_instance_ids[i]
                inst = self._instance_for_id(instance_id)
                if inst is None or not inst.visible:
                    continue
                name = f"elec_inst_{instance_id}"
                halo_name = f"halo_inst_{instance_id}"
                active_names.add(name)
                active_names.add(halo_name)
                pts = np.column_stack([comb[0], comb[1], comb[2]]).astype(np.float32)
                selected = bool(inst.selected)
                halo_size = float(dot_size + (5 if selected else 3))
                color = "#ffffff" if selected else self._instance_color(i)
                halo_color = "#ffffff" if selected else "black"

                if (
                    instance_id in self._electrode_clouds
                    and name in self.plotter.actors
                ):
                    self._electrode_clouds[instance_id].points = pts
                    if instance_id in self._electrode_halos:
                        self._electrode_halos[instance_id].points = pts
                    try:
                        if name in self.plotter.actors:
                            self.plotter.actors[name].GetProperty().SetPointSize(
                                float(dot_size)
                            )
                            self.plotter.actors[name].GetProperty().SetColor(
                                *self._hex_to_rgb01(color)
                            )
                        if halo_name in self.plotter.actors:
                            self.plotter.actors[halo_name].GetProperty().SetPointSize(
                                halo_size
                            )
                            self.plotter.actors[halo_name].GetProperty().SetColor(
                                *self._hex_to_rgb01(
                                    "#ffffff" if selected else "#000000"
                                )
                            )
                    except Exception:
                        pass
                    continue

                self._safe_remove_actor(name)
                self._safe_remove_actor(halo_name)
                cloud = pv.PolyData(pts)
                halo_cloud = pv.PolyData(pts)
                self._electrode_clouds[instance_id] = cloud
                self._electrode_halos[instance_id] = halo_cloud
                self.plotter.add_mesh(
                    halo_cloud,
                    color=halo_color,
                    point_size=halo_size,
                    render_points_as_spheres=not fast_mode,
                    name=halo_name,
                )
                self.plotter.add_mesh(
                    cloud,
                    color=color,
                    point_size=dot_size,
                    render_points_as_spheres=not fast_mode,
                    name=name,
                )
                if name not in self.electrode_actors:
                    self.electrode_actors.append(name)
                if halo_name not in self.electrode_actors:
                    self.electrode_actors.append(halo_name)

            for name in list(self.electrode_actors):
                if name in active_names:
                    continue
                self._safe_remove_actor(name)
                self.electrode_actors.remove(name)
            stale_ids = set(self._electrode_clouds) - set(
                self._last_render_instance_ids
            )
            for instance_id in stale_ids:
                self._electrode_clouds.pop(instance_id, None)
                self._electrode_halos.pop(instance_id, None)
            return

        for i, comb in enumerate(elec_list):
            name = f"elec_{i}"
            halo_name = f"halo_{i}"
            show = self.state["show_combs"].get(i, True)

            if not show:
                self._safe_remove_actor(name)
                self._safe_remove_actor(halo_name)
                self._electrode_clouds.pop(i, None)
                self._electrode_halos.pop(i, None)
                continue

            pts = np.column_stack([comb[0], comb[1], comb[2]]).astype(np.float32)
            if i in self._electrode_clouds and name in self.plotter.actors:
                self._electrode_clouds[i].points = pts
                if i in self._electrode_halos:
                    self._electrode_halos[i].points = pts
                try:
                    if name in self.plotter.actors:
                        self.plotter.actors[name].GetProperty().SetPointSize(
                            float(dot_size)
                        )
                    if halo_name in self.plotter.actors:
                        self.plotter.actors[halo_name].GetProperty().SetPointSize(
                            float(dot_size + 3)
                        )
                except Exception:
                    pass
            else:
                self._safe_remove_actor(name)
                self._safe_remove_actor(halo_name)
                cloud = pv.PolyData(pts)
                halo_cloud = pv.PolyData(pts)
                self._electrode_clouds[i] = cloud
                self._electrode_halos[i] = halo_cloud
                self.plotter.add_mesh(
                    halo_cloud,
                    color="black",
                    point_size=dot_size + 3,
                    render_points_as_spheres=not fast_mode,
                    name=halo_name,
                )
                self.plotter.add_mesh(
                    cloud,
                    color=self.probe_colors[i % len(self.probe_colors)],
                    point_size=dot_size,
                    render_points_as_spheres=not fast_mode,
                    name=name,
                )
                if name not in self.electrode_actors:
                    self.electrode_actors.append(name)
                if halo_name not in self.electrode_actors:
                    self.electrode_actors.append(halo_name)

    def update_all(self):
        """Update all visualizations (skips brain areas if only position changed)."""
        saved_camera = None if self._first_render else self.plotter.camera_position

        elecs = self.compute_electrode_positions()
        self._transform_warnings = []
        if hasattr(self, "template_contacts_transformed_vox"):
            px = float(self.mm_per_pixel[0])
            template_mm = (
                np.asarray(self.template_contacts_transformed_vox, dtype=np.float32)
                * px
            )
            self._transform_warnings.extend(
                validate_contacts_soft(
                    template_mm,
                    scale=float(self.design_transform.get("scale", 1.0)),
                    min_spacing_mm=0.01,
                )
            )
            if self._instance_mode_active():
                shown = [np.column_stack([c[0], c[1], c[2]]) for c in elecs]
            else:
                shown = [
                    np.column_stack([c[0], c[1], c[2]])
                    for i, c in enumerate(elecs)
                    if self.state["show_combs"].get(i, True)
                ]
            if shown:
                vox = np.vstack(shown)
                shp = np.asarray(self.brain_data.shape, dtype=int)
                inb = (
                    (vox[:, 0] >= 0)
                    & (vox[:, 0] < shp[0])
                    & (vox[:, 1] >= 0)
                    & (vox[:, 1] < shp[1])
                    & (vox[:, 2] >= 0)
                    & (vox[:, 2] < shp[2])
                )
                out_n = int((~inb).sum())
                if out_n > 0:
                    self._transform_warnings.append(
                        f"{out_n} rendered contacts are outside volume bounds."
                    )

        if self._needs_anatomy_update:
            self._update_anatomy_context()
            self._needs_anatomy_update = False

        # Only update brain areas if needed (not during position-only changes)
        if self._needs_brain_update:
            self.update_brain_areas()
            self._needs_brain_update = False

        if self._needs_electrode_update:
            self.update_electrodes(elecs)
            self._needs_electrode_update = False

        # Fiducials - only rebuild if needed (size changed or first render)
        fid_size = self.state["fiducial_size"] * 0.05  # Smaller triangles
        if (
            self._fiducials_mesh is None
            or getattr(self, "_last_fid_size", None) != fid_size
        ):
            self._safe_remove_actor("fiducials")

            if self.num_combs > 0:
                fiducials_mesh = pv.PolyData()
                for i in range(self.num_combs):
                    pt = [
                        self.fiducial_anterior_list[i, 0],
                        self.fiducial_anterior_list[i, 1],
                        self._fiducial_plot_z(self.fiducial_anterior_list[i, 2]),
                    ]
                    cone = pv.Cone(
                        center=pt,
                        direction=[0, 1, 0],
                        height=fid_size * 2,
                        radius=fid_size,
                        resolution=3,
                    )
                    fiducials_mesh = fiducials_mesh.merge(cone)
                self._fiducials_mesh = fiducials_mesh
                self._last_fid_size = fid_size
                self.plotter.add_mesh(fiducials_mesh, color="yellow", name="fiducials")

        ecc_l, pol_l, probe_l, sz_l = self.collect_rfs(elecs)

        if self._instance_mode_active():
            selected_combs = len(self._selected_instance_ids)
            total = int(sum(c.shape[1] for c in elecs))
        else:
            selected_combs = sum(
                1
                for i in range(self.num_combs)
                if self.state["show_combs"].get(i, True)
            )
            total = selected_combs * self.electrodes_per_comb
        valid = len(ecc_l)
        pct = 100 * valid / total if total > 0 else 0
        self.polar_plot.update_plot(
            ecc_l,
            pol_l,
            probe_l,
            self.probe_colors,
            self.state["r2"],
            total,
            self.state["rf_dot_size"],
            sz_list=sz_l,
            alpha=float(self.state.get("rf_alpha", 0.5)),
            centers_only=bool(self.state.get("rf_centers_only", False)),
        )

        if not self.state.get("selection_enabled", True):
            if self.selection_overlay_rf["ecc"]:
                self._clear_selection()
        else:
            self._refresh_locked_selection_overlay()
        self._update_selection_markers()

        selected_overlay = len(self.selection_overlay_rf["ecc"])
        sel_state = "Locked" if self.selection_locked else "Hover"
        if selected_overlay > 0:
            selection_text = f"{sel_state}: {selected_overlay} RFs"
            sel_sz = np.asarray(
                self.selection_overlay_rf.get("sz", []), dtype=np.float32
            )
            valid_sel_sz = sel_sz[np.isfinite(sel_sz) & (sel_sz > 0)]
            if valid_sel_sz.size > 0:
                selection_text += (
                    f" | sigma {float(np.nanmin(valid_sel_sz)):.2f}"
                    f"-{float(np.nanmax(valid_sel_sz)):.2f}°"
                    f" (μ {float(np.nanmean(valid_sel_sz)):.2f}°)"
                )
        else:
            selection_text = "None"
        drag_text = "On" if self.state.get("drag_implant_enabled", False) else "Off"

        anatomy_text = "Off"
        if self.state.get("show_anatomy", True):
            anatomy_text = self.state.get("anatomy_source", "auto")
            if self._anatomy_unavailable_note:
                anatomy_text += " (fallback)"

        if self._instance_mode_active():
            position_html = (
                f"<b>Placement:</b><br>"
                f"Selected implants: {selected_combs}/{len(self._scene_instances)}<br>"
                f"X: {self.state['x']:.1f} mm<br>"
                f"Y: {self.state['y']:.1f} mm<br>"
                f"Z: {self.state['z']:.1f} mm<br>"
                f"RotX: {self._last_ui_instance_rot.get('rx_deg', 0.0):.1f} deg<br>"
                f"RotY: {self._last_ui_instance_rot.get('ry_deg', 0.0):.1f} deg<br>"
                f"RotZ: {self._last_ui_instance_rot.get('rz_deg', 0.0):.1f} deg<br><br>"
            )
        else:
            position_html = (
                f"<b>Position:</b><br>"
                f"Depth: {self.state['depth']:.1f} mm<br>"
                f"X: {self.state['x']:.1f} mm<br>"
                f"Y: {self.state['y']:.1f} mm<br>"
                f"Z: {self.state['z']:.1f} mm<br><br>"
            )

        info_html = (
            position_html
            + f"<b>Active Design:</b> {self._current_design_name()}<br><br>"
            f"<b>Design Transform:</b><br>"
            f"RotX: {self.design_transform.get('rx_deg', 0.0):.1f} deg<br>"
            f"RotY: {self.design_transform.get('ry_deg', 0.0):.1f} deg<br>"
            f"RotZ: {self.design_transform.get('rz_deg', 0.0):.1f} deg<br>"
            f"Scale: {self.design_transform.get('scale', 1.0):.2f}<br>"
            f"Mirror X: {'ON' if self.design_transform.get('mirror_x', False) else 'OFF'}<br><br>"
            f"<b>Coverage:</b><br>"
            f"RFs: {valid}/{total} ({pct:.0f}%)<br>"
            f"R2 > {self.state['r2']:.2f}<br><br>"
            f"<b>pRF Source:</b> {self.state['prf_source']}<br>"
            f"<b>Anatomy:</b> {anatomy_text}<br>"
            f"<b>RF Picking:</b> {selection_text}<br>"
            f"<b>Implant Drag:</b> {drag_text}"
        )
        if self._anatomy_unavailable_note:
            info_html += f"<br><span style='color:#bbbbbb'>{self._anatomy_unavailable_note}</span>"
        if self._transform_warnings:
            info_html += "<br><br><b>Warnings:</b>"
            for msg in self._transform_warnings:
                info_html += f"<br><span style='color:#ffcc80'>{msg}</span>"
        self.info_label.setText(info_html)

        if self._first_render:
            self.plotter.reset_camera()
            self._first_render = False
        elif saved_camera:
            self.plotter.camera_position = saved_camera

    def closeEvent(self, event):
        self._hover_pick_timer.stop()
        self.plotter.close()
        event.accept()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vimplant2 Implant Explorer")
    parser.add_argument(
        "--dataset",
        choices=["nhp", "human_demo"],
        default="nhp",
        help="Dataset to load. 'nhp' is the default and works out of the box. "
        "'human_demo' requires fsaverage retinotopy maps under "
        "data/human/demo_subject/subjects/<subject>/T1w/<subject>/mri/ "
        "(the fsaverage subset is shipped in this repo).",
    )
    parser.add_argument("--human-subject", default=DEFAULT_HUMAN_SUBJECT)
    parser.add_argument(
        "--implant-config",
        default=None,
        help="Path to an ImplantSpec v2 JSON from the Implant Designer",
    )
    args = parser.parse_args()
    print("\nStarting OPTIMIZED Qt application...")
    app = QApplication(sys.argv)
    _ensure_valid_qt_app_font(app)
    window = ImplantExplorerWindow(
        dataset_id=args.dataset, human_subject_id=args.human_subject
    )
    if args.implant_config:
        window._load_implant_config(args.implant_config)
    window.show()
    print("\nReady: V2 optimized explorer with vectorized computations.")
    sys.exit(app.exec())
