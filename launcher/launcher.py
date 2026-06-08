"""
PhosLab Pipeline Launcher
Orquestador visual del pipeline completo:
phosLab → CSV → Simulador → Análisis → Aprendizaje
"""

import sys
import os
import shutil
import subprocess
import yaml
import csv
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QFileDialog,
    QTextEdit,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QTabWidget,
    QSplitter,
    QDialog,
    QMessageBox,
    QStackedWidget,
    QGridLayout,
    QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QProcess, QSize
from PyQt6.QtGui import QFont, QColor, QTextCursor, QPixmap, QIcon

# ── Rutas relativas al launcher ─────────────────────────────────────────────
LAUNCHER_DIR = Path(__file__).resolve().parent
PRACTICAS_DIR = LAUNCHER_DIR.parent
PHOSLAB_DIR = PRACTICAS_DIR / "implant_explorer"
SIMULADOR_DIR = PRACTICAS_DIR / "percept_mapper"
PARAMS_YAML = SIMULADOR_DIR / "config" / "params.yaml"
CSV_DEST_DIR = SIMULADOR_DIR / "config"
MAPPING_DIR = SIMULADOR_DIR / "mapping_experiments"
LOGS_DIR = SIMULADOR_DIR / "logs"
LEARNING_DIR = SIMULADOR_DIR / "learning_results"
CORRECTED_MAP_SUMMARY = LEARNING_DIR / "corrected_map_summary.json"
LOGOS_DIR = LAUNCHER_DIR / "assets" / "logos"
IMAGE_ICON_SIZES = {
    "error_comparison.png": QSize(460, 300),
    "neural_training.png": QSize(460, 300),
    "visual_field_comparison.png": QSize(920, 300),
}


# ═══════════════════════════════════════════════════════════════════════════
# HILO VIGILANTE DE CSV
# ═══════════════════════════════════════════════════════════════════════════


class CsvWatcher(QThread):
    csv_detected = pyqtSignal(str)
    watch_error = pyqtSignal(str)

    def __init__(self, watch_dir: Path, dest_dir: Path):
        super().__init__()
        self.watch_dir = watch_dir
        self.dest_dir = dest_dir
        self._running = True
        self._known = {}

    def run(self):
        self._known = self._snapshot()
        while self._running:
            current = self._snapshot()
            changed = [
                name for name, sig in current.items() if self._known.get(name) != sig
            ]
            for name in sorted(changed):
                self._copy_csv(self.watch_dir / name)
            self._known = current
            self.msleep(1000)

    def stop(self):
        self._running = False

    def _snapshot(self):
        snapshot = {}
        try:
            paths = list(self.watch_dir.glob("*.csv"))
        except Exception as e:
            self.watch_error.emit(f"No se puede leer {self.watch_dir}: {e}")
            return snapshot
        for path in paths:
            try:
                stat = path.stat()
                snapshot[path.name] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
        return snapshot

    def _copy_csv(self, src: Path):
        dst = self.dest_dir / src.name
        last_error = None
        for _ in range(3):
            try:
                shutil.copy2(src, dst)
                self.csv_detected.emit(str(dst))
                return
            except Exception as e:
                last_error = e
                self.msleep(150)
        self.watch_error.emit(f"No se pudo copiar {src.name}: {last_error}")


# ═══════════════════════════════════════════════════════════════════════════
# HILO MONITOR DE PROCESO
# ═══════════════════════════════════════════════════════════════════════════


class ProcessMonitor(QThread):
    """Monitoriza un subproceso y emite señales cuando produce output o termina."""

    output_line = pyqtSignal(str)
    process_finished = pyqtSignal(int)  # código de salida

    def __init__(self, proc: subprocess.Popen):
        super().__init__()
        self._proc = proc

    def run(self):
        try:
            for line in iter(self._proc.stdout.readline, b""):
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    self.output_line.emit(decoded)
            self._proc.stdout.close()
            returncode = self._proc.wait()
            self.process_finished.emit(returncode)
        except Exception:
            self.process_finished.emit(-1)


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES CSV
# ═══════════════════════════════════════════════════════════════════════════


def read_implant_ids_from_csv(csv_path: str) -> dict:
    """
    Lee un CSV de phosLab y devuelve un dict:
      { implant_id: [electrode_index, ...] }
    """
    result = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = row.get("implant_id", "").strip()
                eidx = row.get("electrode_index", "").strip()
                if iid and eidx:
                    try:
                        idx = int(eidx)
                        result.setdefault(iid, [])
                        if idx not in result[iid]:
                            result[iid].append(idx)
                    except ValueError:
                        pass
        for k in result:
            result[k].sort()
    except Exception:
        pass
    return result


def load_yaml_safe(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def save_yaml_partial(path: Path, updates: dict):
    """
    Actualiza solo las claves especificadas en updates (deep merge),
    dejando el resto del YAML intacto.
    """
    params = load_yaml_safe(path)
    _deep_update(params, updates)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(params, f, allow_unicode=True, sort_keys=False)


def _deep_update(base: dict, updates: dict):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# WIDGET MAPA POLAR (inline, sin pyqtgraph)
# ═══════════════════════════════════════════════════════════════════════════


class PolarMapWidget(QWidget):
    """Mapa polar de campos receptivos estilo phosLab, dibujado con QPainter."""

    def __init__(self, parent=None, max_ecc: float = 15.0):
        super().__init__(parent)
        self.max_ecc = max_ecc
        self.setMinimumSize(480, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #000000;")

        # Datos: cada entrada es (x_deg_list, y_deg_list, color_hex, label)
        self._series: list[tuple] = []

    def set_series(self, series: list[tuple]):
        """series = [(x_deg_list, y_deg_list, color_hex, label), ...]"""
        self._series = series
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
        from PyQt6.QtCore import QPointF, QRectF

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        # Radio en pixeles que corresponde a max_ecc
        margin = 44
        radius = min(w, h) / 2 - margin

        # Fondo
        p.fillRect(0, 0, w, h, QColor("#000000"))

        # Círculos y etiquetas de excentricidad
        p.setPen(QPen(QColor("#ffffff"), 1))
        step = 2 if self.max_ecc <= 10 else (5 if self.max_ecc <= 30 else 10)
        font_small = QFont("Segoe UI", 8)
        p.setFont(font_small)
        r_val = step
        while r_val <= self.max_ecc:
            r_px = (r_val / self.max_ecc) * radius
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawEllipse(QPointF(cx, cy), r_px, r_px)
            p.setPen(QColor("#e2e8f0"))
            p.drawText(QPointF(cx + r_px + 2, cy - 2), f"{r_val} deg")
            r_val += step

        # Ejes radiales y etiquetas de ángulo
        p.setPen(QPen(QColor("#ffffff"), 1))
        font_ang = QFont("Segoe UI", 9, QFont.Weight.Bold)
        p.setFont(font_ang)
        labels = {
            0: "0",
            45: "45",
            90: "90",
            135: "135",
            180: "180",
            225: "225",
            270: "270",
            315: "315",
        }
        for ang_deg, lbl in labels.items():
            ang_rad = math.radians(ang_deg)
            x1 = cx + radius * math.cos(ang_rad)
            y1 = cy - radius * math.sin(ang_rad)
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(int(cx), int(cy), int(x1), int(y1))
            lx = cx + (radius + 18) * math.cos(ang_rad)
            ly = cy - (radius + 18) * math.sin(ang_rad)
            p.setPen(QColor("#e2e8f0"))
            p.drawText(QPointF(lx - 14, ly + 5), lbl)

        # Puntos de cada serie
        dot_r = max(3.5, radius * 0.025)
        for xs, ys, color_hex, _label in self._series:
            col = QColor(color_hex)
            col.setAlphaF(0.82)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            for x_deg, y_deg in zip(xs, ys):
                ecc = math.sqrt(x_deg**2 + y_deg**2)
                if ecc > self.max_ecc:
                    continue
                px = cx + (x_deg / self.max_ecc) * radius
                py = cy - (y_deg / self.max_ecc) * radius
                p.drawEllipse(QPointF(px, py), dot_r, dot_r)

        # Leyenda
        if self._series:
            lx, ly = 10, h - 10 - len(self._series) * 20
            font_leg = QFont("Segoe UI", 9)
            p.setFont(font_leg)
            for i, (_xs, _ys, color_hex, label) in enumerate(self._series):
                col = QColor(color_hex)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(col))
                p.drawEllipse(QPointF(lx + 6, ly + i * 20 + 6), 5, 5)
                p.setPen(QColor("#ffffff"))
                p.drawText(QPointF(lx + 18, ly + i * 20 + 11), label)

        p.end()


# ═══════════════════════════════════════════════════════════════════════════
# VENTANA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════


class PipelineLauncher(QMainWindow):

    STYLE = """
    QMainWindow, QWidget {
        background-color: #0a0a0a;
        color: #e9ecf5;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 15px;
    }
    QGroupBox {
        border: 0.5px solid rgba(123,108,252,0.35);
        border-radius: 8px;
        margin-top: 14px;
        padding-top: 12px;
        color: rgba(123,108,252,0.7);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
    }
    QPushButton {
        background-color: rgba(123,108,252,0.12);
        border: 0.5px solid rgba(123,108,252,0.55);
        border-radius: 6px;
        color: #d7d2ff;
        padding: 7px 16px;
        font-size: 14px;
    }
    QPushButton:hover  { background-color: rgba(0,212,255,0.14); color: #e6fbff; }
    QPushButton:pressed{ background-color: rgba(123,108,252,0.3); }
    QPushButton:disabled{
        background-color: rgba(60,60,60,0.1);
        border-color: rgba(80,80,80,0.25);
        color: #343446;
    }
    QPushButton#btn_green {
        background-color: rgba(0,255,160,0.08);
        border-color: rgba(0,255,160,0.35);
        color: #00ffa0;
    }
    QPushButton#btn_green:hover { background-color: rgba(0,255,160,0.18); }
    QPushButton#btn_amber {
        background-color: rgba(255,200,0,0.08);
        border-color: rgba(255,200,0,0.35);
        color: #ffc800;
    }
    QPushButton#btn_amber:hover { background-color: rgba(255,200,0,0.18); }
    QPushButton#btn_red {
        background-color: rgba(255,60,60,0.08);
        border-color: rgba(255,60,60,0.35);
        color: #ff3c3c;
    }
    QSlider::groove:horizontal {
        background: #231a44; height: 4px; border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #00d4ff; width: 14px; height: 14px;
        margin: -5px 0; border-radius: 7px;
    }
    QSlider::sub-page:horizontal { background: #00d4ff; border-radius: 2px; }
    QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
        background-color: #111111;
        border: 0.5px solid rgba(123,108,252,0.35);
        border-radius: 5px;
        color: #e9ecf5;
        padding: 3px 6px;
    }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background-color: #111111;
        color: #e2e8f0;
        selection-background-color: rgba(0,212,255,0.25);
    }
    QListWidget::item { padding: 4px 6px; border-radius: 4px; }
    QListWidget::item:selected {
        background: rgba(0,212,255,0.18);
        color: #00d4ff;
    }
    QListWidget::item:hover { background: rgba(123,108,252,0.12); }
    QCheckBox { color: #e9ecf5; spacing: 6px; }
    QCheckBox::indicator {
        width: 14px; height: 14px;
        border: 1px solid rgba(123,108,252,0.5);
        border-radius: 3px;
        background: #111111;
    }
    QCheckBox::indicator:checked {
        background: #00d4ff;
        border-color: #00d4ff;
    }
    QTextEdit {
        background-color: #050505;
        border: 0.5px solid rgba(123,108,252,0.28);
        border-radius: 6px;
        color: #7fe8ff;
        font-family: "Cascadia Code", "Consolas", monospace;
        font-size: 13px;
    }
    QTabWidget::pane {
        border: 0.5px solid rgba(123,108,252,0.25);
        border-radius: 6px;
        background: #0a0a0a;
    }
    QTabBar::tab {
        background: transparent;
        color: #3a3b52;
        padding: 6px 16px;
        border: none;
        font-size: 14px;
    }
    QTabBar::tab:selected { color: #00d4ff; border-bottom: 2px solid #00d4ff; }
    QTabBar::tab:hover { color: #e9ecf5; }
    QScrollBar:vertical {
        background: #0a0a0a; width: 6px; border-radius: 3px;
    }
    QScrollBar::handle:vertical { background: #231a44; border-radius: 3px; }
    QScrollBar:horizontal {
        background: #0a0a0a; height: 6px; border-radius: 3px;
    }
    QScrollBar::handle:horizontal { background: #231a44; border-radius: 3px; }
    QLabel#title {
        font-size: 22px; font-weight: 500; color: #e9ecf5; padding: 4px 0;
    }
    QLabel#subtitle { font-size: 13px; color: #3d3f5a; }
    QFrame#divider { background: rgba(123,108,252,0.2); max-height: 1px; }
    QFrame#card {
        background: #0f0f0f;
        border: 0.5px solid rgba(123,108,252,0.25);
        border-radius: 10px;
    }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhosLab — Pipeline Launcher")
        self.setMinimumSize(980, 760)
        self.setStyleSheet(self.STYLE)

        self._phoslab_proc = None
        self._sim_proc = None
        self._sim_monitor = None
        self._csv_watcher = None
        self._current_csv = None
        self._implant_data = {}  # {implant_id: [electrode_indices]}
        self._step_states = [False, False, False, False, False]
        self._params = {}

        self._build_ui()
        self._load_params()
        self._update_step_states()

        if PHOSLAB_DIR.exists():
            self._start_csv_watcher()

    # ──────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DE UI
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        div = QFrame()
        div.setFixedWidth(1)
        div.setStyleSheet("background: rgba(123,108,252,0.25);")
        root.addWidget(div)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_page_launcher())  # 0
        self._stack.addWidget(self._build_page_params())  # 1
        self._stack.addWidget(self._build_page_analysis())  # 2
        self._stack.addWidget(self._build_page_learning())  # 3
        self._stack.addWidget(self._build_page_optimized_map())  # 4
        root.addWidget(self._stack, stretch=1)

    def _build_sidebar(self):
        w = QWidget()
        w.setFixedWidth(210)
        w.setStyleSheet("background: #0a0a0a;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 22, 14, 22)
        lay.setSpacing(4)

        logo = QLabel("PhosLab")
        logo.setFont(QFont("Segoe UI", 18, QFont.Weight.Medium))
        logo.setStyleSheet("color: #7b6cff; padding-bottom: 2px;")
        lay.addWidget(logo)

        sub = QLabel("Visual Prosthesis Pipeline")
        sub.setObjectName("subtitle")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background: rgba(123,108,252,0.25); margin: 10px 0;")
        lay.addWidget(div)

        nav_items = [
            ("▶  Lanzador", 0),
            ("  Parámetros", 1),
            ("  Análisis", 2),
            ("  Aprendizaje", 3),
            ("  Mapa optimizado", 4),
        ]
        self._nav_btns = []
        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent; border: none;
                    color: #3d3f5a; text-align: left;
                    padding: 8px 10px; border-radius: 6px; font-size: 15px;
                }
                QPushButton:hover   { background: rgba(0,212,255,0.08); color: #e9ecf5; }
                QPushButton:checked { background: rgba(123,108,252,0.18); color: #00d4ff;
                                    border-left: 2px solid #00d4ff; }
            """)

            btn.clicked.connect(lambda _, i=idx: self._show_page(i))
            lay.addWidget(btn)
            self._nav_btns.append(btn)

        self._nav_btns[0].setChecked(True)
        lay.addStretch()

        # Status pills
        self._status_labels = {}
        for key, label in [
            ("phoslab", "PhosLab"),
            ("watcher", "CSV watcher"),
            ("simulador", "Simulador"),
            ("learning", "Aprendizaje"),
        ]:
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setFixedWidth(14)
            dot.setStyleSheet("color: #1a1a1a; font-size: 12px;")
            txt = QLabel(label)
            txt.setStyleSheet("color: #2d3748; font-size: 13px;")
            row.addWidget(dot)
            row.addWidget(txt)
            row.addStretch()
            lay.addLayout(row)
            self._status_labels[key] = (dot, txt)

        return w

    # ── PÁGINA LANZADOR ────────────────────────────────────────────────────

    def _build_page_launcher(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("Pipeline")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()
        new_exp_btn = QPushButton("+ Nuevo experimento completo")
        new_exp_btn.setObjectName("btn_amber")
        new_exp_btn.clicked.connect(self._reset_pipeline)
        hdr.addWidget(new_exp_btn)
        lay.addLayout(hdr)

        # Pasos
        self._step_widgets = []
        steps = [
            (
                "1",
                "Implant Explorer — Colocación del implante",
                "Coloca el implante y exporta el CSV de campos receptivos",
                self._launch_phoslab,
                "Abrir Implant Explorer",
                "implant_explorer.png",
            ),
            (
                "2",
                "Seleccionar Campos Receptivos (Receptive Fields)",
                "Exporta campos receptivos desde Implant Explorer o selecciona un CSV existente",
                self._select_csv_manual,
                "Seleccionar\nCampos Receptivos",
                "receptive_fields.png",
            ),
            (
                "3",
                "Percept Mapper — Simulación ICMS",
                "Lanza el experimento con los parámetros configurados",
                self._launch_simulator,
                "Lanzar experimento",
                "simulator.png",
            ),
            (
                "4",
                "Análisis",
                "Ver resultados del experimento completado",
                lambda: self._show_page(2),
                "Análisis",
                "analysis.png",
            ),
            (
                "5",
                "Learning — Corrección del mapa",
                "Entrena el modelo bayesiano o red neuronal",
                self._run_learning,
                "Entrenar modelo",
                "learning.png",
            ),
        ]
        steps_row = QHBoxLayout()
        steps_row.setSpacing(14)
        for num, title_s, desc, fn, btn_label, logo in steps:
            card = self._make_step_card(num, title_s, desc, fn, btn_label, logo)
            steps_row.addWidget(card)
            self._step_widgets.append(card)
        steps_row.addStretch()

        steps_widget = QWidget()
        steps_widget.setLayout(steps_row)
        steps_scroll = QScrollArea()
        steps_scroll.setWidgetResizable(True)
        steps_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        steps_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        steps_scroll.setFrameShape(QFrame.Shape.NoFrame)
        steps_scroll.setFixedHeight(360)
        steps_scroll.setWidget(steps_widget)
        steps_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        lay.addWidget(steps_scroll)

        # CSV watcher status
        wb_row = QHBoxLayout()
        self._watcher_icon = QLabel("●")
        self._watcher_icon.setStyleSheet("color: #ffc800; font-size: 16px;")
        self._watcher_path = QLabel("Vigilando exportaciones de Implant Explorer...")
        self._watcher_path.setStyleSheet("color: #2d3748; font-size: 13px;")
        wb_row.addWidget(self._watcher_icon)
        wb_row.addWidget(self._watcher_path)
        wb_row.addStretch()
        lay.addLayout(wb_row)

        # CSV seleccionado
        self._csv_label = QLabel("Sin CSV seleccionado")
        self._csv_label.setStyleSheet(
            "color: #2d3748; font-size: 13px; padding-left: 4px;"
        )
        lay.addWidget(self._csv_label)

        # Log
        log_grp = QGroupBox("Log del sistema")
        lg = QVBoxLayout(log_grp)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(130)
        lg.addWidget(self._log)
        lay.addWidget(log_grp)

        return w

    def _make_step_card(self, num, title, desc, fn, btn_label, logo_file=None):
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumSize(240, 320)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        ICON_SIZE = 180

        logo_path = (LOGOS_DIR / logo_file) if logo_file else None
        logo_label = None
        if logo_path and logo_path.exists():
            logo_label = QLabel()
            pix = QPixmap(str(logo_path)).scaled(
                ICON_SIZE,
                ICON_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(pix)
            logo_label.setFixedSize(ICON_SIZE, ICON_SIZE)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_label.setStyleSheet("background: transparent;")
            lay.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        num_label = QLabel(num)
        num_label.setFixedSize(ICON_SIZE, ICON_SIZE)
        num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_label.setStyleSheet(
            f"border: 2px solid #374151; border-radius: {ICON_SIZE // 2}px; "
            "color: #6b7280; font-size: 72px;"
        )
        lay.addWidget(num_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        if logo_label is not None:
            num_label.hide()

        lay.addStretch(1)

        btn = QPushButton(btn_label)
        btn.setMinimumHeight(68)
        btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(123,108,252,0.12);
                border: 0.5px solid rgba(123,108,252,0.55);
                border-radius: 8px;
                color: #d7d2ff;
                padding: 8px 10px;
                font-size: 17px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: rgba(0,212,255,0.14); color: #e6fbff; }
            QPushButton:pressed { background-color: rgba(123,108,252,0.3); }
        """)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.clicked.connect(fn)
        lay.addWidget(btn)

        card._num_label = num_label
        card._logo_label = logo_label
        card._btn = btn
        return card

    # ── PÁGINA PARÁMETROS ──────────────────────────────────────────────────

    def _build_page_params(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        title = QLabel("Parámetros del experimento")
        title.setObjectName("title")
        lay.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        inner = QWidget()
        ilay = QVBoxLayout(inner)
        ilay.setSpacing(14)

        # ── 1. CSV activo ──────────────────────────────────────────────
        csv_grp = QGroupBox("CSV activo")
        cg = QVBoxLayout(csv_grp)
        self._params_csv_label = QLabel("Sin CSV cargado")
        self._params_csv_label.setStyleSheet("color: #ffc800; font-size: 13px;")
        cg.addWidget(self._params_csv_label)
        ilay.addWidget(csv_grp)

        # ── 2. Implant IDs ─────────────────────────────────────────────
        implant_grp = QGroupBox("Implant IDs disponibles")
        ig = QVBoxLayout(implant_grp)
        self._implant_list = QListWidget()
        self._implant_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._implant_list.setFixedHeight(90)
        self._implant_list.setToolTip(
            "Selecciona uno o más implant IDs para el experimento"
        )
        ig.addWidget(self._implant_list)
        ilay.addWidget(implant_grp)

        # ── 3. Input mode ──────────────────────────────────────────────
        input_grp = QGroupBox("Modo de entrada")
        inp_lay = QHBoxLayout(input_grp)
        self._input_mode_combo = QComboBox()
        self._input_mode_combo.addItems(["mouse", "gaze", "pupil"])
        self._input_mode_combo.setFixedWidth(120)
        inp_lay.addWidget(QLabel("Dispositivo de entrada:"))
        inp_lay.addWidget(self._input_mode_combo)

        self._calibrate_btn = QPushButton("Calibrar gaze")
        self._calibrate_btn.setFixedWidth(130)
        self._calibrate_btn.setVisible(False)
        self._calibrate_btn.clicked.connect(self._run_gaze_calibration)
        inp_lay.addWidget(self._calibrate_btn)

        inp_lay.addStretch()
        ilay.addWidget(input_grp)

        self._input_mode_combo.currentTextChanged.connect(self._on_input_mode_changed)

        # ── 4. Corriente default ───────────────────────────────────────
        curr_grp = QGroupBox("Corriente default")
        curr_lay = QHBoxLayout(curr_grp)
        curr_lay.addWidget(QLabel("default_current_uA:"))
        self._default_current_spin = QSpinBox()
        self._default_current_spin.setRange(0, 1000)
        self._default_current_spin.setSuffix(" µA")
        self._default_current_spin.setFixedWidth(110)
        curr_lay.addWidget(self._default_current_spin)
        curr_lay.addStretch()
        ilay.addWidget(curr_grp)

        # ── 5. Tiempos ─────────────────────────────────────────────────
        timing_grp = QGroupBox("Tiempos (ms)")
        tg = QGridLayout(timing_grp)
        self._timing_spins = {}
        timing_params = [
            ("Prestimulación", "prestimulation", 50, 5000),
            ("Estimulación", "stimulation", 50, 5000),
            ("Postestimulación", "poststimulation", 50, 5000),
            ("Interestímulo", "interstimulation", 100, 10000),
        ]
        for row_i, (label, key, lo, hi) in enumerate(timing_params):
            tg.addWidget(QLabel(label), row_i, 0)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSuffix(" ms")
            spin.setFixedWidth(110)
            tg.addWidget(spin, row_i, 1)
            self._timing_spins[key] = spin
        ilay.addWidget(timing_grp)

        # ── 6. Modo experimento ────────────────────────────────────────
        mode_grp = QGroupBox("Modo de experimento")
        mg = QVBoxLayout(mode_grp)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["mapping", "standard"])
        self._mode_combo.setFixedWidth(140)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mg.addWidget(self._mode_combo)
        ilay.addWidget(mode_grp)

        # ── 6a. Parámetros MAPPING ─────────────────────────────────────
        self._mapping_grp = QGroupBox("Configuración — Modo Mapping")
        map_lay = QVBoxLayout(self._mapping_grp)

        # Electrodos por implant (bloques dinámicos)
        electrodes_lbl = QLabel("Electrodos a estimular por implant:")
        electrodes_lbl.setStyleSheet("color: #c9d1e0; font-size: 14px;")
        map_lay.addWidget(electrodes_lbl)

        # Contenedor scroll para los bloques de implant
        self._implant_electrodes_scroll = QScrollArea()
        self._implant_electrodes_scroll.setWidgetResizable(True)
        self._implant_electrodes_scroll.setFixedHeight(160)
        self._implant_electrodes_scroll.setStyleSheet("QScrollArea { border: none; }")
        self._implant_electrodes_container = QWidget()
        self._implant_electrodes_layout = QVBoxLayout(
            self._implant_electrodes_container
        )
        self._implant_electrodes_layout.setSpacing(8)
        self._implant_electrodes_layout.setContentsMargins(0, 0, 0, 0)
        self._implant_electrodes_layout.addStretch()
        self._implant_electrodes_scroll.setWidget(self._implant_electrodes_container)
        map_lay.addWidget(self._implant_electrodes_scroll)

        # Dict que guarda los QLineEdit por implant_id
        self._implant_electrode_edits = {}  # {implant_id: QLineEdit} — mapping
        self._std_implant_electrode_edits = {}  # {implant_id: QLineEdit} — standard

        no_csv_lbl = QLabel("Carga un CSV para ver los implants disponibles")
        no_csv_lbl.setStyleSheet("color: #4a5568; font-size: 13px; padding: 8px;")
        self._implant_electrodes_layout.insertWidget(0, no_csv_lbl)
        self._implant_no_csv_lbl = no_csv_lbl

        # num_repetitions
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel("Repeticiones por electrodo:"))
        self._rep_spin = QSpinBox()
        self._rep_spin.setRange(1, 50)
        self._rep_spin.setFixedWidth(90)
        rep_row.addWidget(self._rep_spin)
        rep_row.addStretch()
        map_lay.addLayout(rep_row)

        # stimulation mapping
        stim_map_grp = QGroupBox("Estimulación (mapping)")
        sm_lay = QGridLayout(stim_map_grp)
        self._map_stim_spins = {}
        map_stim_params = [
            ("Pulse width (µs)", "pulse_width", 50, 1000, " µs"),
            ("Frecuencia (Hz)", "frequency", 1, 300, " Hz"),
        ]
        for ri, (lbl, key, lo, hi, sfx) in enumerate(map_stim_params):
            sm_lay.addWidget(QLabel(lbl), ri, 0)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSuffix(sfx)
            spin.setFixedWidth(110)
            sm_lay.addWidget(spin, ri, 1)
            self._map_stim_spins[key] = spin
        sm_lay.addWidget(
            QLabel("Corrientes mapping (uA, lista o dict):"), len(map_stim_params), 0
        )
        self._map_currents_edit = QLineEdit()
        self._map_currents_edit.setPlaceholderText(
            "30:90,45:120 o {default_uA: 90, overrides: {5: 120}}"
        )
        self._map_currents_edit.setStyleSheet(
            "background:#111111; border:0.5px solid rgba(123,108,252,0.35); "
            "border-radius:5px; color:#e9ecf5; padding:3px 6px;"
        )
        sm_lay.addWidget(self._map_currents_edit, len(map_stim_params), 1)
        stim_map_grp.setLayout(sm_lay)
        map_lay.addWidget(stim_map_grp)
        ilay.addWidget(self._mapping_grp)

        # ── 6b. Parámetros STANDARD ────────────────────────────────────
        self._standard_grp = QGroupBox("Configuración — Modo Standard")
        std_lay = QVBoxLayout(self._standard_grp)

        # electrode_selection
        esel_grp = QGroupBox("Selección de electrodos")
        esel_lay = QVBoxLayout(esel_grp)
        self._esel_combo = QComboBox()
        self._esel_combo.addItems(["all", "range", "manual"])
        self._esel_combo.setFixedWidth(120)
        self._esel_combo.currentTextChanged.connect(self._on_esel_changed)
        esel_lay.addWidget(self._esel_combo)

        # range params
        self._esel_range_widget = QWidget()
        rng_lay = QHBoxLayout(self._esel_range_widget)
        rng_lay.setContentsMargins(0, 0, 0, 0)
        rng_lay.addWidget(QLabel("Inicio:"))
        self._esel_start_spin = QSpinBox()
        self._esel_start_spin.setRange(0, 9999)
        self._esel_start_spin.setFixedWidth(80)
        rng_lay.addWidget(self._esel_start_spin)
        rng_lay.addWidget(QLabel("Fin:"))
        self._esel_end_spin = QSpinBox()
        self._esel_end_spin.setRange(0, 9999)
        self._esel_end_spin.setFixedWidth(80)
        rng_lay.addWidget(self._esel_end_spin)
        rng_lay.addWidget(QLabel("Paso:"))
        self._esel_step_spin = QSpinBox()
        self._esel_step_spin.setRange(1, 100)
        self._esel_step_spin.setFixedWidth(70)
        rng_lay.addWidget(self._esel_step_spin)
        rng_lay.addStretch()
        esel_lay.addWidget(self._esel_range_widget)
        self._esel_range_widget.hide()

        # manual indices
        self._esel_manual_widget = QWidget()
        man_lay = QHBoxLayout(self._esel_manual_widget)
        man_lay.setContentsMargins(0, 0, 0, 0)
        man_lay.addWidget(QLabel("Índices (ej: 0,1,5,10):"))
        self._esel_manual_edit = QLineEdit()
        self._esel_manual_edit.setPlaceholderText("0,1,2,3")
        self._esel_manual_edit.setStyleSheet(
            "background:#16213e; border:0.5px solid rgba(100,120,200,0.25); "
            "border-radius:5px; color:#c9d1e0; padding:3px 6px;"
        )
        man_lay.addWidget(self._esel_manual_edit)
        esel_lay.addWidget(self._esel_manual_widget)
        self._esel_manual_widget.hide()

        # ── Electrodos disponibles (desde CSV) ─────────────────────────
        avail_lbl = QLabel("Electrodos disponibles:")
        avail_lbl.setStyleSheet("color: #c9d1e0; font-size: 14px; margin-top: 6px;")
        esel_lay.addWidget(avail_lbl)

        self._std_electrodes_scroll = QScrollArea()
        self._std_electrodes_scroll.setWidgetResizable(True)
        self._std_electrodes_scroll.setFixedHeight(130)
        self._std_electrodes_scroll.setStyleSheet("QScrollArea { border: none; }")
        self._std_electrodes_container = QWidget()
        self._std_electrodes_layout = QVBoxLayout(self._std_electrodes_container)
        self._std_electrodes_layout.setSpacing(6)
        self._std_electrodes_layout.setContentsMargins(0, 0, 0, 0)
        self._std_electrodes_layout.addStretch()
        self._std_electrodes_scroll.setWidget(self._std_electrodes_container)

        _std_no_csv_lbl = QLabel("Carga un CSV para ver los electrodos disponibles")
        _std_no_csv_lbl.setStyleSheet("color: #4a5568; font-size: 13px; padding: 8px;")
        self._std_electrodes_layout.insertWidget(0, _std_no_csv_lbl)

        esel_lay.addWidget(self._std_electrodes_scroll)

        std_lay.addWidget(esel_grp)

        # stimulation standard
        stim_std_grp = QGroupBox("Estimulación (standard)")
        ss_lay = QGridLayout(stim_std_grp)
        self._std_stim_spins = {}
        std_stim_params = [
            ("Pulse width (µs)", "pulse_width", 50, 1000, " µs"),
            ("Frecuencia (Hz)", "frequency", 1, 300, " Hz"),
        ]
        for ri, (lbl, key, lo, hi, sfx) in enumerate(std_stim_params):
            ss_lay.addWidget(QLabel(lbl), ri, 0)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSuffix(sfx)
            spin.setFixedWidth(110)
            ss_lay.addWidget(spin, ri, 1)
            self._std_stim_spins[key] = spin

        # stimulation_currents_standard_uA (lista o dict)
        ss_lay.addWidget(QLabel("Corrientes standard (uA, lista o dict):"), 2, 0)
        self._std_currents_edit = QLineEdit()
        self._std_currents_edit.setPlaceholderText(
            "150,30,90,40 o {default_uA: 90, overrides: {5: 120}}"
        )
        self._std_currents_edit.setStyleSheet(
            "background:#16213e; border:0.5px solid rgba(100,120,200,0.25); "
            "border-radius:5px; color:#c9d1e0; padding:3px 6px;"
        )
        ss_lay.addWidget(self._std_currents_edit, 2, 1)
        stim_std_grp.setLayout(ss_lay)
        std_lay.addWidget(stim_std_grp)
        ilay.addWidget(self._standard_grp)

        # ── 7. Guardar ─────────────────────────────────────────────────
        save_btn = QPushButton("Guardar parámetros")
        save_btn.setObjectName("btn_green")
        save_btn.clicked.connect(self._save_params)
        ilay.addWidget(save_btn)

        ilay.addStretch()
        scroll.setWidget(inner)
        lay.addWidget(scroll, stretch=1)
        return w

    # ── PÁGINA ANÁLISIS ────────────────────────────────────────────────────

    def _build_page_analysis(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        hdr = QHBoxLayout()
        title = QLabel("Resultados del análisis")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()
        delete_btn = QPushButton("Eliminar")
        delete_btn.setObjectName("btn_red")
        delete_btn.clicked.connect(self._delete_selected_analysis)
        hdr.addWidget(delete_btn)
        refresh_btn = QPushButton("↻ Actualizar")
        refresh_btn.clicked.connect(self._refresh_analysis)
        hdr.addWidget(refresh_btn)
        lay.addLayout(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Arbol de experimentos
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Experimentos guardados:"))
        self._exp_tree = QTreeWidget()
        self._exp_tree.setHeaderHidden(True)
        self._exp_tree.currentItemChanged.connect(self._on_exp_selected)
        ll.addWidget(self._exp_tree)
        splitter.addWidget(left)

        # Detalle del experimento
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self._exp_detail = QTextEdit()
        self._exp_detail.setReadOnly(True)
        rl.addWidget(QLabel("Detalle:"))
        rl.addWidget(self._exp_detail)
        splitter.addWidget(right)

        splitter.setSizes([280, 600])
        lay.addWidget(splitter, stretch=1)

        self._refresh_analysis()
        return w

    # ── PÁGINA APRENDIZAJE ─────────────────────────────────────────────────

    def _build_page_learning(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        title = QLabel("Módulo de aprendizaje")
        title.setObjectName("title")
        lay.addWidget(title)

        # Modelo / Train
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Modelo:"))
        self._learn_combo = QComboBox()
        self._learn_combo.addItems(["both", "bayesian", "neural"])
        self._learn_combo.setFixedWidth(130)
        model_row.addWidget(self._learn_combo)
        model_row.addSpacing(16)
        model_row.addWidget(QLabel("Train:"))
        self._learn_scope_combo = QComboBox()
        self._learn_scope_combo.addItems(
            ["all", "latest_mapping", "latest_standard", "latest_any"]
        )
        self._learn_scope_combo.setFixedWidth(160)
        model_row.addWidget(self._learn_scope_combo)
        model_row.addSpacing(16)
        model_row.addWidget(QLabel("Datos:"))
        self._learn_input_mode_combo = QComboBox()
        self._learn_input_mode_combo.addItems(["all", "pupil", "gaze", "mouse"])
        self._learn_input_mode_combo.setFixedWidth(100)
        model_row.addWidget(self._learn_input_mode_combo)
        model_row.addStretch()
        run_btn = QPushButton("▶  Ejecutar aprendizaje")
        run_btn.setObjectName("btn_green")
        run_btn.clicked.connect(self._run_learning)
        model_row.addWidget(run_btn)
        lay.addLayout(model_row)

        # Test
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Test:"))
        self._test_mode_combo = QComboBox()
        self._test_mode_combo.addItems(["none", "last", "select"])
        self._test_mode_combo.setFixedWidth(120)
        self._test_mode_combo.currentTextChanged.connect(self._on_test_mode_changed)
        test_row.addWidget(self._test_mode_combo)
        test_row.addSpacing(10)
        test_row.addWidget(QLabel("Origen:"))
        self._test_source_combo = QComboBox()
        self._test_source_combo.addItems(["any", "mapping", "standard"])
        self._test_source_combo.setFixedWidth(130)
        test_row.addWidget(self._test_source_combo)
        test_row.addSpacing(10)
        test_row.addWidget(QLabel("Experimento:"))
        self._test_exp_combo = QComboBox()
        self._test_exp_combo.setFixedWidth(360)
        test_row.addWidget(self._test_exp_combo)
        test_row.addStretch()
        lay.addLayout(test_row)

        hint = QLabel(
            "Train: datos usados para entrenar. Test: experimento nuevo para evaluar generalizacion."
        )
        hint.setStyleSheet("color: #4a5568; font-size: 13px;")
        lay.addWidget(hint)

        # Resultados
        results_grp = QGroupBox("Resultados del aprendizaje")
        rg = QVBoxLayout(results_grp)

        results_split = QSplitter(Qt.Orientation.Horizontal)

        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)

        summary_box = QWidget()
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.addWidget(QLabel("dataset_summary.json:"))
        self._dataset_text = QTextEdit()
        self._dataset_text.setReadOnly(True)
        self._dataset_text.setMinimumHeight(220)
        self._dataset_text.setStyleSheet(
            "font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 14px;"
        )
        summary_layout.addWidget(self._dataset_text)
        left_layout.addWidget(summary_box)

        metrics_box = QWidget()
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.addWidget(QLabel("evaluation_metrics.json:"))
        self._metrics_text = QTextEdit()
        self._metrics_text.setReadOnly(True)
        self._metrics_text.setMinimumHeight(220)
        self._metrics_text.setStyleSheet(
            "font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 14px;"
        )
        metrics_layout.addWidget(self._metrics_text)
        left_layout.addWidget(metrics_box)

        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)

        img_grid = QGridLayout()
        img_specs = [
            (
                "_img_error",
                "error_comparison.png",
                "error_comparison.png",
                0,
                0,
                1,
                1,
                (460, 300),
            ),
            (
                "_img_neural",
                "neural_training.png",
                "neural_training.png",
                0,
                1,
                1,
                1,
                (460, 300),
            ),
            (
                "_img_visual",
                "visual_field_comparison.png",
                "visual_field_comparison.png",
                1,
                0,
                1,
                2,
                (920, 300),
            ),
        ]
        for attr, label, fname, row, col, rs, cs, size in img_specs:
            col_box = QVBoxLayout()
            btn = QPushButton("Sin imagen")
            btn.setMinimumSize(*size)
            btn.setStyleSheet(
                "QPushButton { background:#0a0f1a; border:0.5px solid rgba(100,120,200,0.15); border-radius:6px; color:#6b8cba; }"
                "QPushButton:hover { border-color: rgba(129,140,248,0.6); }"
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("image_label", label)
            btn.clicked.connect(
                lambda _, b=btn: self._open_image_dialog(b.property("image_path"))
            )
            col_box.addWidget(btn)
            img_grid.addLayout(col_box, row, col, rs, cs)
            setattr(self, attr, btn)
        right_layout.addLayout(img_grid)

        results_split.addWidget(left_col)
        results_split.addWidget(right_col)
        results_split.setSizes([520, 760])
        rg.addWidget(results_split)

        actions_row = QHBoxLayout()
        refresh_learn_btn = QPushButton("↻ Cargar resultados")
        refresh_learn_btn.clicked.connect(self._refresh_learning)
        actions_row.addWidget(refresh_learn_btn)

        export_learn_btn = QPushButton("Guardar learning_results (.zip)")
        export_learn_btn.setObjectName("btn_amber")
        export_learn_btn.clicked.connect(self._export_learning_zip)
        actions_row.addWidget(export_learn_btn)
        actions_row.addStretch()
        rg.addLayout(actions_row)

        lay.addWidget(results_grp, stretch=1)
        self._refresh_learning_experiments()
        self._refresh_learning()
        return w

    # ── PÁGINA MAPA OPTIMIZADO ─────────────────────────────────────────────

    def _build_page_optimized_map(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("Mapa de fosfenos optimizado")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()
        refresh_btn = QPushButton("↻ Actualizar")
        refresh_btn.clicked.connect(self._refresh_optimized_map)
        hdr.addWidget(refresh_btn)
        lay.addLayout(hdr)

        desc = QLabel(
            "Muestra las posiciones originales del CSV y las corregidas por cada modelo. "
            "Genera el mapa corregido y descarga el CSV optimizado."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #4a5568; font-size: 13px;")
        lay.addWidget(desc)

        # Controles: CSV + modelo + botón generar
        ctrl_grp = QGroupBox("Generar mapa corregido")
        cg = QHBoxLayout(ctrl_grp)

        cg.addWidget(QLabel("CSV:"))
        self._opt_csv_label = QLabel("(usar CSV activo del pipeline)")
        self._opt_csv_label.setStyleSheet("color: #f39c12; font-size: 13px;")
        cg.addWidget(self._opt_csv_label, stretch=1)

        browse_btn = QPushButton("Examinar...")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._opt_browse_csv)
        cg.addWidget(browse_btn)

        cg.addWidget(QLabel("  Modelo:"))
        self._opt_model_combo = QComboBox()
        self._opt_model_combo.addItems(["both", "bayesian", "neural"])
        self._opt_model_combo.setFixedWidth(100)
        cg.addWidget(self._opt_model_combo)

        gen_btn = QPushButton("Generar")
        gen_btn.setObjectName("btn_green")
        gen_btn.setFixedWidth(100)
        gen_btn.clicked.connect(self._run_generate_map)
        cg.addWidget(gen_btn)
        lay.addWidget(ctrl_grp)

        # Splitter: mapa polar + panel derecho
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Mapa polar
        map_frame = QFrame()
        map_frame.setStyleSheet(
            "QFrame { background: #000; border: 0.5px solid rgba(100,120,200,0.2); border-radius: 8px; }"
        )
        mfl = QVBoxLayout(map_frame)
        mfl.setContentsMargins(8, 8, 8, 8)

        map_hdr = QHBoxLayout()
        map_hdr.addWidget(QLabel("Campo visual (grados)"))
        map_hdr.addStretch()
        self._opt_scope_spin = QSpinBox()
        self._opt_scope_spin.setRange(2, 60)
        self._opt_scope_spin.setValue(15)
        self._opt_scope_spin.setSuffix(" deg")
        self._opt_scope_spin.setFixedWidth(70)
        self._opt_scope_spin.valueChanged.connect(self._on_opt_scope_changed)
        map_hdr.addWidget(QLabel("Scope:"))
        map_hdr.addWidget(self._opt_scope_spin)
        mfl.addLayout(map_hdr)

        self._polar_map_widget = PolarMapWidget(max_ecc=15.0)
        mfl.addWidget(self._polar_map_widget, stretch=1)
        splitter.addWidget(map_frame)

        # Panel derecho: resumen + descarga
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        # Resumen
        sum_grp = QGroupBox("Resumen de corrección")
        sg = QVBoxLayout(sum_grp)
        self._opt_summary_text = QTextEdit()
        self._opt_summary_text.setReadOnly(True)
        self._opt_summary_text.setMinimumHeight(180)
        sg.addWidget(self._opt_summary_text)
        rl.addWidget(sum_grp)

        # Leyenda
        leg_grp = QGroupBox("Leyenda")
        lg = QVBoxLayout(leg_grp)
        for color, label in [
            ("#9ca3af", "Original (CSV PhosLab)"),
            ("#00b4ff", "Corregido — Bayesiano"),
            ("#00ffa0", "Corregido — Neural"),
        ]:
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 18px;")
            dot.setFixedWidth(20)
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #c9d1e0; font-size: 14px;")
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            lg.addLayout(row)
        rl.addWidget(leg_grp)

        # Descarga
        dl_grp = QGroupBox("Descargar CSV corregido")
        dg = QVBoxLayout(dl_grp)
        for label, fname in [
            ("Bayesiano", "corrected_map_bayesian.csv"),
            ("Neural", "corrected_map_neural.csv"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label}:"))
            btn = QPushButton(f"Guardar {label.lower()}")
            btn.setFixedWidth(170)
            btn.clicked.connect(lambda _, f=fname: self._download_corrected_csv(f))
            row.addWidget(btn)
            row.addStretch()
            dg.addLayout(row)
        rl.addWidget(dl_grp)
        rl.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([560, 340])
        lay.addWidget(splitter, stretch=1)

        # Cargar si ya hay datos
        self._opt_csv_path = None
        self._refresh_optimized_map()
        return w

    # ──────────────────────────────────────────────────────────────────────
    # NAVEGACIÓN
    # ──────────────────────────────────────────────────────────────────────

    def _show_page(self, idx: int):
        self._stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_btns):
            btn.setChecked(i == idx)

    # ──────────────────────────────────────────────────────────────────────
    # ACCIONES PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def _launch_phoslab(self):
        entry = PHOSLAB_DIR / "src" / "implant_explorer.py"
        if not entry.exists():
            self._log_msg(f"No se encontró phosLab en {entry}", "warn")
            return
        self._phoslab_proc = subprocess.Popen(
            ["uv", "run", "python", str(entry)],
            cwd=str(PHOSLAB_DIR),
        )
        self._set_status("Phoslab", True)
        self._log_msg("PhosLab lanzado")
        self._mark_step(0, True)

    def _select_csv_manual(self):
        # Directorio inicial: config/ del simulador (donde ya hay CSVs copiados)
        start_dir = str(CSV_DEST_DIR) if CSV_DEST_DIR.exists() else str(PHOSLAB_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Campos Receptivos (Receptive Fields)",
            start_dir,
            "CSV Files (*.csv)",
        )
        if path:
            self._handle_csv(path)

    def _handle_csv(self, src_path: str):
        src = Path(src_path)
        dst = CSV_DEST_DIR / src.name
        try:
            same_file = src.resolve() == dst.resolve()
        except OSError:
            same_file = src == dst
        if not same_file:
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                self._log_msg(f"Error copiando CSV: {e}", "error")
                return

        self._current_csv = str(dst)
        try:
            self._implant_data = read_implant_ids_from_csv(self._current_csv)
        except Exception as e:
            self._log_msg(f"Error leyendo CSV: {e}", "error")
            return

        # Actualizar UI
        self._csv_label.setText(f"CSV: {src.name}")
        self._params_csv_label.setText(src.name)
        self._params_csv_label.setStyleSheet("color: #00ffa0; font-size: 13px;")

        # Rellenar lista de implant IDs
        self._implant_list.clear()
        for iid, indices in self._implant_data.items():
            item = QListWidgetItem(
                f"{iid}  ({len(indices)} electrodos: {indices[:4]}{'…' if len(indices)>4 else ''})"
            )
            item.setData(Qt.ItemDataRole.UserRole, iid)
            self._implant_list.addItem(item)
        self._implant_list.selectAll()

        # Reconstruir bloques de electrodos por implant en la página de parámetros
        self._rebuild_implant_electrode_blocks()
        # Actualizar panel de electrodos disponibles en modo Standard
        self._rebuild_std_electrode_info()

        # Actualizar params.yaml con ruta del CSV
        self._update_params_csv(str(dst))
        self._log_msg(f"CSV cargado: {src.name} — {len(self._implant_data)} implant(s)")
        self._mark_step(1, True)
        self._update_step_states()
        self._watcher_icon.setStyleSheet("color: #00ffa0; font-size: 16px;")

    def _launch_simulator(self):
        if not self._current_csv:
            self._log_msg("Selecciona un CSV primero", "warn")
            return
        if self._sim_proc and self._sim_proc.poll() is None:
            self._log_msg("El simulador ya está en ejecución", "warn")
            return

        self._sim_proc = subprocess.Popen(
            ["uv", "run", "python", "main.py"],
            cwd=str(SIMULADOR_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._set_status("simulador", True)
        self._log_msg("Simulador lanzado — experimento en curso…")
        self._mark_step(2, False)  # Pendiente hasta que termine

        # Monitor en hilo separado
        self._sim_monitor = ProcessMonitor(self._sim_proc)
        self._sim_monitor.output_line.connect(self._on_sim_output)
        self._sim_monitor.process_finished.connect(self._on_sim_finished)
        self._sim_monitor.start()

    def _on_sim_output(self, line: str):
        """Captura output del simulador y detecta errores de electrodo."""
        if "no encontrado en CSV" in line.lower() or "not found in csv" in line.lower():
            self._log_msg(f"WARN: {line}", "warn")
        elif "error" in line.lower():
            self._log_msg(line, "error")

    def _on_sim_finished(self, returncode: int):
        if returncode == 0:
            self._log_msg("OK: Experimento completado correctamente", "ok")
            self._mark_step(2, True)
            self._mark_step(3, True)
            self._refresh_analysis()
        else:
            self._log_msg(
                f"ERROR: El simulador terminó con error (código {returncode})",
                "error",
            )
        self._set_status("simulador", False)

    def _run_learning(self):
        model = self._learn_combo.currentText()
        scope = self._learn_scope_combo.currentText()
        input_mode = self._learn_input_mode_combo.currentText()
        test_mode = self._test_mode_combo.currentText()
        test_source = self._test_source_combo.currentText()
        test_experiment = ""
        if test_mode == "select":
            data = self._test_exp_combo.currentData()
            if isinstance(data, dict):
                test_source = data.get("source", test_source)
                test_experiment = data.get("name", "")
            else:
                test_experiment = self._test_exp_combo.currentText().strip()
        cmd = [
            "uv",
            "run",
            "python",
            "run_learning.py",
            "--model",
            model,
            "--scope",
            scope,
            "--input-mode",
            input_mode,
            "--test-mode",
            test_mode,
            "--test-source",
            test_source,
        ]
        if test_mode == "select" and test_experiment:
            cmd.extend(["--test-experiment", test_experiment])
        proc = subprocess.Popen(
            cmd,
            cwd=str(SIMULADOR_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._set_status("learning", True)
        msg = (
            f"Aprendizaje iniciado — modelo: {model}, scope: {scope}, test: {test_mode}"
        )
        if test_mode == "select" and test_experiment:
            msg += f" ({test_source}:{test_experiment})"
        self._log_msg(msg)

        monitor = ProcessMonitor(proc)
        monitor.output_line.connect(self._on_learning_output)
        monitor.process_finished.connect(self._on_learning_finished)
        monitor.start()
        self._learning_monitor = monitor

    def _on_learning_output(self, line: str):
        lower = line.lower()
        if "traceback" in lower or "error" in lower or "exception" in lower:
            self._log_msg(f"[learn] {line}", "error")
        else:
            self._log_msg(f"[learn] {line}")

    def _on_learning_finished(self, returncode: int):
        if returncode == 0:
            self._log_msg("OK: Aprendizaje completado", "ok")
            self._mark_step(4, True)
            self._refresh_learning()
            self._refresh_optimized_map()
        else:
            self._log_msg(
                f"ERROR: Aprendizaje terminó con error (código {returncode})",
                "error",
            )
        self._set_status("learning", False)

    def _reset_pipeline(self):
        reply = QMessageBox.question(
            self,
            "Nuevo experimento",
            "¿Iniciar un nuevo experimento completo? Se reiniciarán los pasos del pipeline.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._current_csv = None
            self._implant_data = {}
            self._csv_label.setText("Sin CSV seleccionado")
            for i in range(5):
                self._mark_step(i, False)
            self._step_states = [False] * 5
            self._log_msg("Pipeline reiniciado — nuevo experimento")
            self._launch_phoslab()

    # ──────────────────────────────────────────────────────────────────────
    # CSV WATCHER
    # ──────────────────────────────────────────────────────────────────────

    def _start_csv_watcher(self):
        watch_dir = PHOSLAB_DIR / "data" / "exported_RFs"
        if not watch_dir.exists():
            watch_dir = PHOSLAB_DIR / "src"
        if not watch_dir.exists():
            watch_dir = PHOSLAB_DIR
        self._csv_watcher = CsvWatcher(watch_dir, CSV_DEST_DIR)
        self._csv_watcher.csv_detected.connect(self._handle_csv)
        self._csv_watcher.watch_error.connect(lambda msg: self._log_msg(msg, "error"))
        self._csv_watcher.start()
        self._set_status("watcher", True)
        self._watcher_path.setText(
            f"Vigilando {watch_dir.name}/ → percept_mapper/config/"
        )
        self._log_msg("Vigilante CSV activo")

    # ──────────────────────────────────────────────────────────────────────
    # PARÁMETROS YAML
    # ──────────────────────────────────────────────────────────────────────

    def _load_params(self):
        if not PARAMS_YAML.exists():
            return
        self._params = load_yaml_safe(PARAMS_YAML)

        # Input mode
        im = self._params.get("input_mode", "mouse")
        idx = self._input_mode_combo.findText(im)
        if idx >= 0:
            self._input_mode_combo.setCurrentIndex(idx)

        # Default current
        dc = self._params.get("stimulation", {}).get("default_current_uA", 90)
        self._default_current_spin.setValue(int(dc))

        # Timing
        timing = self._params.get("timing", {})
        for key, spin in self._timing_spins.items():
            spin.setValue(int(timing.get(key, spin.minimum())))

        # Modo experimento
        mode = self._params.get("experiment_mode", "mapping")
        idx = self._mode_combo.findText(mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._on_mode_changed(mode)

        # Mapping params
        pm = self._params.get("phosphene_mapping", {})
        self._rep_spin.setValue(int(pm.get("num_repetitions", 5)))

        stim = self._params.get("stimulation", {})
        self._map_stim_spins["pulse_width"].setValue(int(stim.get("pulse_width", 200)))
        self._map_stim_spins["frequency"].setValue(int(stim.get("frequency", 50)))
        map_currents = stim.get("stimulation_currents_uA")
        if isinstance(map_currents, list):
            self._map_currents_edit.setText(",".join(str(c) for c in map_currents))
        elif isinstance(map_currents, dict):
            self._map_currents_edit.setText(json.dumps(map_currents, ensure_ascii=True))
        else:
            self._map_currents_edit.setText("")

        # Standard params
        esel = self._params.get("retinotopic_mapping", {}).get(
            "electrode_selection", {}
        )
        esel_mode = esel.get("mode", "range") if isinstance(esel, dict) else "all"
        idx = self._esel_combo.findText(esel_mode)
        if idx >= 0:
            self._esel_combo.setCurrentIndex(idx)
        if isinstance(esel, dict):
            self._esel_start_spin.setValue(int(esel.get("start", 0)))
            self._esel_end_spin.setValue(int(esel.get("end", 10)))
            self._esel_step_spin.setValue(int(esel.get("step", 1)))
            if esel_mode == "manual":
                indices = esel.get("indices", [])
                if isinstance(indices, list):
                    self._esel_manual_edit.setText(",".join(str(x) for x in indices))
                elif indices is not None:
                    self._esel_manual_edit.setText(str(indices))

        self._std_stim_spins["pulse_width"].setValue(int(stim.get("pulse_width", 200)))
        self._std_stim_spins["frequency"].setValue(int(stim.get("frequency", 50)))

        currents_std = stim.get("stimulation_currents_standard_uA", [150, 30, 90, 40])
        if isinstance(currents_std, list):
            self._std_currents_edit.setText(",".join(str(c) for c in currents_std))
        elif isinstance(currents_std, dict):
            self._std_currents_edit.setText(json.dumps(currents_std, ensure_ascii=True))
        else:
            self._std_currents_edit.setText("")

        # CSV actual en params
        csv_path = self._params.get("retinotopic_mapping", {}).get(
            "coords_csv_path", ""
        )
        if csv_path and Path(csv_path).exists():
            self._current_csv = csv_path
            self._implant_data = read_implant_ids_from_csv(csv_path)
            self._csv_label.setText(f"CSV: {Path(csv_path).name}")
            self._params_csv_label.setText(Path(csv_path).name)
            self._params_csv_label.setStyleSheet("color: #00ffa0; font-size: 13px;")
            self._implant_list.clear()
            for iid, indices in self._implant_data.items():
                item = QListWidgetItem(
                    f"{iid}  ({len(indices)} electrodos: {indices[:4]}{'…' if len(indices)>4 else ''})"
                )
                item.setData(Qt.ItemDataRole.UserRole, iid)
                self._implant_list.addItem(item)
            self._implant_list.selectAll()

            self._rebuild_implant_electrode_blocks()
            self._rebuild_std_electrode_info()
            saved_by_implant = self._params.get("phosphene_mapping", {}).get(
                "electrodes_by_implant", []
            )
            for block in saved_by_implant:
                iid = str(block.get("implant_id", ""))
                indices = block.get("electrode_index", [])
                if iid in self._implant_electrode_edits and indices:
                    self._implant_electrode_edits[iid].setText(
                        ",".join(str(x) for x in indices)
                    )

    def _save_params(self):
        if not PARAMS_YAML.exists():
            self._log_msg("params.yaml no encontrado", "error")
            return

        updates = {}

        updates["input_mode"] = self._input_mode_combo.currentText()

        updates.setdefault("stimulation", {})
        updates["stimulation"][
            "default_current_uA"
        ] = self._default_current_spin.value()

        updates["timing"] = {k: s.value() for k, s in self._timing_spins.items()}

        mode = self._mode_combo.currentText()
        updates["experiment_mode"] = mode

        if mode == "mapping":
            electrodes_by_implant = []
            for iid, edit in self._implant_electrode_edits.items():
                text = edit.text().strip()
                if not text:
                    continue
                try:
                    indices = [int(x.strip()) for x in text.split(",") if x.strip()]
                    if indices:
                        electrodes_by_implant.append(
                            {
                                "implant_id": iid,
                                "electrode_index": indices,
                            }
                        )
                except ValueError:
                    self._log_msg(
                        f"WARN: Formato de índices inválido para implant {iid}",
                        "warn",
                    )
            if electrodes_by_implant:
                updates.setdefault("phosphene_mapping", {})
                updates["phosphene_mapping"][
                    "electrodes_by_implant"
                ] = electrodes_by_implant

        if mode == "mapping":
            updates.setdefault("phosphene_mapping", {})
            updates["phosphene_mapping"]["num_repetitions"] = self._rep_spin.value()
            updates["stimulation"]["pulse_width"] = self._map_stim_spins[
                "pulse_width"
            ].value()
            updates["stimulation"]["frequency"] = self._map_stim_spins[
                "frequency"
            ].value()
            map_currents_text = self._map_currents_edit.text().strip()
            if map_currents_text:
                try:
                    parsed = self._parse_currents_text(map_currents_text)
                    if isinstance(parsed, (list, dict)):
                        updates["stimulation"]["stimulation_currents_uA"] = parsed
                    elif isinstance(parsed, (int, float)):
                        updates["stimulation"]["stimulation_currents_uA"] = [
                            float(parsed)
                        ]
                    else:
                        self._log_msg(
                            "WARN: Corrientes mapping invalidas (usa lista o dict)",
                            "warn",
                        )
                except Exception:
                    self._log_msg("WARN: No se pudo parsear corrientes mapping", "warn")

        else:  # standard
            # Electrodos por implante (igual que mapping)
            electrodes_by_implant = []
            for iid, edit in self._std_implant_electrode_edits.items():
                text = edit.text().strip()
                if not text:
                    continue
                try:
                    indices = [int(x.strip()) for x in text.split(",") if x.strip()]
                    if indices:
                        electrodes_by_implant.append(
                            {"implant_id": iid, "electrode_index": indices}
                        )
                except ValueError:
                    self._log_msg(
                        f"WARN: Formato de índices inválido para implant {iid} (standard)",
                        "warn",
                    )
            if electrodes_by_implant:
                updates.setdefault("retinotopic_mapping", {})
                updates["retinotopic_mapping"][
                    "electrodes_by_implant"
                ] = electrodes_by_implant

            esel_mode = self._esel_combo.currentText()
            esel = {"mode": esel_mode}
            if esel_mode == "range":
                esel["start"] = self._esel_start_spin.value()
                esel["end"] = self._esel_end_spin.value()
                esel["step"] = self._esel_step_spin.value()
            elif esel_mode == "manual":
                try:
                    indices = [
                        int(x.strip())
                        for x in self._esel_manual_edit.text().split(",")
                        if x.strip()
                    ]
                    esel["indices"] = indices
                except ValueError:
                    pass
            updates.setdefault("retinotopic_mapping", {})
            updates["retinotopic_mapping"]["electrode_selection"] = esel

            updates["stimulation"]["pulse_width"] = self._std_stim_spins[
                "pulse_width"
            ].value()
            updates["stimulation"]["frequency"] = self._std_stim_spins[
                "frequency"
            ].value()

            std_currents_text = self._std_currents_edit.text().strip()
            if std_currents_text:
                try:
                    parsed = self._parse_currents_text(std_currents_text)
                    if isinstance(parsed, (list, dict)):
                        updates["stimulation"][
                            "stimulation_currents_standard_uA"
                        ] = parsed
                    elif isinstance(parsed, (int, float)):
                        updates["stimulation"]["stimulation_currents_standard_uA"] = [
                            float(parsed)
                        ]
                    else:
                        self._log_msg(
                            "WARN: Corrientes standard invalidas (usa lista o dict)",
                            "warn",
                        )
                except Exception:
                    self._log_msg(
                        "WARN: No se pudo parsear corrientes standard", "warn"
                    )

        try:
            save_yaml_partial(PARAMS_YAML, updates)
            self._log_msg("OK: params.yaml actualizado", "ok")
        except Exception as e:
            self._log_msg(f"Error guardando params: {e}", "error")

    def _update_params_csv(self, csv_path: str):
        updates = {
            "retinotopic_mapping": {
                "coordinate_source": "phoslab_csv",
                "coords_csv_path": csv_path.replace("\\", "/"),
            }
        }
        try:
            save_yaml_partial(PARAMS_YAML, updates)
            self._log_msg("params.yaml actualizado con ruta del CSV")
        except Exception as e:
            self._log_msg(f"Error actualizando CSV path: {e}", "error")

    def _parse_currents_text(self, text: str):
        cleaned = text.strip()
        if not cleaned:
            return None
        if "{" in cleaned or "[" in cleaned or "\n" in cleaned:
            return yaml.safe_load(cleaned)
        if ":" in cleaned:
            pairs = {}
            for part in cleaned.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError("formato de par invalido")
                k, v = part.split(":", 1)
                pairs[int(k.strip())] = float(v.strip())
            return pairs
        if "," in cleaned:
            return [float(x.strip()) for x in cleaned.split(",") if x.strip()]
        return float(cleaned)

    def _rebuild_implant_electrode_blocks(self):
        """Reconstruye los bloques de entrada de electrodos por implant_id."""
        # Limpiar layout anterior (excepto el stretch final)
        while self._implant_electrodes_layout.count() > 1:
            item = self._implant_electrodes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._implant_electrode_edits.clear()

        if not self._implant_data:
            no_csv = QLabel("Carga un CSV para ver los implants disponibles")
            no_csv.setStyleSheet("color: #4a5568; font-size: 13px; padding: 8px;")
            self._implant_electrodes_layout.insertWidget(0, no_csv)
            return

        for i, (iid, available_indices) in enumerate(self._implant_data.items()):
            block = QFrame()
            block.setStyleSheet(
                "QFrame { background: rgba(99,102,241,0.06); "
                "border: 0.5px solid rgba(100,120,200,0.2); border-radius: 6px; }"
            )
            bl = QVBoxLayout(block)
            bl.setContentsMargins(10, 8, 10, 8)
            bl.setSpacing(4)

            hdr = QHBoxLayout()
            id_lbl = QLabel(f"Implant: {iid}")
            id_lbl.setStyleSheet("color: #818cf8; font-size: 14px; font-weight: 500;")
            hdr.addWidget(id_lbl)
            hdr.addStretch()
            all_btn = QPushButton("Todos")
            all_btn.setFixedWidth(60)
            all_btn.setStyleSheet(
                "QPushButton { background: rgba(99,102,241,0.1); border: 0.5px solid "
                "rgba(99,102,241,0.3); border-radius: 4px; color: #818cf8; "
                "padding: 2px 6px; font-size: 13px; }"
            )
            hdr.addWidget(all_btn)
            bl.addLayout(hdr)

            input_row = QHBoxLayout()
            edit = QLineEdit()
            edit.setPlaceholderText(
                f"ej: {','.join(str(x) for x in available_indices[:5])}{',...' if len(available_indices)>5 else ''}"
            )
            edit.setStyleSheet(
                "background:#0f1623; border:0.5px solid rgba(100,120,200,0.3); "
                "border-radius:4px; color:#c9d1e0; padding:3px 6px; font-size:14px;"
            )
            count_lbl = QLabel("0 electrodos")
            count_lbl.setFixedWidth(100)
            count_lbl.setStyleSheet("color: #4a5568; font-size: 13px;")

            def _on_text_changed(text, lbl=count_lbl):
                try:
                    idxs = [int(x.strip()) for x in text.split(",") if x.strip()]
                    lbl.setText(f"{len(idxs)} electrodo{'s' if len(idxs)!=1 else ''}")
                    lbl.setStyleSheet("color: #27ae60; font-size: 13px;")
                except ValueError:
                    lbl.setText("formato inválido")
                    lbl.setStyleSheet("color: #e74c3c; font-size: 13px;")

            edit.textChanged.connect(_on_text_changed)
            all_btn.clicked.connect(
                lambda _, e=edit, idxs=available_indices: e.setText(
                    ",".join(str(x) for x in idxs)
                )
            )

            input_row.addWidget(edit)
            input_row.addWidget(count_lbl)
            bl.addLayout(input_row)

            hint = QLabel(
                f"Disponibles: {', '.join(str(x) for x in available_indices)}"
            )
            hint.setStyleSheet("color: #374151; font-size: 12px;")
            hint.setWordWrap(True)
            bl.addWidget(hint)

            self._implant_electrodes_layout.insertWidget(i, block)
            self._implant_electrode_edits[iid] = edit

    def _on_input_mode_changed(self, mode: str):
        self._calibrate_btn.setVisible(mode == "gaze")

    def _run_gaze_calibration(self):
        """Lanza la calibración de 5 puntos del eye tracker en un proceso separado."""
        calib_script = SIMULADOR_DIR / "calibrate_gaze.py"
        if not calib_script.exists():
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Calibración",
                f"No se encontró el script de calibración:\n{calib_script}",
            )
            return
        subprocess.Popen(
            ["uv", "run", "python", str(calib_script)],
            cwd=str(SIMULADOR_DIR),
        )

    def _on_mode_changed(self, mode: str):
        self._mapping_grp.setVisible(mode == "mapping")
        self._standard_grp.setVisible(mode == "standard")

    def _rebuild_std_electrode_info(self):
        """Reconstruye el panel de electrodos por implant en modo Standard."""
        while self._std_electrodes_layout.count() > 1:
            item = self._std_electrodes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._std_implant_electrode_edits = {}

        if not self._implant_data:
            no_csv = QLabel("Carga un CSV para ver los electrodos disponibles")
            no_csv.setStyleSheet("color: #4a5568; font-size: 13px; padding: 8px;")
            self._std_electrodes_layout.insertWidget(0, no_csv)
            return

        for i, (iid, available_indices) in enumerate(self._implant_data.items()):
            block = QFrame()
            block.setStyleSheet(
                "QFrame { background: rgba(99,102,241,0.06); "
                "border: 0.5px solid rgba(100,120,200,0.2); border-radius: 6px; }"
            )
            bl = QVBoxLayout(block)
            bl.setContentsMargins(10, 8, 10, 8)
            bl.setSpacing(4)

            hdr = QHBoxLayout()
            id_lbl = QLabel(f"Implant: {iid}")
            id_lbl.setStyleSheet("color: #818cf8; font-size: 14px; font-weight: 500;")
            hdr.addWidget(id_lbl)
            hdr.addStretch()
            all_btn = QPushButton("Todos")
            all_btn.setFixedWidth(60)
            all_btn.setStyleSheet(
                "QPushButton { background: rgba(99,102,241,0.1); border: 0.5px solid "
                "rgba(99,102,241,0.3); border-radius: 4px; color: #818cf8; "
                "padding: 2px 6px; font-size: 13px; }"
            )
            hdr.addWidget(all_btn)
            bl.addLayout(hdr)

            input_row = QHBoxLayout()
            edit = QLineEdit()
            edit.setPlaceholderText(
                f"ej: {','.join(str(x) for x in available_indices[:5])}{',...' if len(available_indices)>5 else ''}"
            )
            edit.setStyleSheet(
                "background:#0f1623; border:0.5px solid rgba(100,120,200,0.3); "
                "border-radius:4px; color:#c9d1e0; padding:3px 6px; font-size:14px;"
            )
            count_lbl = QLabel("0 electrodos")
            count_lbl.setFixedWidth(100)
            count_lbl.setStyleSheet("color: #4a5568; font-size: 13px;")

            def _on_text_changed(text, lbl=count_lbl):
                try:
                    idxs = [int(x.strip()) for x in text.split(",") if x.strip()]
                    lbl.setText(f"{len(idxs)} electrodo{'s' if len(idxs)!=1 else ''}")
                    lbl.setStyleSheet("color: #27ae60; font-size: 13px;")
                except ValueError:
                    lbl.setText("formato inválido")
                    lbl.setStyleSheet("color: #e74c3c; font-size: 13px;")

            edit.textChanged.connect(_on_text_changed)

            all_indices_str = ",".join(str(x) for x in available_indices)
            all_btn.clicked.connect(lambda _, e=edit, s=all_indices_str: e.setText(s))

            input_row.addWidget(edit)
            input_row.addWidget(count_lbl)
            bl.addLayout(input_row)

            hint = QLabel(
                f"Disponibles: {', '.join(str(x) for x in available_indices)}"
            )
            hint.setStyleSheet("color: #6b7280; font-size: 12px;")
            hint.setWordWrap(True)
            bl.addWidget(hint)

            self._std_implant_electrode_edits[iid] = edit
            self._std_electrodes_layout.insertWidget(i, block)

    def _on_esel_changed(self, mode: str):
        self._esel_range_widget.setVisible(mode == "range")
        self._esel_manual_widget.setVisible(mode == "manual")

    # ──────────────────────────────────────────────────────────────────────
    # ANÁLISIS
    # ──────────────────────────────────────────────────────────────────────

    def _refresh_analysis(self):
        self._exp_tree.clear()

        root_mapping = QTreeWidgetItem(["mapping_experiments"])
        root_mapping.setData(0, Qt.ItemDataRole.UserRole, str(MAPPING_DIR))
        self._exp_tree.addTopLevelItem(root_mapping)
        if MAPPING_DIR.exists():
            self._populate_tree(root_mapping, MAPPING_DIR)

        root_logs = QTreeWidgetItem(["logs"])
        root_logs.setData(0, Qt.ItemDataRole.UserRole, str(LOGS_DIR))
        self._exp_tree.addTopLevelItem(root_logs)
        if LOGS_DIR.exists():
            self._populate_tree(root_logs, LOGS_DIR)

        self._exp_tree.expandToDepth(1)
        self._refresh_learning_experiments()

    def _on_exp_selected(self, current, previous):
        if not current:
            return
        path_str = current.data(0, Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        path = Path(path_str)
        lines = [f"📁 {path.name}\n"]

        for json_file in sorted(path.rglob("*.json")):
            lines.append(f"── {json_file.relative_to(path)}")
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if not isinstance(v, (list, dict)):
                        lines.append(f"   {k}: {v}")
                    elif isinstance(v, list) and len(v) < 8:
                        lines.append(f"   {k}: {v}")
            except Exception:
                lines.append("   (no se pudo leer)")
            lines.append("")

        lines.append("── Archivos:")
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix in (".png", ".csv", ".json", ".txt"):
                lines.append(f"   {f.relative_to(path)}")

        self._exp_detail.setPlainText("\n".join(lines))

    def _delete_selected_analysis(self):
        item = self._exp_tree.currentItem()
        if not item:
            QMessageBox.information(
                self, "Eliminar", "Selecciona un experimento para eliminar."
            )
            return

        path_str = item.data(0, Qt.ItemDataRole.UserRole)
        if not path_str:
            QMessageBox.information(
                self, "Eliminar", "Selecciona un experimento para eliminar."
            )
            return

        path = Path(path_str)
        if not path.exists():
            QMessageBox.warning(self, "Eliminar", "El elemento ya no existe.")
            self._exp_detail.clear()
            self._refresh_analysis()
            return

        path_res = path.resolve()
        allowed = False
        for root in (MAPPING_DIR, LOGS_DIR):
            root_res = root.resolve()
            if path_res == root_res:
                QMessageBox.information(
                    self,
                    "Eliminar",
                    "No se puede eliminar la raiz de analisis.",
                )
                return
            if root_res in path_res.parents:
                allowed = True
                break

        if not allowed:
            QMessageBox.warning(
                self, "Eliminar", "La ruta seleccionada no esta en analisis."
            )
            return

        reply = QMessageBox.question(
            self,
            "Eliminar",
            f"Se eliminara '{path.name}' y todo su contenido. Continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception as exc:
            QMessageBox.critical(self, "Eliminar", f"No se pudo eliminar: {exc}")
            return

        self._exp_detail.clear()
        self._refresh_analysis()

    def _populate_tree(self, parent_item, root_path: Path):
        for child in sorted(root_path.iterdir()):
            if not child.is_dir():
                continue
            item = QTreeWidgetItem([child.name])
            item.setData(0, Qt.ItemDataRole.UserRole, str(child))
            parent_item.addChild(item)
            self._populate_tree(item, child)

    # ──────────────────────────────────────────────────────────────────────
    # APRENDIZAJE
    # ──────────────────────────────────────────────────────────────────────

    def _refresh_learning(self):
        summary_file = LEARNING_DIR / "dataset_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._dataset_text.setPlainText(
                    json.dumps(data, indent=2, ensure_ascii=False)
                )
            except Exception:
                self._dataset_text.setPlainText("Error leyendo resumen del dataset")
        else:
            self._dataset_text.setPlainText("Sin resumen del dataset")

        metrics_file = LEARNING_DIR / "evaluation_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._metrics_text.setPlainText(
                    json.dumps(data, indent=2, ensure_ascii=False)
                )
            except Exception:
                self._metrics_text.setPlainText("Error leyendo métricas")
        else:
            self._metrics_text.setPlainText("Sin resultados todavía")

        for attr, fname in [
            ("_img_error", "error_comparison.png"),
            ("_img_neural", "neural_training.png"),
            ("_img_visual", "visual_field_comparison.png"),
        ]:
            btn = getattr(self, attr)
            img_path = LEARNING_DIR / fname
            if img_path.exists():
                btn.setIcon(QIcon(str(img_path)))
                btn.setIconSize(IMAGE_ICON_SIZES.get(fname, QSize(420, 280)))
                btn.setText("")
                btn.setProperty("image_path", str(img_path))
            else:
                btn.setIcon(QIcon())
                btn.setText(f"Sin imagen: {fname}")
                btn.setProperty("image_path", None)

    def _export_learning_zip(self):
        if not LEARNING_DIR.exists():
            QMessageBox.warning(
                self,
                "Exportar",
                "No existe la carpeta learning_results.",
            )
            return

        results_dir = SIMULADOR_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        default_name = (
            f"learning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
        default_path = str(results_dir / default_name)
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar learning_results (.zip)",
            default_path,
            "ZIP Files (*.zip)",
        )
        if not out_path:
            return

        out = Path(out_path)
        base_name = out.with_suffix("")
        try:
            shutil.make_archive(
                str(base_name),
                "zip",
                root_dir=str(LEARNING_DIR),
            )
            self._log_msg(f"OK: ZIP guardado: {base_name.name}.zip", "ok")
        except Exception as exc:
            QMessageBox.critical(self, "Exportar", f"No se pudo crear el ZIP: {exc}")

    def _refresh_learning_experiments(self):
        if not hasattr(self, "_test_exp_combo"):
            return
        self._test_exp_combo.clear()

        items = []
        for source, root in [("mapping", MAPPING_DIR), ("standard", LOGS_DIR)]:
            if not root.exists():
                continue
            for d in sorted(root.iterdir(), reverse=True):
                if not d.is_dir():
                    continue
                items.append((source, d.name))

        if not items:
            self._test_exp_combo.addItem("(sin experimentos)", None)
        else:
            for source, name in items:
                label = f"{source}: {name}"
                self._test_exp_combo.addItem(label, {"source": source, "name": name})

        self._on_test_mode_changed(self._test_mode_combo.currentText())

    def _on_test_mode_changed(self, mode: str):
        is_select = mode == "select"
        is_last = mode == "last"
        self._test_exp_combo.setEnabled(is_select)
        self._test_source_combo.setEnabled(is_last)

    def _open_image_dialog(self, img_path: str | None):
        if not img_path or not Path(img_path).exists():
            return

        pix = QPixmap(str(img_path))
        if pix.isNull():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(Path(img_path).name)
        dialog.resize(1000, 720)

        layout = QVBoxLayout(dialog)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))
        zoom_slider = QSlider(Qt.Orientation.Horizontal)
        zoom_slider.setRange(25, 200)
        zoom_slider.setValue(100)
        zoom_value = QLabel("100%")
        zoom_row.addWidget(zoom_slider)
        zoom_row.addWidget(zoom_value)
        layout.addLayout(zoom_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(img_label)
        layout.addWidget(scroll)

        def _apply_zoom(value: int):
            zoom_value.setText(f"{value}%")
            scale = value / 100.0
            scaled = pix.scaled(
                int(pix.width() * scale),
                int(pix.height() * scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            img_label.setPixmap(scaled)

        zoom_slider.valueChanged.connect(_apply_zoom)
        _apply_zoom(100)

        dialog.exec()

    # ──────────────────────────────────────────────────────────────────────
    # MAPA OPTIMIZADO — ACCIONES
    # ──────────────────────────────────────────────────────────────────────

    def _opt_browse_csv(self):
        start = str(CSV_DEST_DIR) if CSV_DEST_DIR.exists() else str(SIMULADOR_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar CSV de Implant Explorer", start, "CSV Files (*.csv)"
        )
        if path:
            self._opt_csv_path = path
            self._opt_csv_label.setText(Path(path).name)
            self._opt_csv_label.setStyleSheet("color: #27ae60; font-size: 13px;")

    def _run_generate_map(self):
        csv_path = self._opt_csv_path or self._current_csv
        if not csv_path or not Path(csv_path).exists():
            self._log_msg("Selecciona un CSV antes de generar el mapa", "warn")
            return

        model = self._opt_model_combo.currentText()
        cmd = [
            "uv",
            "run",
            "python",
            str(SIMULADOR_DIR / "generate_corrected_map.py"),
            "--csv",
            csv_path,
            "--model",
            model,
        ]
        self._log_msg(f"Generando mapa corregido ({model})…")
        proc = subprocess.Popen(
            cmd,
            cwd=str(SIMULADOR_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        monitor = ProcessMonitor(proc)
        monitor.process_finished.connect(self._on_map_generated)
        monitor.start()
        self._map_monitor = monitor

    def _on_map_generated(self, returncode: int):
        if returncode == 0:
            self._log_msg("OK: Mapa corregido generado", "ok")
            self._refresh_optimized_map()
            self._show_page(4)
        else:
            self._log_msg("ERROR: Error generando mapa corregido", "error")

    def _refresh_optimized_map(self):
        """Carga los CSVs corregidos y actualiza el mapa polar."""
        series = []

        csv_path = self._opt_csv_path or self._current_csv
        if csv_path and Path(csv_path).exists():
            xs, ys = self._load_xy_from_csv(csv_path, "x_deg", "y_deg")
            if xs:
                series.append((xs, ys, "#9ca3af", "Original"))

        bayes_csv = LEARNING_DIR / "corrected_map_bayesian.csv"
        if bayes_csv.exists():
            xs, ys = self._load_xy_from_csv(
                str(bayes_csv), "x_deg_corrected", "y_deg_corrected"
            )
            if xs:
                series.append((xs, ys, "#00b4ff", "Bayesiano"))

        neural_csv = LEARNING_DIR / "corrected_map_neural.csv"
        if neural_csv.exists():
            xs, ys = self._load_xy_from_csv(
                str(neural_csv), "x_deg_corrected", "y_deg_corrected"
            )
            if xs:
                series.append((xs, ys, "#00ffa0", "Neural"))

        self._polar_map_widget.set_series(series)

        if CORRECTED_MAP_SUMMARY.exists():
            try:
                with open(CORRECTED_MAP_SUMMARY, encoding="utf-8") as f:
                    data = json.load(f)
                lines = [
                    f"CSV: {data.get('source_csv', '-')}",
                    f"Electrodos: {data.get('n_electrodes', '-')}",
                    f"Implants: {', '.join(data.get('implant_ids', []))}",
                    "",
                ]
                orig = data.get("original", {})
                lines.append(
                    f"Original  x_mean={orig.get('x_mean', 0):.3f}  "
                    f"y_mean={orig.get('y_mean', 0):.3f}  "
                    f"ecc_mean={orig.get('ecc_mean', 0):.3f}"
                )
                if "bayesian" in data:
                    b = data["bayesian"]
                    lines.append(
                        f"Bayesiano  shift_mean={b.get('mean_shift_deg', 0):.3f}  "
                        f"shift_max={b.get('max_shift_deg', 0):.3f}"
                    )
                if "neural" in data:
                    n = data["neural"]
                    lines.append(
                        f"Neural  shift_mean={n.get('mean_shift_deg', 0):.3f}  "
                        f"shift_max={n.get('max_shift_deg', 0):.3f}"
                    )
                self._opt_summary_text.setPlainText("\n".join(lines))
            except Exception:
                self._opt_summary_text.setPlainText("Error leyendo resumen")
        else:
            if not series:
                self._opt_summary_text.setPlainText(
                    "Sin datos todavía.\nGenera el mapa corregido primero."
                )
            else:
                self._opt_summary_text.setPlainText(
                    "Mapa cargado. Sin resumen JSON disponible."
                )

    def _load_xy_from_csv(self, path: str, x_col: str, y_col: str) -> tuple[list, list]:
        xs, ys = [], []
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if x_col in row and y_col in row:
                        try:
                            xs.append(float(row[x_col]))
                            ys.append(float(row[y_col]))
                        except ValueError:
                            pass
        except Exception:
            pass
        return xs, ys

    def _on_opt_scope_changed(self, value: int):
        self._polar_map_widget.max_ecc = float(value)
        self._polar_map_widget.update()

    def _download_corrected_csv(self, fname: str):
        src = LEARNING_DIR / fname
        if not src.exists():
            QMessageBox.warning(
                self,
                "No disponible",
                f"El archivo {fname} no existe todavía.\nGenera el mapa corregido primero.",
            )
            return
        dst, _ = QFileDialog.getSaveFileName(
            self, "Guardar CSV corregido", fname, "CSV Files (*.csv)"
        )
        if dst:
            shutil.copy2(str(src), dst)
            self._log_msg(f"OK: CSV guardado: {Path(dst).name}", "ok")

    # ──────────────────────────────────────────────────────────────────────
    # ESTADO DE PASOS
    # ──────────────────────────────────────────────────────────────────────

    def _mark_step(self, idx: int, done: bool):
        if idx >= len(self._step_widgets):
            return
        card = self._step_widgets[idx]
        has_logo = getattr(card, "_logo_label", None) is not None
        if done:
            card.setStyleSheet("""
                QFrame#card {
                    background: rgba(0,255,160,0.04);
                    border: 0.5px solid rgba(0,255,160,0.35);
                    border-radius: 10px;
                }
            """)
            card._num_label.setStyleSheet("""
                border: 2px solid #00ffa0; border-radius: 90px;
                color: #00ffa0; font-size: 96px;
                background: rgba(0,255,160,0.12);
            """)
            card._num_label.setText("✓")
            if has_logo:
                card._logo_label.hide()
                card._num_label.show()
        else:
            card.setStyleSheet(
                "QFrame#card { background: #0f0f0f; border: 0.5px solid rgba(0,200,255,0.15); border-radius: 10px; }"
            )
            card._num_label.setStyleSheet(
                "border: 2px solid #1a1a2e; border-radius: 90px; color: #2d3748; font-size: 72px;"
            )
            card._num_label.setText(str(idx + 1))
            if has_logo:
                card._num_label.hide()
                card._logo_label.show()
        self._step_states[idx] = done

    def _update_step_states(self):
        csv_ok = self._current_csv is not None
        self._mark_step(1, csv_ok)

    # ──────────────────────────────────────────────────────────────────────
    # STATUS SIDEBAR
    # ──────────────────────────────────────────────────────────────────────

    def _set_status(self, key: str, active: bool):
        if key not in self._status_labels:
            return
        dot, _ = self._status_labels[key]
        dot.setStyleSheet(
            f"color: {'#00ffa0' if active else '#1a1a1a'}; font-size: 12px;"
        )

    # ──────────────────────────────────────────────────────────────────────
    # LOG
    # ──────────────────────────────────────────────────────────────────────

    def _log_msg(self, msg: str, level: str = "info"):
        colors = {
            "info": "#818cf8",
            "warn": "#f39c12",
            "error": "#e74c3c",
            "ok": "#27ae60",
        }
        color = colors.get(level, "#818cf8")
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(
            f'<span style="color:#374151">[{ts}]</span> '
            f'<span style="color:{color}">{msg}</span>'
        )
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    # ──────────────────────────────────────────────────────────────────────
    # CIERRE
    # ──────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._csv_watcher:
            self._csv_watcher.stop()
            self._csv_watcher.wait(2000)
        event.accept()
