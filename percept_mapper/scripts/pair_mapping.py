"""Método de mapeo PAREADO (paired) — dos fosfenos por ensayo.

En cada ensayo se estimulan DOS electrodos en orden: A primero, luego un
descanso breve (rest, ~1 s sin marcador), luego B. El participante traza una
línea dirigida desde donde percibió A hasta donde percibió B. El dato
load-bearing no es la línea sino sus dos extremos ordenados; su diferencia es
el vector de desplazamiento Δ(A→B) que alimenta a `embed_displacement_lsq`
(ver scripts/relative_map.py).

Esta clase es deliberadamente paralela a PhospheneMappingExperiment pero con
una forma de ensayo distinta (dos estimulaciones + una respuesta de línea), por
lo que NO encaja en el trial-loop por-electrodo de absolute/relative/forced.
main.py la conduce con un bucle de pares propio cuando
`mapping_method == "paired"`. Reutiliza las mismas funciones de fase
(run_prestim/stim/poststim) que el resto del experimento.

Salida en disco: un único `pairs/metadata.json` con un registro por par
(no una carpeta por electrodo), pensado para que
`scripts/analysis/build_relative_map.py` (Stage 4) reconstruya el mapa.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pygame

from scripts.schemas import TrialRecord


class PairMappingExperiment:
    """Gestiona un experimento de mapeo pareado: itera sobre pares de
    electrodos, estimula A→rest→B y captura una respuesta de línea por par.

    Mantiene un único directorio `pairs/` con `metadata.json` (lista de
    `trials`, un registro por par) más PNGs de respuesta `pair_XXX.png`.
    """

    def __init__(
        self,
        params,
        screen,
        clock,
        eye_tracker,
        anchor_screen,
        drawing_tablet,
        webcam_viewer,
        gaze_trace,
        timing_config,
        display_info=None,
        experiment_name="paired",
        experiment_dir=None,
        apriltag_overlay=None,
        debug_overlay=None,
        input_mode="mouse",
        coords_csv="",
        rest_ms=1000.0,
    ):
        self.params = params
        self.screen = screen
        self.clock = clock
        self.eye_tracker = eye_tracker
        self.anchor_screen = anchor_screen
        self.drawing_tablet = drawing_tablet
        self.webcam_viewer = webcam_viewer
        self.gaze_trace = gaze_trace
        self.display_info = display_info
        self.apriltag_overlay = apriltag_overlay
        self.debug_overlay = debug_overlay
        self.timing_config = timing_config
        self.experiment_name = experiment_name
        self.input_mode = input_mode
        self.coords_csv = coords_csv
        self.rest_ms = float(rest_ms)
        self.mapping_method = "paired"

        experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_dir is not None:
            self.experiment_dir = Path(experiment_dir)
        else:
            self.experiment_dir = (
                Path("mapping_experiments")
                / f"paired_{experiment_name}_{experiment_timestamp}"
            )
        # Un único subdirectorio para todos los pares (no carpeta por electrodo).
        self.pairs_dir = self.experiment_dir / "pairs"
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_metadata = {
            "experiment_name": experiment_name,
            "experiment_id": experiment_timestamp,
            "start_time": datetime.now().isoformat(),
            "display": display_info,
            "input_mode": input_mode,
            "mapping_method": "paired",
            "coords_csv": coords_csv,
            "rest_ms": self.rest_ms,
            "timing": timing_config,
            "trials": [],  # un registro por par
        }
        print(
            f"[PairMappingExperiment] Inicializado  carpeta={self.pairs_dir}  "
            f"rest={self.rest_ms:.0f}ms"
        )

    # ------------------------------------------------------------------ #
    # Bucle por par                                                       #
    # ------------------------------------------------------------------ #

    def run_pair(
        self,
        pair_index,
        electrode_a,
        electrode_b,
        stim_a,
        stim_b,
        pos_a,
        pos_b,
        current_a,
        current_b,
        pulse_width_us,
        frequency_hz,
        run_prestim_func,
        run_stim_func,
        run_poststim_func,
        drawing_tablet_reset_func,
        FPS,
        *,
        electrode_info_a=None,
        electrode_info_b=None,
        is_practice=False,
    ):
        """Ejecuta UN par: prestim→stim(A)→poststim, rest, prestim→stim(B)→
        poststim, y luego la respuesta de línea. Devuelve el registro del par
        o None si el usuario canceló.

        Cada estimulación reusa el ciclo de fijación con reintento idéntico al
        de PhospheneMappingExperiment: una pérdida de fijación reintenta esa
        estimulación desde su prestim sin invalidar la otra mitad del par."""
        tag = " [PRACTICE]" if is_practice else ""
        print("\n" + "=" * 70)
        print(f"PAR {pair_index}{tag}: electrodo A={electrode_a} → B={electrode_b}")
        print("=" * 70)

        record = {
            "pair_index": pair_index,
            "electrode_a": electrode_a,
            "electrode_b": electrode_b,
            "electrode_info_a": electrode_info_a,
            "electrode_info_b": electrode_info_b,
            "position_a": list(pos_a) if pos_a is not None else None,
            "position_b": list(pos_b) if pos_b is not None else None,
            "is_practice": bool(is_practice),
            "rest_ms": self.rest_ms,
            "start_time": datetime.now().isoformat(),
            "events": {},
            "fixation_losses": 0,
            "gaze_tracking": {"prestim": [], "stim": [], "poststim": [], "drawing": []},
        }

        # --- Estimulación A ------------------------------------------------
        if not self._stimulate_one(
            "A", stim_a, record, run_prestim_func, run_stim_func, run_poststim_func
        ):
            return None

        # --- Descanso (rest) entre A y B: solo el anchor, sin marcador ------
        record["events"]["rest_start"] = datetime.now().isoformat()
        if not self._run_rest(self.rest_ms):
            return None
        record["events"]["rest_end"] = datetime.now().isoformat()

        # --- Estimulación B ------------------------------------------------
        if not self._stimulate_one(
            "B", stim_b, record, run_prestim_func, run_stim_func, run_poststim_func
        ):
            return None

        # --- Respuesta de línea (un único trazo dirigido A→B) --------------
        print("[RESP] Traza la línea del 1er punto al 2º")
        record["events"]["drawing_start"] = datetime.now().isoformat()
        drawing_tablet_reset_func(self.drawing_tablet)

        completed = False
        draw_start = time.time()
        while not completed:
            events = pygame.event.get()
            finished = self.drawing_tablet.update(self.screen, events)
            # Cruz de fijación central durante la respuesta (como en relative):
            # la posición se reporta relativa a la fijación, no a la pantalla.
            self._draw_center_cross()

            if (
                self.eye_tracker
                and hasattr(self.eye_tracker, "last_raw_gaze")
                and self.eye_tracker.last_raw_gaze
            ):
                elapsed_ms = (time.time() - draw_start) * 1000
                gx, gy = self.eye_tracker.last_raw_gaze
                record["gaze_tracking"]["drawing"].append(
                    {"time_ms": int(elapsed_ms), "gaze_x": int(gx), "gaze_y": int(gy)}
                )

            if finished:
                completed = True

            for event in events:
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

            self._display_flip()
            self.clock.tick(FPS)

        record["events"]["drawing_end"] = datetime.now().isoformat()

        # --- Guardado ------------------------------------------------------
        if is_practice:
            print(f"      [PRACTICE] Par {pair_index} ejecutado sin guardado")
            return record

        response_result = self.drawing_tablet.save_result(
            self.pairs_dir,
            drawing_filename=f"pair_{pair_index:03d}.png",
            saccade_filename=None,
        )
        meta = response_result.to_metadata()
        record.update(meta)
        record["response_status"] = response_result.status
        record["end_time"] = datetime.now().isoformat()

        self.experiment_metadata["trials"].append(record)
        self._save_metadata()
        print(
            f"      ✓ Par {pair_index} guardado (status={response_result.status}, "
            f"A_px={meta.get('endpoint_a_px')}, B_px={meta.get('endpoint_b_px')})"
        )
        return record

    # ------------------------------------------------------------------ #
    # Helpers de fase                                                     #
    # ------------------------------------------------------------------ #

    def _stimulate_one(
        self, label, stim_screen, record, run_prestim_func, run_stim_func, run_poststim_func
    ):
        """Ejecuta prestim→stim→poststim para un fosfeno, con reintento ante
        pérdida de fijación. Devuelve True si completó, False si el usuario
        canceló (ESC/QUIT)."""
        print(f"[STIM {label}] electrodo {getattr(stim_screen, 'active_electrode_index', '?')}")
        record["events"][f"stim_{label}_start"] = datetime.now().isoformat()

        phase_completed = False
        attempts = 0
        while not phase_completed:
            attempts += 1
            if attempts > 1:
                print(f"      [RETRY {label}] Intento #{attempts}")

            if self.gaze_trace:
                self.gaze_trace.clear()
            ok = run_prestim_func(
                self.screen, self.clock, self.anchor_screen, self.eye_tracker,
                record, self.webcam_viewer, self.gaze_trace,
            )
            if not ok:
                return False

            ok = run_stim_func(
                self.screen, self.clock, stim_screen, self.eye_tracker,
                record, self.webcam_viewer, self.gaze_trace,
            )
            if ok is None:
                print(f"      [RETRY {label}] Volviendo a prestim...")
                continue
            if not ok:
                return False

            ok = run_poststim_func(
                self.screen, self.clock, self.anchor_screen, self.eye_tracker,
                record, self.webcam_viewer, self.gaze_trace,
            )
            if ok is None:
                print(f"      [RETRY {label}] Volviendo a prestim...")
                continue
            if not ok:
                return False

            phase_completed = True

        record["events"][f"stim_{label}_end"] = datetime.now().isoformat()
        record[f"stim_{label}_attempts"] = attempts
        return True

    def _run_rest(self, duration_ms):
        """Descanso entre A y B: solo el anchor visible (igual filosofía que
        _run_interstimulation_mapping — sin contador que el participante pueda
        anticipar). Devuelve False si el usuario cancela."""
        background_color = tuple(self.params["screen"]["background_color"]) \
            if self.params and "screen" in self.params else (0, 0, 0)
        start = time.time()
        while (time.time() - start) * 1000 < duration_ms:
            if self.webcam_viewer is not None:
                self.webcam_viewer.update()
            self.screen.fill(background_color)
            if self.anchor_screen is not None and hasattr(self.anchor_screen, "draw"):
                self.anchor_screen.draw(self.screen)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
            self._display_flip()
            self.clock.tick(60)
        return True

    def _draw_center_cross(self):
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2
        arm = 24
        pygame.draw.line(self.screen, (255, 255, 255), (cx - arm, cy), (cx + arm, cy), 2)
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy - arm), (cx, cy + arm), 2)

    def _display_flip(self):
        if self.debug_overlay is not None:
            self.debug_overlay.draw(self.screen)
        if self.apriltag_overlay is not None:
            self.apriltag_overlay.draw(self.screen)
        pygame.display.flip()

    # ------------------------------------------------------------------ #
    # Persistencia                                                        #
    # ------------------------------------------------------------------ #

    def _save_metadata(self):
        """Persiste pairs/metadata.json. Cada registro de par pasa por
        TrialRecord.from_dict().to_dict() para heredar schema_version y mantener
        los campos pareados (endpoint_*_px, pair_index, ...) en `extras`."""
        metadata_file = self.pairs_dir / "metadata.json"
        validated = dict(self.experiment_metadata)
        validated["trials"] = [
            TrialRecord.from_dict(t).to_dict() if isinstance(t, dict) else t
            for t in self.experiment_metadata.get("trials", [])
        ]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)

    def finalize(self):
        self.experiment_metadata["end_time"] = datetime.now().isoformat()
        self._save_metadata()
        n = len(self.experiment_metadata["trials"])
        print("\n" + "=" * 70)
        print(f"EXPERIMENTO PAREADO COMPLETADO — {n} pares guardados")
        print(f"📁 {self.pairs_dir}")
        print("=" * 70)
