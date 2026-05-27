"""
Módulo de mapeo de fosfenos - Experimentos de repetición N veces para un electrodo

Este módulo permite estimular UN electrodo específico N veces consecutivas
para obtener múltiples representaciones del mismo fosfeno y calcular su
posición promedio.
"""

import pygame
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from scripts.response_capture import apply_response_metadata, write_response_summary
from scripts.schemas import TrialRecord


class PhospheneMappingExperiment:
    """
    Gestiona experimentos de mapeo de fosfenos con N repeticiones por electrodo
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
        electrode_index,
        electrode_info=None,
        display_info=None,
        num_repetitions=5,
        experiment_name="default",
        experiment_dir=None,
        apriltag_overlay=None,
    ):
        """
        Inicializa un experimento de mapeo de fosfenos

        Args:
            params: Diccionario con configuración general
            screen: Superficie de pygame
            clock: Reloj de pygame
            eye_tracker: Instancia de EyeTracker o MouseTracker
            anchor_screen: Instancia de AnchorScreen
            drawing_tablet: Instancia de DrawingTablet
            webcam_viewer: Instancia de WebcamViewer (o None)
            gaze_trace: Instancia de GazeTrace (o None)
            timing_config: Diccionario con tiempos (prestim, stim, poststim, etc)
            electrode_index: Índice del electrodo a mapear
            num_repetitions: Número de repeticiones N por electrodo
            experiment_name: Nombre descriptivo del experimento
            experiment_dir: Si se proporciona, usa esta carpeta en lugar de crear una nueva
                            (usado en modo multi-electrodo para evitar carpetas duplicadas)
        """
        print(
            f"\n[PhospheneMappingExperiment] Inicializando mapeo de electrodo {electrode_index}"
        )
        print(f"                             Repeticiones: {num_repetitions}")

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
        self.timing_config = timing_config
        self.electrode_index = electrode_index
        self.electrode_info = electrode_info
        self.num_repetitions = num_repetitions
        self.experiment_name = experiment_name

        # Crear carpeta del experimento de mapeo
        experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_dir is not None:
            # Modo multi-electrodo: usar carpeta externa ya creada, no crear carpeta propia
            self.experiment_dir = Path(experiment_dir)
        else:
            # Modo electrodo único: crear carpeta propia
            self.experiment_dir = (
                Path("mapping_experiments")
                / f"mapping_{experiment_name}_{experiment_timestamp}"
            )
        self.electrode_dir = self.experiment_dir / f"electrode_{electrode_index:03d}"
        self.electrode_dir.mkdir(parents=True, exist_ok=True)

        print(f"                             Carpeta: {self.experiment_dir}")

        # Metadata del experimento completo
        self.experiment_metadata = {
            "experiment_name": experiment_name,
            "experiment_id": experiment_timestamp,
            "start_time": datetime.now().isoformat(),
            "electrode_index": electrode_index,
            "electrode_info": electrode_info,
            "display": display_info,
            "num_repetitions": num_repetitions,
            "timing": timing_config,
            "repetitions": [],  # Se llenará con datos de cada repetición
        }

    def run_single_repetition(
        self,
        repetition_number,
        stimulation_screen,
        phosphene_position,
        current_uA,
        pulse_width_us,
        frequency_hz,
        run_prestim_func,
        run_stim_func,
        run_poststim_func,
        run_interstim_func,
        check_quit_func,
        drawing_tablet_reset_func,
        FPS,
        *,
        trial_idx=None,
        is_catch=False,
        is_practice=False,
        run_interstim_after=True,
    ):
        """
        Ejecuta UNA repetición del experimento de mapeo

        Args:
            repetition_number: Número de repetición actual (1 a N)
            stimulation_screen: Instancia de StimulationScreen
            phosphene_position: Posición (x, y) del fosfeno
            current_uA: Corriente de estimulación
            pulse_width_us: Ancho de pulso
            frequency_hz: Frecuencia
            run_prestim_func: Función para ejecutar prestimulación
            run_stim_func: Función para ejecutar estimulación
            run_poststim_func: Función para ejecutar postestimulación
            run_interstim_func: Función para ejecutar interestimulación
            check_quit_func: Función para comprobar si el usuario quiere salir
            drawing_tablet_reset_func: Función para resetear la tablet
            FPS: Frames por segundo

        Returns:
            dict: Metadata de esta repetición, o None si el usuario canceló
        """
        tag = " [CATCH]" if is_catch else (" [PRACTICE]" if is_practice else "")
        print("\n" + "=" * 70)
        print(f"REPETICIÓN {repetition_number}/{self.num_repetitions}{tag}")
        print(f"Electrodo: {self.electrode_index}")
        if trial_idx is not None:
            print(f"trial_idx global: {trial_idx}")
        print(f"Posición: {phosphene_position}")
        print(f"Corriente: {current_uA} µA")
        print("=" * 70)

        # Activar/desactivar modo catch en la pantalla de estim. Esto se restaura
        # al final del método.
        prev_catch_mode = getattr(stimulation_screen, "catch_mode", False)
        if hasattr(stimulation_screen, "catch_mode"):
            stimulation_screen.catch_mode = bool(is_catch)

        # Metadata de esta repetición
        repetition_metadata = {
            "repetition_number": repetition_number,
            "electrode_index": self.electrode_index,
            "trial_idx": trial_idx,
            "is_catch": bool(is_catch),
            "is_practice": bool(is_practice),
            "position": phosphene_position,
            "stimulation_parameters": {
                "current_uA": current_uA,
                "pulse_width_us": pulse_width_us,
                "frequency_hz": frequency_hz,
            },
            "start_time": datetime.now().isoformat(),
            "events": {},
            "fixation_losses": 0,
            "gaze_tracking": {"prestim": [], "stim": [], "poststim": [], "drawing": []},
        }

        # ============================================
        # ESTADOS 1-3: PRESTIM → STIM → POSTSTIM
        # Con reintento automático en caso de pérdida de fijación
        # ============================================
        trial_attempt = 0
        phase_completed = False

        while not phase_completed:
            trial_attempt += 1
            if trial_attempt > 1:
                print(
                    f"\n      [RETRY] Intento #{trial_attempt} para repetición {repetition_number}"
                )

            # ESTADO 1: PRESTIMULATION
            if self.gaze_trace:
                self.gaze_trace.clear()
            success = run_prestim_func(
                self.screen,
                self.clock,
                self.anchor_screen,
                self.eye_tracker,
                repetition_metadata,
                self.webcam_viewer,
                self.gaze_trace,
            )
            if not success:
                return None  # Usuario canceló (ESC/QUIT)

            # ESTADO 2: STIMULATION
            success = run_stim_func(
                self.screen,
                self.clock,
                stimulation_screen,
                self.eye_tracker,
                repetition_metadata,
                self.webcam_viewer,
                self.gaze_trace,
            )
            if success is None:
                # Pérdida de fijación → reintentar desde prestim
                print(f"      [RETRY] Volviendo a prestimulation...")
                continue
            if not success:
                return None  # Usuario canceló

            # ESTADO 3: POSTSTIMULATION
            success = run_poststim_func(
                self.screen,
                self.clock,
                self.anchor_screen,
                self.eye_tracker,
                repetition_metadata,
                self.webcam_viewer,
                self.gaze_trace,
            )
            if success is None:
                # Pérdida de fijación → reintentar desde prestim
                print(f"      [RETRY] Volviendo a prestimulation...")
                continue
            if not success:
                return None  # Usuario canceló

            # Todas las fases completadas exitosamente
            phase_completed = True

        repetition_metadata["trial_attempts"] = trial_attempt

        # ============================================
        # ESTADO 4: DRAWING
        # ============================================
        print(f"[4/4] DRAWING: Dibuja el fosfeno (repetición {repetition_number})")
        repetition_metadata["drawing_start"] = datetime.now().isoformat()

        # Resetear pantalla de respuesta para nuevo trial
        drawing_tablet_reset_func(self.drawing_tablet)

        drawing_completed = False
        drawing_start_time = time.time()

        while not drawing_completed:
            events = pygame.event.get()

            finished = self.drawing_tablet.update(self.screen, events)

            # Registrar gaze coordinates durante drawing (sólo informativo;
            # en modo saccade el payload ya contiene las muestras).
            if (
                self.eye_tracker
                and hasattr(self.eye_tracker, "last_raw_gaze")
                and self.eye_tracker.last_raw_gaze
            ):
                elapsed_ms = (time.time() - drawing_start_time) * 1000
                gaze_x, gaze_y = self.eye_tracker.last_raw_gaze
                repetition_metadata["gaze_tracking"]["drawing"].append(
                    {
                        "time_ms": int(elapsed_ms),
                        "gaze_x": int(gaze_x),
                        "gaze_y": int(gaze_y),
                    }
                )

            if finished:
                print(
                    f"      ✓ Repetición {repetition_number} completada "
                    f"({self.drawing_tablet.mode}, status={self.drawing_tablet.last_status})"
                )
                drawing_completed = True

            # Comprobar ESC/QUIT
            for event in events:
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

            self._display_flip()
            self.clock.tick(FPS)

        repetition_metadata["drawing_end"] = datetime.now().isoformat()

        # ============================================
        # GUARDADO
        # ============================================
        if is_practice:
            # Practice trials run the same flow but do not write artifacts
            # nor join the analyzed repetitions list.
            print(f"      [PRACTICE] Repetición {repetition_number} ejecutada sin guardado")
        else:
            tag = "CATCH" if is_catch else "GUARDANDO"
            print(f"      [{tag}] Repetición {repetition_number}...")

            file_prefix = "catch" if is_catch else "repetition"
            response_result = self.drawing_tablet.save_result(
                self.electrode_dir,
                drawing_filename=f"{file_prefix}_{repetition_number:03d}.png",
                saccade_filename=f"saccade_samples_{file_prefix}_{repetition_number:03d}.json",
            )
            apply_response_metadata(repetition_metadata, response_result)
            print(f"        ✓ Respuesta: {response_result.response_file}")

            repetition_metadata["end_time"] = datetime.now().isoformat()

            # Añadir a la lista de repeticiones
            self.experiment_metadata["repetitions"].append(repetition_metadata)

            # Guardar metadata intermedia (por si el experimento se interrumpe)
            self._save_metadata()

            print(f"        ✓ Metadata guardada")

        # Restaurar modo catch del stim_screen para no contaminar el siguiente trial.
        if hasattr(stimulation_screen, "catch_mode"):
            stimulation_screen.catch_mode = prev_catch_mode

        # ============================================
        # INTERSTIMULATION (modo legacy: solo si run_interstim_after=True y no es la última rep)
        # En el modelo de trial-list, main.py setea run_interstim_after=False
        # y maneja los breaks entre trials externamente.
        # ============================================
        if run_interstim_after and repetition_number < self.num_repetitions:
            print()
            print(f"      [BREAK] Descanso antes de la repetición {repetition_number + 1}...")
            repetition_metadata["interstim_start"] = datetime.now().isoformat()
            success = self._run_interstimulation_mapping(
                repetition_number, self.num_repetitions
            )
            if not success:
                return None
            repetition_metadata["interstim_end"] = datetime.now().isoformat()

        print()
        return repetition_metadata

    def _run_interstimulation_mapping(self, current_rep, total_reps, *, duration_ms=None):
        """Pantalla mínima entre estimulaciones: anchor visible, sin texto ni
        contador. Un contador visible entrena al participante a anticipar el
        próximo estímulo según el reloj — ese es exactamente el confound que
        el cue de audio existe para evitar. Mostramos solo el anchor."""
        background_color = tuple(self.params["screen"]["background_color"])

        interstim_ms = float(self.timing_config["interstimulation_ms"] if duration_ms is None else duration_ms)
        start_time = time.time()

        while True:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= interstim_ms:
                return True

            if self.webcam_viewer is not None:
                if not self.webcam_viewer.update():
                    print("      ⚠ Ventana de webcam cerrada")

            self.screen.fill(background_color)
            if self.anchor_screen is not None and hasattr(self.anchor_screen, "draw"):
                # Anchor neutro (no hay fixación activa todavía)
                self.anchor_screen.draw(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_SPACE:
                        return True  # saltar break con ESPACIO

            self._display_flip()
            self.clock.tick(60)

    def _display_flip(self):
        if self.apriltag_overlay is not None:
            self.apriltag_overlay.draw(self.screen)
        pygame.display.flip()

    def _save_metadata(self):
        """Guarda el metadata del experimento (llamada intermedia y final).

        Cada repetición se valida via `TrialRecord.from_dict(...).to_dict()`
        antes de persistir — el dict resultante lleva schema_version,
        nombres canónicos y mantiene campos desconocidos en `extras`. Esto
        garantiza que el JSON en disco siempre sea cargable por analizadores
        que esperen el schema, sin obligar al trial loop a mutar dataclasses
        in-flight.
        """
        metadata_file = self.electrode_dir / "metadata.json"
        validated = dict(self.experiment_metadata)
        validated["repetitions"] = [
            TrialRecord.from_dict(rep).to_dict() if isinstance(rep, dict) else rep
            for rep in self.experiment_metadata.get("repetitions", [])
        ]
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(validated, f, indent=2, ensure_ascii=False)

    def finalize(self):
        """
        Finaliza el experimento y guarda todos los datos
        """
        print("\n" + "=" * 70)
        print("FINALIZANDO EXPERIMENTO DE MAPEO")
        print("=" * 70)

        self.experiment_metadata["end_time"] = datetime.now().isoformat()

        # Guardar metadata final
        self._save_metadata()
        print(f"✓ Metadata JSON: metadata.json")

        # Guardar también como TXT legible
        txt_filename = self.electrode_dir / "metadata.txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("EXPERIMENTO DE MAPEO DE FOSFENOS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Nombre: {self.experiment_metadata['experiment_name']}\n")
            f.write(f"ID Experimento: {self.experiment_metadata['experiment_id']}\n")
            f.write(f"Inicio: {self.experiment_metadata['start_time']}\n")
            f.write(f"Fin: {self.experiment_metadata['end_time']}\n")
            f.write(f"Electrodo: {self.experiment_metadata['electrode_index']}\n")
            f.write(f"Repeticiones: {self.experiment_metadata['num_repetitions']}\n\n")

            f.write("TIEMPOS:\n")
            f.write(
                f"  - Prestimulation: {self.timing_config['prestimulation_ms']}ms\n"
            )
            f.write(f"  - Stimulation: {self.timing_config['stimulation_ms']}ms\n")
            f.write(
                f"  - Poststimulation: {self.timing_config['poststimulation_ms']}ms\n"
            )
            f.write(
                f"  - Interstimulation: {self.timing_config['interstimulation_ms']}ms\n\n"
            )

            f.write("=" * 70 + "\n")
            f.write("DETALLES DE CADA REPETICIÓN\n")
            f.write("=" * 70 + "\n\n")

            for rep in self.experiment_metadata["repetitions"]:
                f.write(f"REPETICIÓN {rep['repetition_number']}:\n")
                f.write(f"  Posición: {rep['position']}\n")
                f.write(f"  Inicio: {rep['start_time']}\n")
                f.write(f"  Fin: {rep['end_time']}\n")
                f.write(f"  Pérdidas de fijación: {rep['fixation_losses']}\n")
                f.write(f"  Intentos: {rep['trial_attempts']}\n")
                write_response_summary(f, rep)
                f.write("\n")

        print(f"✓ Metadata TXT: {txt_filename.name}")

        print()
        print("=" * 70)
        print("EXPERIMENTO DE MAPEO COMPLETADO")
        print("=" * 70)
        print()
        print(f"📁 Carpeta: {self.electrode_dir}")
        print(f"📷 Dibujos: {self.num_repetitions} archivos PNG")
        print(f"📄 Metadata: JSON + TXT")
        print()
        print("Para analizar los resultados, ejecuta el script mapping_analyzer.py")
        print()
