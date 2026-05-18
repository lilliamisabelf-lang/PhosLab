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
        self.timing_config = timing_config
        self.electrode_index = electrode_index
        self.electrode_info = electrode_info
        self.num_repetitions = num_repetitions
        self.experiment_name = experiment_name

        # Crear carpeta del experimento de mapeo
        experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        print("\n" + "=" * 70)
        print(f"REPETICIÓN {repetition_number}/{self.num_repetitions}")
        print(f"Electrodo: {self.electrode_index}")
        print(f"Posición: {phosphene_position}")
        print(f"Corriente: {current_uA} µA")
        print("=" * 70)

        # Metadata de esta repetición
        repetition_metadata = {
            "repetition_number": repetition_number,
            "electrode_index": self.electrode_index,
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

        # Resetear tablet para nuevo dibujo
        drawing_tablet_reset_func(self.drawing_tablet)

        drawing_completed = False
        canvas = None
        drawing_start_time = time.time()

        while not drawing_completed:
            events = pygame.event.get()

            finished, canvas = self.drawing_tablet.update(self.screen, events)

            # Registrar gaze coordinates durante drawing
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
                print(f"      ✓ Repetición {repetition_number} completada")
                drawing_completed = True

            # Comprobar ESC/QUIT
            for event in events:
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

            pygame.display.flip()
            self.clock.tick(FPS)

        repetition_metadata["drawing_end"] = datetime.now().isoformat()

        # ============================================
        # GUARDADO
        # ============================================
        print(f"      [GUARDANDO] Repetición {repetition_number}...")

        # Guardar dibujo individual
        drawing_filename = (
            self.electrode_dir / f"repetition_{repetition_number:03d}.png"
        )
        pygame.image.save(canvas, str(drawing_filename))
        print(f"        ✓ Dibujo: {drawing_filename.name}")

        repetition_metadata["end_time"] = datetime.now().isoformat()
        repetition_metadata["drawing_file"] = drawing_filename.name

        # Añadir a la lista de repeticiones
        self.experiment_metadata["repetitions"].append(repetition_metadata)

        # Guardar metadata intermedia (por si el experimento se interrumpe)
        self._save_metadata()

        print(f"        ✓ Metadata guardada")

        # ============================================
        # INTERSTIMULATION (solo si NO es la última repetición)
        # ============================================
        if repetition_number < self.num_repetitions:
            print()
            print(
                f"      [BREAK] Descanso antes de la repetición {repetition_number + 1}..."
            )

            repetition_metadata["interstim_start"] = datetime.now().isoformat()

            # Usar run_interstimulation pero con mensaje personalizado
            success = self._run_interstimulation_mapping(
                repetition_number, self.num_repetitions
            )
            if not success:
                return None  # Usuario canceló

            repetition_metadata["interstim_end"] = datetime.now().isoformat()
        else:
            print()
            print("      [FIN] Última repetición completada - No hay break")

        print()
        return repetition_metadata

    def _run_interstimulation_mapping(self, current_rep, total_reps):
        """
        Pantalla de descanso entre repeticiones
        Similar a run_interstimulation pero adaptado para mapeo
        """
        background_color = tuple(self.params["screen"]["background_color"])
        font = pygame.font.Font(None, 72)
        font_small = pygame.font.Font(None, 48)

        INTERSTIMULATION_MS = self.timing_config["interstimulation_ms"]
        start_time = time.time()

        while True:
            elapsed_ms = (time.time() - start_time) * 1000

            if elapsed_ms >= INTERSTIMULATION_MS:
                return True

            # Actualizar webcam viewer
            if self.webcam_viewer is not None:
                if not self.webcam_viewer.update():
                    print("      ⚠ Ventana de webcam cerrada")

            self.screen.fill(background_color)

            # Texto principal
            text = f"Intervalo entre estimulaciones - Repetición {current_rep}/{total_reps}"
            text_surface = font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(
                center=(
                    self.screen.get_width() // 2,
                    self.screen.get_height() // 2 - 50,
                )
            )
            self.screen.blit(text_surface, text_rect)

            # Tiempo restante
            remaining_s = (INTERSTIMULATION_MS - elapsed_ms) / 1000
            time_text = f"{remaining_s:.1f}s"
            time_surface = font_small.render(time_text, True, (200, 200, 200))
            time_rect = time_surface.get_rect(
                center=(
                    self.screen.get_width() // 2,
                    self.screen.get_height() // 2 + 50,
                )
            )
            self.screen.blit(time_surface, time_rect)

            # Comprobar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_SPACE:
                        # Permitir saltar el descanso con ESPACIO
                        return True

            pygame.display.flip()
            self.clock.tick(60)

    def _save_metadata(self):
        """Guarda el metadata del experimento (llamada intermedia y final)"""
        metadata_file = self.electrode_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.experiment_metadata, f, indent=2, ensure_ascii=False)

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
                f.write(f"  Archivo de dibujo: {rep['drawing_file']}\n\n")

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
