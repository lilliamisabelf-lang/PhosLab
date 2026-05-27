"""
Pantalla de estimulación - Círculo rojo + punto brillante
El punto SOLO aparece si se completó prestimulation correctamente
"""

import pygame
import numpy as np


class StimulationScreen:
    """
    Muestra círculo rojo de anclaje + punto brillante (fosfeno simulado)
    El punto brillante SOLO se muestra si show_phosphene=True

    La apariencia del fosfeno varía según parámetros de estimulación:
    - current_uA: Corriente en microamperios (afecta tamaño y brillo)
    - pulse_width_us: Ancho de pulso en microsegundos (afecta brillo)
    - frequency_hz: Frecuencia en Hz (afecta brillo por integración temporal)
    """

    def __init__(
        self,
        params,
        eye_tracker,
        phosphene_position=(768, 432),
        current_uA=50.0,
        pulse_width_us=170.0,
        frequency_hz=50.0,
    ):
        self.dynaphos_mapper = None
        self.active_electrode_index = None
        """
        Inicializa la pantalla de estimulación

        Args:
            params: Diccionario con configuración
            eye_tracker: Objeto EyeTracker
            phosphene_position: (x, y) posición ABSOLUTA en píxeles del fosfeno (desde Dynaphos)
            current_uA: Corriente de estimulación en microamperios (default: 90 µA)
            pulse_width_us: Ancho de pulso en microsegundos (default: 170 µs)
            frequency_hz: Frecuencia de estimulación en Hz (default: 50 Hz)
        """
        print("[StimulationScreen] Inicializando...")

        self.params = params
        self.eye_tracker = eye_tracker

        # Configuración de pantalla
        screen_config = params["screen"]
        self.width = screen_config["width"]
        self.height = screen_config["height"]
        self.background_color = tuple(screen_config["background_color"])

        # Configuración del círculo de anclaje
        anchor_config = screen_config["anchor_circle"]
        self.circle_radius = anchor_config["radius"]
        self.circle_color = tuple(anchor_config["color"])
        self.circle_thickness = anchor_config["thickness"]
        self.tolerance_radius = anchor_config["tolerance_radius"]

        # Centro de la pantalla (donde está el círculo rojo)
        self.circle_center = (self.width // 2, self.height // 2)

        # Posición del punto brillante (fosfeno)
        # phosphene_position YA VIENE como posición ABSOLUTA desde Dynaphos
        self.phosphene_position = phosphene_position

        # Calcular offset relativo al centro (para debug/análisis)
        self.phosphene_offset = (
            phosphene_position[0] - self.circle_center[0],
            phosphene_position[1] - self.circle_center[1],
        )

        # ============================================
        # PARÁMETROS DE ESTIMULACIÓN
        # ============================================
        self.current_uA = current_uA
        self.pulse_width_us = pulse_width_us
        self.frequency_hz = frequency_hz

        # ============================================
        # CALCULAR CARACTERÍSTICAS DEL FOSFENO
        # Basado en modelo de Dynaphos (Bosking et al., 2017; Fernández et al., 2021)
        # ============================================

        # Parámetros del modelo (desde dynaphos params.yaml)
        self.I_half = 20.0  # µA - corriente para la mitad del tamaño máximo
        self.MD_mm = 0.7  # mm - diámetro máximo en córtex
        self.slope_size = 0.08  # mm/µA
        self.radius_to_sigma = 0.5  # Factor para convertir radio a sigma gaussiana

        # Parámetros de brillo
        self.rheobase = 23.9  # µA - corriente mínima efectiva
        self.cps_half = 1.058e-07 * 1e6  # Convertido a µA*µs
        self.slope_brightness = 19152642.5

        # Parámetros visuales (conversión cortical a píxeles)
        self.pixels_per_mm_cortex = 60.0  # Aproximación para magnificación cortical

        # Calcular tamaño basado en corriente (ecuación sqrt de Dynaphos)
        self.phosphene_size = self._calculate_phosphene_size(current_uA)

        # Calcular brillo basado en corriente, pulse width y frecuencia
        self.phosphene_brightness = self._calculate_phosphene_brightness(
            current_uA, pulse_width_us, frequency_hz
        )

        # Color del fosfeno (blanco con brillo variable)
        self.phosphene_color = (
            self.phosphene_brightness,
            self.phosphene_brightness,
            self.phosphene_brightness,
        )

        # Estado
        self.is_looking = False
        self.show_phosphene = False  # Controla si se muestra el punto
        # Catch trial: cuando True, set_show_phosphene(True) es un no-op.
        # Esto garantiza que la pantalla de stim corre con la misma duración
        # y feedback visual del anchor que un trial real, pero sin fosfeno.
        self.catch_mode = False

        print(
            f"[StimulationScreen] ✓ Inicializado - Fosfeno en posición absoluta {phosphene_position}"
        )
        print(f"                     Offset desde centro: {self.phosphene_offset}")
        print(
            f"                     Parámetros: {current_uA:.1f}µA, {pulse_width_us:.1f}µs, {frequency_hz:.1f}Hz"
        )
        print(f"                     Tamaño calculado: {self.phosphene_size:.1f} px")
        print(f"                     Brillo calculado: {self.phosphene_brightness}/255")

    def _calculate_phosphene_size(self, current_uA):
        """
        Calcula el tamaño del fosfeno basado en la corriente
        Usa ecuación sqrt de Dynaphos (Bosking et al., 2017)

        Args:
            current_uA: Corriente en microamperios

        Returns:
            float: Radio del fosfeno en píxeles
        """
        # Ecuación sqrt: diameter = MD * sqrt(I / I_half)
        # donde MD es el diámetro máximo y I_half es la corriente para la mitad del tamaño

        # Aplicar threshold mínimo
        if current_uA < self.rheobase:
            return 0.0  # Bajo umbral, no hay fosfeno visible

        # Calcular diámetro en mm (espacio cortical)
        diameter_mm = self.MD_mm * np.sqrt(current_uA / self.I_half)

        # Convertir a radio en píxeles (espacio visual)
        # Usar conversión aproximada de magnificación cortical
        radius_px = (diameter_mm / 2.0) * self.pixels_per_mm_cortex

        # Limitar entre valores razonables (5-50 píxeles)
        radius_px = np.clip(radius_px, 5.0, 50.0)

        return radius_px

    def _calculate_phosphene_brightness(self, current_uA, pulse_width_us, frequency_hz):
        """
        Calcula el brillo del fosfeno basado en corriente, pulse width y frecuencia
        Usa modelo de saturación sigmoidal de Dynaphos (Fernández et al., 2021)

        Args:
            current_uA: Corriente en microamperios
            pulse_width_us: Ancho de pulso en microsegundos
            frequency_hz: Frecuencia en Hz

        Returns:
            int: Valor de brillo (0-255)
        """
        # Calcular charge per phase (carga por fase)
        # CPS = current * pulse_width (en unidades consistentes)
        charge_per_phase = current_uA * pulse_width_us  # µA * µs = µC (aproximación)

        # Aplicar threshold mínimo
        if current_uA < self.rheobase:
            return 0

        # Función sigmoidea de saturación
        # brightness = 1 / (1 + exp(-slope * (charge - threshold)))
        activation = 1.0 / (
            1.0
            + np.exp(
                -self.slope_brightness * (charge_per_phase / 1e6 - self.cps_half / 1e6)
            )
        )

        # Ajuste por frecuencia (integración temporal)
        # Mayor frecuencia → mayor brillo percibido (hasta un límite)
        freq_factor = np.clip(
            frequency_hz / 100.0, 0.5, 2.0
        )  # Normalizar alrededor de 100 Hz

        # Calcular brillo final (0-255)
        brightness = int(activation * freq_factor * 255)
        brightness = np.clip(brightness, 30, 255)  # Mínimo visible = 30

        return brightness

    def set_show_phosphene(self, show):
        """
        Activa o desactiva la visualización del punto brillante.
        Si catch_mode está activo, fuerza show=False (catch trial: misma
        pantalla pero sin fosfeno).
        """
        if show and self.catch_mode:
            self.show_phosphene = False
            print("[StimulationScreen] ◇ Catch trial: fosfeno suprimido")
            return
        self.show_phosphene = show
        if show:
            print("[StimulationScreen] ⚪ Punto brillante ACTIVADO")
        else:
            print("[StimulationScreen] ○ Punto brillante DESACTIVADO")

    def update(self, screen, eye_tracker_frame):
        """
        Actualiza y dibuja la pantalla de estimulación

        Args:
            screen: Superficie de Pygame donde dibujar
            eye_tracker_frame: Frame de la webcam para eye tracking

        Returns:
            bool: True si está mirando al círculo de anclaje, False si no
        """
        # Detectar si está mirando al círculo de anclaje (no al punto brillante)
        if self.eye_tracker and eye_tracker_frame is not None:
            self.is_looking = self.eye_tracker.is_looking_at_point(
                frame=eye_tracker_frame,
                target_point=self.circle_center,  # Verificar que mira al centro (círculo rojo)
                screen_size=(self.width, self.height),
                tolerance_radius=self.tolerance_radius,
            )
        else:
            # Si no hay eye tracker, asumir que está mirando (para testing)
            self.is_looking = True

        # Dibujar
        self.draw(screen)

        return self.is_looking

    def draw(self, screen):
        """Dibuja la pantalla con círculo de anclaje + punto brillante (si está activado).

        Traffic-light state encoding for the anchor (siempre solo contorno —
        nunca rellenado — para no competir visualmente con el fosfeno):
          - red ring   : stimulation active ("hold still")
          - green ring : fixation acquired, stim about to fire
          - white ring : idle / not looking
        """
        screen.fill(self.background_color)

        if self.show_phosphene:
            anchor_color = (220, 30, 30)
        elif self.is_looking:
            anchor_color = (0, 220, 0)
        else:
            anchor_color = (200, 200, 200)

        pygame.draw.circle(
            screen,
            anchor_color,
            self.circle_center,
            self.circle_radius,
            self.circle_thickness,
        )

        # Punto brillante (sólo si la fase de stim lo ha activado).
        if self.show_phosphene:
            self._draw_gaussian_phosphene(screen)

    def _draw_gaussian_phosphene(self, screen):
        """Renderiza el fosfeno como una gaussiana 2D centrada en
        phosphene_position. Vectorizado para evitar el bucle pixel-a-pixel."""
        cx, cy = self.phosphene_position
        radius = int(self.phosphene_size)
        sigma = radius * self.radius_to_sigma
        size = max(1, int(radius * 3))  # ≈3σ cubre 99.7%

        ys, xs = np.mgrid[-size:size, -size:size]
        distance = np.sqrt(xs * xs + ys * ys)
        intensity = np.exp(-0.5 * (distance / max(sigma, 1e-6)) ** 2)
        brightness = (intensity * self.phosphene_brightness).clip(0, 255).astype(np.uint8)

        rgba = np.stack([brightness, brightness, brightness, brightness], axis=-1)
        surface = pygame.image.frombuffer(rgba.tobytes(), (size * 2, size * 2), "RGBA")
        screen.blit(surface, (cx - size, cy - size))
