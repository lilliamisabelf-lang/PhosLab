"""
Pantalla con círculo rojo de anclaje (anchor)
"""

import pygame


class AnchorScreen:
    """
    Muestra pantalla negra con círculo rojo de anclaje
    Detecta si el sujeto está mirando al círculo
    """

    def __init__(self, params, eye_tracker):
        """
        Inicializa la pantalla de anchor

        Args:
            params: Diccionario con configuración
            eye_tracker: Objeto EyeTracker
        """
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

        # Centro de la pantalla
        self.circle_center = (self.width // 2, self.height // 2)

        # Estado
        self.is_looking = False

        # Fuente para texto
        self.font = pygame.font.Font(None, 32)

        print("[AnchorScreen] Inicializado")

    def update(self, screen, eye_tracker_frame):
        """
        Actualiza y dibuja la pantalla de anclaje

        Args:
            screen: Superficie de Pygame donde dibujar
            eye_tracker_frame: Frame de la webcam para eye tracking

        Returns:
            bool: True si está mirando al círculo, False si no
        """
        # Detectar si está mirando al círculo
        if self.eye_tracker and eye_tracker_frame is not None:
            self.is_looking = self.eye_tracker.is_looking_at_point(
                frame=eye_tracker_frame,
                target_point=self.circle_center,
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
        """Dibuja la pantalla"""
        # Fondo negro
        screen.fill(self.background_color)

        # Color del círculo (verde si está mirando, rojo si no)
        if self.is_looking:
            circle_color = (0, 255, 0)  # Verde
        else:
            circle_color = self.circle_color  # Rojo

        # Dibujar círculo
        pygame.draw.circle(
            screen,
            circle_color,
            self.circle_center,
            self.circle_radius,
            self.circle_thickness,
        )

        # Texto de estado
        status_text = (
            "ANCLAJE DETECTADO" if self.is_looking else "Mira al circulo rojo"
        )
        text_color = (0, 255, 0) if self.is_looking else (255, 255, 0)

        text_surface = self.font.render(status_text, True, text_color)
        text_rect = text_surface.get_rect(center=(self.width // 2, 80))
        screen.blit(text_surface, text_rect)
    