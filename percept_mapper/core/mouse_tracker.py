"""
Mouse Tracker - Uses mouse position as gaze input

Drop-in replacement for EyeTracker. Provides the same interface
so both can be used interchangeably based on config.
"""

import numpy as np
import pygame


class MouseTracker:
    """
    Tracks mouse position as a substitute for eye gaze.
    Same interface as EyeTracker for seamless swapping.
    """

    def __init__(self):
        print("[MouseTracker] Inicializando...")
        self.last_raw_gaze = None
        self.last_smooth_gaze = None
        self.gaze_buffer = []
        self.buffer_size = 5
        print("[MouseTracker] Listo - usando raton como entrada")

    def get_frame(self):
        """No webcam needed for mouse mode. Returns a dummy value."""
        return "mouse"

    def is_looking_at_point(self, frame, target_point, screen_size, tolerance_radius=100):
        """
        Check if the mouse cursor is within tolerance of the target point.

        Args:
            frame: Ignored (kept for interface compatibility)
            target_point: (x, y) target in screen coordinates
            screen_size: (width, height) of the screen
            tolerance_radius: Pixel radius for "looking" detection

        Returns:
            bool: True if mouse is within tolerance of target
        """
        mouse_pos = pygame.mouse.get_pos()
        self.last_raw_gaze = mouse_pos

        # Smooth with simple buffer
        self.gaze_buffer.append(mouse_pos)
        if len(self.gaze_buffer) > self.buffer_size:
            self.gaze_buffer.pop(0)
        avg = np.mean(self.gaze_buffer, axis=0)
        self.last_smooth_gaze = (float(avg[0]), float(avg[1]))

        distance = np.sqrt(
            (mouse_pos[0] - target_point[0]) ** 2
            + (mouse_pos[1] - target_point[1]) ** 2
        )

        is_within = distance <= tolerance_radius
        if not is_within:
            print(
                f"[MouseTracker] Miss: pos=({mouse_pos[0]},{mouse_pos[1]}) "
                f"target=({target_point[0]},{target_point[1]}) "
                f"dist={distance:.1f}px tol={tolerance_radius}px"
            )

        return is_within

    def release(self):
        """Nothing to release for mouse input."""
        print("[MouseTracker] Recursos liberados")
