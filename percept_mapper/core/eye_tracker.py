"""
Eye Tracker - Detección de mirada usando MediaPipe

Carga automáticamente percept_mapper/config/gaze_calibration.json si existe
(generado por calibrate_gaze.py). Sin él usa un mapeo normalizado de
emergencia que suele ser impreciso pero permite arrancar sin calibrar.
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_CALIB_PATH = Path(__file__).resolve().parents[1] / "config" / "gaze_calibration.json"


class EyeTracker:
    """
    Detecta la mirada del usuario usando la webcam y MediaPipe Face Mesh
    """

    def __init__(self, camera_index=0):
        """
        Inicializa el eye tracker

        Args:
            camera_index: Índice de la cámara (0 = webcam por defecto)
        """
        print("[EyeTracker] Inicializando...")

        # ============================================
        # 1. ABRIR WEBCAM
        # ============================================
        # Intentar con diferentes backends y índices
        self.cap = None

        # Lista de backends a probar en Windows
        backends = [
            cv2.CAP_DSHOW,  # DirectShow (Windows)
            cv2.CAP_MSMF,  # Microsoft Media Foundation
            cv2.CAP_ANY,  # Autodetección
        ]

        # Intentar abrir la cámara con diferentes backends
        for backend in backends:
            print(
                f"[EyeTracker] Intentando abrir cámara {camera_index} con backend {backend}..."
            )
            self.cap = cv2.VideoCapture(camera_index, backend)

            if self.cap.isOpened():
                # Verificar que realmente funciona capturando un frame
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(
                        f"[EyeTracker] ✓ Cámara {camera_index} abierta correctamente con backend {backend}"
                    )
                    break
                else:
                    self.cap.release()
                    self.cap = None
            else:
                if self.cap:
                    self.cap.release()
                self.cap = None

        # Si no funcionó, intentar con otros índices de cámara (0, 1, 2)
        if self.cap is None or not self.cap.isOpened():
            print("[EyeTracker] Intentando otros índices de cámara...")
            for cam_idx in [1, 2]:
                for backend in backends:
                    print(
                        f"[EyeTracker] Probando cámara {cam_idx} con backend {backend}..."
                    )
                    self.cap = cv2.VideoCapture(cam_idx, backend)

                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            print(
                                f"[EyeTracker] ✓ Cámara {cam_idx} funciona (backend {backend})"
                            )
                            camera_index = cam_idx
                            break
                        else:
                            self.cap.release()
                            self.cap = None
                    else:
                        if self.cap:
                            self.cap.release()
                        self.cap = None

                if self.cap and self.cap.isOpened():
                    break

        # Si aún no funciona, lanzar excepción con información útil
        if self.cap is None or not self.cap.isOpened():
            error_msg = (
                "No se pudo abrir ninguna cámara.\n"
                "Posibles soluciones:\n"
                "1. Verifica que la cámara esté conectada\n"
                "2. Verifica permisos de cámara en Windows (Configuración > Privacidad > Cámara)\n"
                "3. Cierra otras aplicaciones que usen la cámara (Zoom, Teams, etc.)\n"
                "4. Prueba desconectar/reconectar la cámara USB"
            )
            raise Exception(error_msg)

        # Configurar resolución (más pequeña = más rápido)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ============================================
        # 2. INICIALIZAR MEDIAPIPE FACE MESH
        # ============================================
        # MediaPipe Face Mesh detecta 478 puntos en la cara
        # Incluye puntos específicos para los iris (ojos)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Solo detectar 1 cara
            refine_landmarks=True,  # Incluir landmarks de iris (⭐ IMPORTANTE)
            min_detection_confidence=0.2,  # Confianza mínima para detectar
            min_tracking_confidence=0.2,  # Confianza mínima para seguir
        )

        # ============================================
        # 3. ÍNDICES DE LOS IRIS
        # ============================================
        # MediaPipe Face Mesh tiene puntos específicos para los iris:
        # - Iris izquierdo: landmarks 474-477 (4 puntos)
        # - Iris derecho: landmarks 469-472 (4 puntos)
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        # ============================================
        # 4. BUFFER PARA SUAVIZADO TEMPORAL
        # ============================================
        # Guardar las últimas N detecciones para promediar
        # Esto reduce el "jitter" (temblor) en la detección
        self.gaze_buffer = []
        self.buffer_size = 10  # Promediar últimas 10 detecciones

        # Last computed gaze points (exposed for external use)
        self.last_raw_gaze = None  # Raw unsmoothed gaze point (x, y)
        self.last_smooth_gaze = None  # Smoothed gaze point (x, y)

        # ── Calibración afín (cargada desde gaze_calibration.json) ──
        self._calib_coeff_x: np.ndarray | None = None
        self._calib_coeff_y: np.ndarray | None = None
        if _CALIB_PATH.exists():
            try:
                data = json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
                self._calib_coeff_x = np.array(data["coeff_x"], dtype=float)
                self._calib_coeff_y = np.array(data["coeff_y"], dtype=float)
                print(
                    f"[EyeTracker] ✓ Calibración cargada ({data.get('n_samples', '?')} muestras, "
                    f"error residual: {data.get('mean_residual_px', '?')} px)"
                )
            except Exception as e:
                print(f"[EyeTracker] ⚠ No se pudo cargar calibración: {e}")
        else:
            print("[EyeTracker] ⚠ Sin calibración — ejecuta calibrate_gaze.py para mejorar la precisión")

        print("[EyeTracker] ✓ Inicializado correctamente")

    def get_frame(self):
        """
        Captura un frame de la webcam

        Returns:
            numpy.ndarray: Frame capturado (formato BGR de OpenCV)
        """
        ret, frame = self.cap.read()

        if not ret:
            print("[EyeTracker] ⚠ No se pudo capturar frame")
            return None

        return frame

    def update_gaze(self, frame, screen_size):
        """
        Procesa un frame y actualiza last_raw_gaze / last_smooth_gaze, sin
        comprobar ningún punto objetivo. Necesario para que la mirada siga
        siendo fresca en fases que no comprueban fijación (p.ej. la ventana
        de captura de saccade), ya que last_smooth_gaze solo se refresca aquí.

        Args:
            frame: Frame de la webcam (numpy array BGR)
            screen_size: (width, height) tamaño de la pantalla

        Returns:
            (x, y) punto suavizado, o None si no se detectó cara/iris.
        """
        if frame is None:
            return None

        # OpenCV usa BGR, MediaPipe usa RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        # Tomar la primera cara detectada
        face_landmarks = results.multi_face_landmarks[0]

        gaze_point = self._get_gaze_point(face_landmarks, frame.shape, screen_size)
        if gaze_point is None:
            return None

        self.last_raw_gaze = gaze_point

        # Suavizado temporal: promedio de las últimas N detecciones
        self.gaze_buffer.append(gaze_point)
        if len(self.gaze_buffer) > self.buffer_size:
            self.gaze_buffer.pop(0)
        avg_gaze = np.mean(self.gaze_buffer, axis=0)

        self.last_smooth_gaze = (float(avg_gaze[0]), float(avg_gaze[1]))
        return self.last_smooth_gaze

    def is_looking_at_point(
        self, frame, target_point, screen_size, tolerance_radius=100
    ):
        """
        Detecta si el usuario está mirando a un punto específico

        Args:
            frame: Frame de la webcam (numpy array BGR)
            target_point: (x, y) punto objetivo en coordenadas de pantalla
            screen_size: (width, height) tamaño de la pantalla
            tolerance_radius: Radio de tolerancia en píxeles

        Returns:
            bool: True si está mirando al punto, False si no
        """
        avg_gaze = self.update_gaze(frame, screen_size)
        if avg_gaze is None:
            return False

        distance = np.sqrt(
            (avg_gaze[0] - target_point[0]) ** 2 + (avg_gaze[1] - target_point[1]) ** 2
        )

        is_within = distance <= tolerance_radius
        if not is_within:
            print(
                f"[EyeTracker] Gaze miss: smooth=({avg_gaze[0]:.0f},{avg_gaze[1]:.0f}) "
                f"target=({target_point[0]},{target_point[1]}) "
                f"dist={distance:.1f}px tol={tolerance_radius}px"
            )

        return is_within

    def _get_gaze_point(self, face_landmarks, frame_shape, screen_size):
        """
        Calcula el punto de mirada en coordenadas de pantalla

        Args:
            face_landmarks: Landmarks de MediaPipe
            frame_shape: (height, width, channels) del frame de la webcam
            screen_size: (width, height) de la pantalla

        Returns:
            tuple: (x, y) punto de mirada en coordenadas de pantalla, o None
        """
        frame_h, frame_w = frame_shape[:2]
        screen_w, screen_h = screen_size

        # ============================================
        # PASO 1: OBTENER COORDENADAS DE LOS IRIS
        # ============================================
        left_iris_points = []
        right_iris_points = []

        # Iris izquierdo (landmarks 474-477)
        for idx in self.LEFT_IRIS:
            landmark = face_landmarks.landmark[idx]
            # Convertir coordenadas normalizadas [0,1] a píxeles
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            left_iris_points.append([x, y])

        # Iris derecho (landmarks 469-472)
        for idx in self.RIGHT_IRIS:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            right_iris_points.append([x, y])

        # ============================================
        # PASO 2: CALCULAR CENTRO DE CADA IRIS
        # ============================================
        # Promedio de los 4 puntos de cada iris
        left_center = np.mean(left_iris_points, axis=0)
        right_center = np.mean(right_iris_points, axis=0)

        # ============================================
        # PASO 3: PROMEDIO DE AMBOS OJOS
        # ============================================
        # Usar el promedio de ambos ojos para mayor precisión
        iris_center = (left_center + right_center) / 2

        # ============================================
        # PASO 4: MAPEAR A COORDENADAS DE PANTALLA
        # ============================================
        if self._calib_coeff_x is not None:
            # Modelo afín calibrado: screen ≈ [ix, iy, 1] · coeff
            v = np.array([iris_center[0], iris_center[1], 1.0])
            gaze_x = float(self._calib_coeff_x @ v)
            gaze_y = float(self._calib_coeff_y @ v)
        else:
            # Fallback normalizado (sin calibración — menos preciso)
            normalized_x = 1.0 - (iris_center[0] / frame_w)
            normalized_y = iris_center[1] / frame_h
            gaze_x = normalized_x * screen_w
            gaze_y = normalized_y * screen_h

        gaze_x = float(np.clip(gaze_x, 0, screen_w))
        gaze_y = float(np.clip(gaze_y, 0, screen_h))

        return (gaze_x, gaze_y)

    def release(self):
        """Libera la webcam"""
        print("[EyeTracker] Liberando recursos...")

        if self.cap:
            self.cap.release()

        print("[EyeTracker] ✓ Recursos liberados")
