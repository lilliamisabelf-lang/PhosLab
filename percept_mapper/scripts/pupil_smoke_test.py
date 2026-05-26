"""Quick diagnostic: subscribe to Pupil Capture and report what's flowing.

Run from repo root:
    uv run --project percept_mapper python percept_mapper/scripts/pupil_smoke_test.py
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import zmq
import msgpack
import yaml

DEFAULT_ADDRESS = "127.0.0.1"
DEFAULT_REQ_PORT = 50020
DEFAULT_SURFACE_NAME = "phoslab_screen"
WATCH_SECONDS = 5.0


def load_pupil_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "params.yaml"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[smoke] ⚠ no se pudo leer params.yaml: {e}")
        cfg = {}

    pupil_cfg = cfg.get("pupil") or {}
    return {
        "address": pupil_cfg.get("address", DEFAULT_ADDRESS),
        "port": int(pupil_cfg.get("port", DEFAULT_REQ_PORT)),
        "surface_name": pupil_cfg.get("surface_name", DEFAULT_SURFACE_NAME),
    }


def main():
    pupil_cfg = load_pupil_config()
    address = pupil_cfg["address"]
    req_port = pupil_cfg["port"]
    expected_surface = pupil_cfg["surface_name"]

    ctx = zmq.Context.instance()

    print(f"[smoke] REQ tcp://{address}:{req_port} ...")
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 2000)
    req.setsockopt(zmq.SNDTIMEO, 2000)
    try:
        req.connect(f"tcp://{address}:{req_port}")
        req.send_string("SUB_PORT")
        sub_port = req.recv_string()
    except zmq.Again:
        print("[smoke] ✗ Pupil Capture no responde en el puerto REQ.")
        print("        Comprueba: Pupil Capture abierto y 'Pupil Remote' activado en :50020.")
        sys.exit(2)
    print(f"[smoke] ✓ SUB_PORT={sub_port}")

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{address}:{sub_port}")

    # Subscribe broadly so we can tier the diagnosis:
    #   pupil.*    -> eye cameras + pupil detection running (pre-calibration)
    #   gaze.*     -> calibration done, gaze in world-camera coords
    #   surfaces.* -> Surface Tracker plugin + tagged screen surface
    for topic in ("surfaces.", "gaze.", "fixations", "pupil."):
        sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    print(f"[smoke] Escuchando {WATCH_SECONDS:.0f}s... mueve los ojos por la pantalla.\n")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    surface_names = set()
    surface_with_gaze = set()
    pupil_count = 0
    gaze_only_count = 0
    fixations_count = 0
    other_count = 0

    deadline = time.time() + WATCH_SECONDS
    while time.time() < deadline:
        socks = dict(poller.poll(timeout=200))
        if sub not in socks:
            continue
        try:
            topic = sub.recv_string(zmq.NOBLOCK)
            payload = sub.recv(zmq.NOBLOCK)
        except zmq.Again:
            continue
        try:
            msg = msgpack.unpackb(payload, raw=False)
        except Exception:
            continue

        if topic.startswith("surfaces."):
            name = topic[len("surfaces."):]
            if name not in surface_names:
                print(f"[smoke] superficie detectada: '{name}'")
                surface_names.add(name)
            if isinstance(msg, dict):
                gos = msg.get("gaze_on_surfaces") or msg.get("gaze_on_srf") or []
                if gos and name not in surface_with_gaze:
                    # report norm_pos + confidence of first valid sample
                    first = next(
                        (
                            g for g in gos
                            if isinstance(g, dict)
                            and g.get("norm_pos") is not None
                        ),
                        None,
                    )
                    if first is not None:
                        np_ = first.get("norm_pos")
                        conf = first.get("confidence", "n/a")
                        print(
                            f"[smoke]   → gaze_on_surface ok: norm_pos={np_} conf={conf}"
                        )
                        surface_with_gaze.add(name)
        elif topic.startswith("gaze."):
            gaze_only_count += 1
        elif topic.startswith("fixations"):
            fixations_count += 1
        elif topic.startswith("pupil."):
            pupil_count += 1
        else:
            other_count += 1

    print("\n[smoke] === RESUMEN ===")
    print(f"[smoke] muestras pupil.*        : {pupil_count}      (eye cameras + pupil detection)")
    print(f"[smoke] muestras gaze.*         : {gaze_only_count}      (calibracion completada)")
    print(f"[smoke] muestras fixations      : {fixations_count}")
    print(f"[smoke] superficies vistas      : {sorted(surface_names) or 'ninguna'}")
    print(f"[smoke] superficies con gaze    : {sorted(surface_with_gaze) or 'ninguna'}")
    print(f"[smoke] otros mensajes          : {other_count}")

    # Tiered diagnosis
    print("\n[smoke] === DIAGNOSTICO ===")
    if pupil_count == 0:
        print("[smoke] ✗ NO llegan muestras pupil.*")
        print("        -> Las camaras de ojo no estan capturando o no se detecta pupila.")
        print("        Acciones en Pupil Capture:")
        print("          1) Conecta el headset Pupil Core (o activa las camaras de ojo).")
        print("          2) En las ventanas 'Pupil Cam ID 0/1' verifica que se ve el ojo")
        print("             y que el algoritmo de deteccion (circulo rojo) sigue a la pupila.")
    elif gaze_only_count == 0:
        print("[smoke] ✓ pupil.* llega (deteccion de pupila OK)")
        print("[smoke] ✗ NO llega gaze.* -> falta calibrar.")
        print("        En Pupil Capture: 'Calibration' -> elige metodo (screen marker o")
        print("        natural features) y ejecuta la rutina de calibracion.")
    elif not surface_names:
        print("[smoke] ✓ gaze.* llega (calibracion OK)")
        print("[smoke] ✗ NO se publican superficies -> falta Surface Tracker.")
        print("        En Pupil Capture: activa el plugin 'Surface Tracker',")
        print("        imprime/pega AprilTag markers en las esquinas del monitor,")
        print("        y define una superficie con nombre.")

    # Comparar con lo que PhosLab espera (params.yaml -> pupil.surface_name)
    print(f"\n[smoke] PhosLab espera surface_name='{expected_surface}'")
    if expected_surface in surface_with_gaze:
        print("[smoke] ✓ Listo. PhosLab deberia recibir gaze al cambiar input_mode: pupil.")
    elif expected_surface in surface_names:
        print("[smoke] ⚠ La superficie existe pero no llego gaze valido todavia.")
        print("        Recalibra y mira a la pantalla mientras corre este test.")
    elif surface_names:
        print("[smoke] ⚠ La superficie esperada no se publica. Cambia 'surface_name'")
        print(f"        en config/params.yaml a una de: {sorted(surface_names)}")
    else:
        print("[smoke] ⚠ No se vieron superficies. En Pupil Capture activa el plugin")
        print("        'Surface Tracker', coloca AprilTags en la pantalla y crea una")
        print("        superficie con el nombre esperado.")

    sub.close(0)
    req.close(0)


if __name__ == "__main__":
    main()
