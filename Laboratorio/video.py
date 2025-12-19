import cv2
from datetime import datetime
from pathlib import Path

# -------- config de la prueba --------
TEST_ID = "EVA-LIVE-001"
COMPONENT = "Braille OCR"
OPERATOR = "Mizael"
SAVE_DIR = Path("recordings")

# Cámara: 0 suele ser /dev/video0
CAM_INDEX = 0

# Video output
SAVE_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = SAVE_DIR / f"{ts}_{TEST_ID}_{COMPONENT.replace(' ', '-')}.mp4"

cap = cv2.VideoCapture(CAM_INDEX)

# Ajustes (pueden depender de la cámara)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("No pude abrir la cámara (revisa /dev/video* o permisos)")

# Lee un frame para conocer tamaño real
ok, frame = cap.read()
if not ok:
    raise RuntimeError("No pude leer frames de la cámara")

h, w = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

def draw_overlay(frame, lines, origin=(20, 20)):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_h = 22
    pad = 10

    max_w = 0
    for s in lines:
        (tw, _), _ = cv2.getTextSize(s, font, font_scale, thickness)
        max_w = max(max_w, tw)

    panel_w = max_w + 2 * pad
    panel_h = len(lines) * line_h + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    ty = y + pad + 16
    for s in lines:
        cv2.putText(frame, s, (x + pad, ty), font, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        ty += line_h
    return frame

start = datetime.now()
frame_idx = 0

print("Grabando... (q = salir)")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    now = datetime.now()
    t_sec = (now - start).total_seconds()

    lines = [
        f"Fecha: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Prueba: {TEST_ID}",
        f"Componente: {COMPONENT}",
        f"Operador: {OPERATOR}",
        f"t={t_sec:0.2f}s | Frame={frame_idx} | {w}x{h}",
    ]

    draw_overlay(frame, lines, origin=(20, 20))

    writer.write(frame)
    cv2.imshow("EVA Live Recorder", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    frame_idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Listo: {out_path}")
