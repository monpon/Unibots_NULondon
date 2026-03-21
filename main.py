import cv2
import numpy as np
import math
import serial
import time
import threading

# ── Serial ────────────────────────────────────────────────────
PORT_A = "/dev/ttyUSB0"
PORT_B = "/dev/ttyUSB1"
BAUD   = 115200

serA = serial.Serial(PORT_A, BAUD, timeout=1)
serB = serial.Serial(PORT_B, BAUD, timeout=1)
time.sleep(2)

# ── Drive helpers ─────────────────────────────────────────────
def send_pwm(ser, m1, m2):
    ser.write(f"M1:{int(m1)},M2:{int(m2)}\n".encode())

def set_wheel_pwm(fl, fr, bl, br):
    send_pwm(serB, fl, -fr)
    send_pwm(serA, -br, bl)

def mecanum(fwd, strafe, rotate):
    # left side is front
    fl =  fwd - strafe - rotate
    fr = -fwd - strafe + rotate
    bl = -fwd + strafe + rotate
    br =  fwd + strafe - rotate
    m  = max(abs(fl), abs(fr), abs(bl), abs(br), 255.0)
    set_wheel_pwm(fl/m*255, fr/m*255, bl/m*255, br/m*255)

def stop():
    set_wheel_pwm(0, 0, 0, 0)

# ── Tuning ────────────────────────────────────────────────────
SPIN_SPEED    = 120   # PWM while searching
DRIVE_SPEED   = 200   # PWM while chasing
ROTATE_KP     = 0.8   # keep ball centered
CENTER_DEAD   = 25    # pixel deadband
APPROACH_DIST = 150   # mm — stop driving when this close
INTAKE_TIME   = 1.5   # seconds to commit after ball disappears
BOTTOM_THRESH = 210   # y pixel — below = commit to intake

# ── Camera ────────────────────────────────────────────────────
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

CAMERA_HEIGHT  = 150
CAMERA_TILT    = 15
HORIZONTAL_FOV = 90
VERTICAL_FOV   = 50.625
CX, CY         = 160, 120

dist_fn = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2

def get_ball_pos(circ):
    x_comp     = circ[0] - CX
    y_comp     = circ[1] - CY
    x_angle_rad = math.radians(x_comp * (HORIZONTAL_FOV / 320))
    y_angle_rad = math.radians(y_comp * (VERTICAL_FOV   / 240))
    tilt_rad    = math.radians(CAMERA_TILT)
    eff_down    = tilt_rad + y_angle_rad
    if eff_down <= 0:
        return None
    fwd_mm = CAMERA_HEIGHT / math.tan(eff_down)
    lat_mm = fwd_mm * math.tan(x_angle_rad)
    return lat_mm, fwd_mm

# ── Main autonomous loop ──────────────────────────────────────
prevCircle   = None
circ1        = None
deltax = deltay = deltas = 0
frame_count  = 0
state        = "search"
intake_start = None

print("Autonomous ball seeker running. Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1

        # ── Vision ──
        if frame_count % 2 == 0:
            hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask    = cv2.inRange(hsv, np.array([5,150,150]), np.array([20,255,255]))
            res     = cv2.bitwise_and(frame, frame, mask=mask)
            gray    = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            blur    = cv2.GaussianBlur(gray, (9,9), 0)
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 300,
                                        param1=50, param2=20,
                                        minRadius=5, maxRadius=100)
            chosen = None
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    if chosen is None:
                        chosen = i
                    elif prevCircle is not None:
                        if dist_fn(*i[:2], *prevCircle[:2]) < dist_fn(*chosen[:2], *prevCircle[:2]):
                            chosen = i

            if chosen is not None:
                circ1      = list(chosen)
                prevCircle = chosen
                deltax = deltay = deltas = 0
                if state == "search":
                    print("[DETECT] Ball found, switching to chase.")
                    state = "chase"
            else:
                if circ1 is not None:
                    circ1[0] += deltax
                    circ1[1] += deltay
                    circ1[2] += deltas
        else:
            if circ1 is not None:
                circ1[0] += deltax
                circ1[1] += deltay
                circ1[2] += deltas

        # ── Control ──
        if state == "search":
            mecanum(0, 0, SPIN_SPEED)

        elif state == "chase" and circ1 is not None:
            # check if ball left frame
            if circ1[0] < 0 or circ1[0] > 320 or circ1[1] < 0:
                print("[CHASE] Ball lost, searching.")
                state = "search"
                circ1 = None
                stop()
                continue

            # ball near bottom — commit to intake
            if circ1[1] > BOTTOM_THRESH or circ1[1] > 240:
                print("[CHASE] Ball in intake zone, committing.")
                state        = "intake"
                intake_start = time.time()
                continue

            pos = get_ball_pos(circ1)
            if pos is None:
                state = "search"
                stop()
                continue

            lat_mm, fwd_mm = pos
            x_err = circ1[0] - CX

            # drive toward ball (camera forward = robot left = fwd in our frame)
            drive = DRIVE_SPEED if fwd_mm > APPROACH_DIST else 0

            # rotate to center ball horizontally
            rotate = 0
            if abs(x_err) > CENTER_DEAD:
                rotate = int(ROTATE_KP * x_err)
                rotate = max(-150, min(150, rotate))

            mecanum(drive, 0, rotate)

        elif state == "intake":
            if time.time() - intake_start < INTAKE_TIME:
                mecanum(DRIVE_SPEED, 0, 0)
            else:
                stop()
                print("[INTAKE] Done. Searching for next ball.")
                state = "search"
                circ1 = None

except KeyboardInterrupt:
    stop()
    cap.release()
    print("Stopped.")