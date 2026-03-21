"""
Unibots 2026 — Motor & Direction Test
=======================================
Tests every motor and direction against the actual wheels_control.cpp protocol.

Two Arduinos:
  serA (PORT_A /dev/ttyUSB0): BR wheel (M1), BL wheel (M2), lift pin5, intake pin4+pin13
  serB (PORT_B /dev/ttyUSB1): FL wheel (M1), FR wheel (M2)

Protocol:
  M1:val,M2:val   — wheels (both boards)
  L:speed         — lift motor  (serA, analogWrite pin 5)
  C:1 / C:0       — intake on/off (serA, pin4→128, pin13→255)

KEYBOARD CONTROLS
-----------------
Combined drive:
  W / S    Forward / Backward
  A / D    Strafe left / right
  Q / E    Rotate CCW / CW

Individual wheel (forward / reverse):
  1 / !    serA M1 = BR  (fwd / rev)
  2 / @    serA M2 = BL  (fwd / rev)
  3 / #    serB M1 = FL  (fwd / rev)
  4 / $    serB M2 = FR  (fwd / rev)

Auxiliary:
  U        Lift UP  (serA pin5 @ 255)
  C        Intake ON  (serA pin4@128 + pin13@255)
  X        Intake OFF
  +/-      Increase / decrease drive test speed

General:
  SPACE    STOP ALL
  ESC      Quit

Usage:
  python3 test_motors.py [--port-a /dev/ttyUSB0] [--port-b /dev/ttyUSB1]
"""

import cv2
import numpy as np
import serial
import time
import argparse

BAUD = 115200

# Test speeds
DEFAULT_DRIVE_SPEED  = 150
DEFAULT_SINGLE_SPEED = 120
LIFT_SPEED           = 255   # serA analogWrite(5, 255)

WINDOW_W = 720
WINDOW_H = 520


# ============================================================
#  PROTOCOL HELPERS
# ============================================================

def send_pwm(ser, m1, m2):
    """M1:val,M2:val — matches wheels_control.cpp."""
    ser.write(f"M1:{int(m1)},M2:{int(m2)}\n".encode())


def set_wheel_pwm(serA, serB, fl, fr, bl, br):
    """
    serB: M1=FL, M2=FR  (send as fl, -fr)
    serA: M1=BR, M2=BL  (send as -br, bl)
    """
    send_pwm(serB, fl, -fr)
    send_pwm(serA, -br, bl)


def mecanum(serA, serB, fwd, strafe, rotate):
    fl =  fwd - strafe - rotate
    fr = -fwd - strafe + rotate
    bl = -fwd + strafe + rotate
    br =  fwd + strafe - rotate
    m = max(abs(fl), abs(fr), abs(bl), abs(br), 255.0)
    set_wheel_pwm(serA, serB, fl/m*255, fr/m*255, bl/m*255, br/m*255)


def stop_all_wheels(serA, serB):
    send_pwm(serA, 0, 0)
    send_pwm(serB, 0, 0)


def set_lift(serA, speed):
    """L:speed → serA analogWrite(5, speed). One-direction only."""
    speed = max(0, min(255, int(speed)))
    serA.write(f"L:{speed}\n".encode())


def set_intake(serA, on: bool):
    """C:1 → pin4@128 + pin13@255.  C:0 → both off."""
    serA.write(f"C:{1 if on else 0}\n".encode())


# ============================================================
#  DISPLAY
# ============================================================

def draw_ui(canvas, action, speed, fl, fr, bl, br, lift_on, intake_on):
    canvas[:] = 20  # dark grey bg

    cv2.putText(canvas, "UNIBOTS 2026 — Motor Test",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2)
    cv2.line(canvas, (20, 50), (WINDOW_W - 20, 50), (70, 70, 70), 1)

    # Action
    cv2.putText(canvas, f"Action: {action}",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)
    cv2.putText(canvas, f"Speed: {speed}   (+/- to adjust)",
                (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

    # Wheel bars  [FL  FR  BL  BR]
    labels = [("FL\n(serB M1)", fl,  30),
              ("FR\n(serB M2)", fr, 200),
              ("BL\n(serA M2)", bl, 370),
              ("BR\n(serA M1)", br, 540)]

    for label, val, bx in labels:
        col = (0, 200, 60) if val > 0 else ((0, 60, 220) if val < 0 else (70, 70, 70))
        bh  = int(abs(val) / 255 * 65)
        top = 185 - bh if val >= 0 else 185
        cv2.rectangle(canvas, (bx, top), (bx + 120, 185), col, -1)
        cv2.rectangle(canvas, (bx, 120), (bx + 120, 185), (100, 100, 100), 1)
        for ln in label.split("\n"):
            cv2.putText(canvas, ln, (bx + 8, 205 + label.split("\n").index(ln) * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
        cv2.putText(canvas, str(int(val)), (bx + 35, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Aux motors
    lc = (0, 220, 60) if lift_on   else (70, 70, 70)
    ic = (0, 220, 60) if intake_on else (70, 70, 70)
    cv2.putText(canvas, f"LIFT (pin5@255): {'ON' if lift_on else 'off'}",
                (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.62, lc, 2)
    cv2.putText(canvas, f"INTAKE (pin4@128, pin13@255): {'ON' if intake_on else 'off'}",
                (20, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ic, 2)

    # Controls reference
    cv2.line(canvas, (20, 325), (WINDOW_W - 20, 325), (60, 60, 60), 1)
    lines = [
        "W/S: fwd/back    A/D: strafe L/R    Q/E: rotate CCW/CW",
        "1/!: serA-M1(BR)  2/@: serA-M2(BL)  3/#: serB-M1(FL)  4/$: serB-M2(FR)",
        "U: lift up (pin5@255)   C: intake on   X: intake off",
        "SPACE: stop all   +/-: speed   ESC: quit",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(canvas, ln, (20, 348 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)


# ============================================================
#  MAIN
# ============================================================

def run(port_a, port_b):
    serA = serial.Serial(port_a, BAUD, timeout=0.05)
    serB = serial.Serial(port_b, BAUD, timeout=0.05)
    time.sleep(2)
    for ser in (serA, serB):
        while ser.in_waiting:
            ser.readline()
    print(f"[TEST] Connected — serA={port_a}  serB={port_b}")
    print("[TEST] Click the window and use keyboard.")

    speed     = DEFAULT_DRIVE_SPEED
    fl = fr = bl = br = 0
    lift_on   = False
    intake_on = False
    action    = "STOPPED"

    cv2.namedWindow("Motor Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motor Test", WINDOW_W, WINDOW_H)
    canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

    try:
        while True:
            draw_ui(canvas, action, speed, fl, fr, bl, br, lift_on, intake_on)
            cv2.imshow("Motor Test", canvas)
            key = cv2.waitKey(50) & 0xFF

            if key == 255:
                continue

            # ---- Quit ----
            if key == 27:
                break

            # ---- Speed ----
            elif key in (ord('+'), ord('=')):
                speed = min(255, speed + 10)
                continue
            elif key in (ord('-'), ord('_')):
                speed = max(40, speed - 10)
                continue

            # ---- Stop all ----
            elif key == ord(' '):
                stop_all_wheels(serA, serB)
                set_lift(serA, 0)
                set_intake(serA, False)
                fl = fr = bl = br = 0
                lift_on = intake_on = False
                action = "STOPPED"
                print("[TEST] STOP ALL")
                continue

            # ---- Combined drive ----
            elif key == ord('w'):
                mecanum(serA, serB, speed, 0, 0)
                fl, fr, bl, br = speed, speed, speed, speed   # approximate for display
                action = "FORWARD"
            elif key == ord('s'):
                mecanum(serA, serB, -speed, 0, 0)
                fl, fr, bl, br = -speed, -speed, -speed, -speed
                action = "BACKWARD"
            elif key == ord('a'):
                mecanum(serA, serB, 0, -speed, 0)
                action = "STRAFE LEFT"
                fl, fr, bl, br = -speed, speed, speed, -speed
            elif key == ord('d'):
                mecanum(serA, serB, 0, speed, 0)
                action = "STRAFE RIGHT"
                fl, fr, bl, br = speed, -speed, -speed, speed
            elif key == ord('q'):
                mecanum(serA, serB, 0, 0, -speed)
                action = "ROTATE CCW"
                fl, fr, bl, br = -speed, speed, -speed, speed
            elif key == ord('e'):
                mecanum(serA, serB, 0, 0, speed)
                action = "ROTATE CW"
                fl, fr, bl, br = speed, -speed, speed, -speed

            # ---- Individual wheels (forward) ----
            elif key == ord('1'):
                s = DEFAULT_SINGLE_SPEED
                send_pwm(serA, s, 0)    # serA M1 = BR fwd → send -br so send -s? Actually:
                # serA M1 gets sent as -br, so to spin BR fwd we send +s here
                # (driveMotor handles sign for direction)
                fl, fr, bl, br = 0, 0, 0, s
                action = "BR wheel fwd (serA M1)"
            elif key == ord('2'):
                s = DEFAULT_SINGLE_SPEED
                send_pwm(serA, 0, s)    # serA M2 = BL
                fl, fr, bl, br = 0, 0, s, 0
                action = "BL wheel fwd (serA M2)"
            elif key == ord('3'):
                s = DEFAULT_SINGLE_SPEED
                send_pwm(serB, s, 0)    # serB M1 = FL
                fl, fr, bl, br = s, 0, 0, 0
                action = "FL wheel fwd (serB M1)"
            elif key == ord('4'):
                s = DEFAULT_SINGLE_SPEED
                send_pwm(serB, 0, s)    # serB M2 = FR (inverted by -fr in normal drive)
                fl, fr, bl, br = 0, s, 0, 0
                action = "FR wheel fwd (serB M2)"

            # ---- Individual wheels (reverse) ----
            elif key == ord('!'):
                s = DEFAULT_SINGLE_SPEED
                send_pwm(serA, -s, 0)
                fl, fr, bl, br = 0, 0, 0, -s
                action = "BR wheel rev (serA M1)"
            elif key == ord('@'):
                s = DEFAULT_SINGLE_SPEED
                send_pwm(serA, 0, -s)
                fl, fr, bl, br = 0, 0, -s, 0
                action = "BL wheel rev (serA M2)"
            elif key == ord('#'):
                s = DEFAULT_SINGLE_SPEED
                send_pwm(serB, -s, 0)
                fl, fr, bl, br = -s, 0, 0, 0
                action = "FL wheel rev (serB M1)"
            elif key == ord('$'):
                s = DEFAULT_SINGLE_SPEED
                send_pwm(serB, 0, -s)
                fl, fr, bl, br = 0, -s, 0, 0
                action = "FR wheel rev (serB M2)"

            # ---- Lift ----
            elif key == ord('u'):
                set_lift(serA, LIFT_SPEED)
                lift_on = True
                action = "LIFT UP (serA pin5 @ 255)"

            # ---- Intake ----
            elif key == ord('c'):
                set_intake(serA, True)
                intake_on = True
                action = "INTAKE ON (pin4@128 + pin13@255)"
            elif key == ord('x'):
                set_intake(serA, False)
                intake_on = False
                action = "INTAKE OFF"

            else:
                continue

            print(f"[TEST] {action}  FL={int(fl)} FR={int(fr)} BL={int(bl)} BR={int(br)} "
                  f"lift={'ON' if lift_on else 'off'} intake={'ON' if intake_on else 'off'}")

    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted")

    finally:
        stop_all_wheels(serA, serB)
        set_lift(serA, 0)
        set_intake(serA, False)
        serA.close()
        serB.close()
        cv2.destroyAllWindows()
        print("[EXIT] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unibots 2026 — Motor Test")
    parser.add_argument("--port-a", default="/dev/ttyUSB0",
                        help="Arduino A: BR/BL wheels + lift + intake")
    parser.add_argument("--port-b", default="/dev/ttyUSB1",
                        help="Arduino B: FL/FR wheels")
    args = parser.parse_args()
    run(args.port_a, args.port_b)
