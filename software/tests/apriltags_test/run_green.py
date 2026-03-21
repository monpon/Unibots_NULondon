"""
Unibots 2026 — GREEN Zone
==========================
Assigned zone: GREEN (left wall, AprilTags 20 & 21).

Serial protocol (extends wheels_control.cpp):
  M1:val,M2:val   — wheel pair per Arduino (existing)
  L:speed         — lift:   Arduino serA analogWrite(5, speed)    [NEW]
  C:1 / C:0       — intake: serA pin4→128, pin13→255 on/off       [NEW]

Two Arduinos:
  serB (PORT_B): M1=FL, M2=FR   (send_pwm(serB, fl, -fr))
  serA (PORT_A): M1=BR, M2=BL   (send_pwm(serA, -br, bl))

State machine:
  seek_ball → (ball at bottom) → intake_commit → (time low) → return_zone
  return_zone → (arrived) → turning → backing → lift_up → seek_ball / done

Usage:
  python3 run_green.py [--port-a /dev/ttyUSB0] [--port-b /dev/ttyUSB1]
                       [--tag-cam 0] [--ball-cam 1]
"""

import cv2
import numpy as np
import math
import serial
import time
import argparse

from april_tag_func import (
    direction_to_zone,
    CAMERA_FOV_HORIZONTAL,
)
from april_tag_camera import (
    detect_tags, get_robot_position,
    draw_crosshair, draw_tag_info, draw_robot_position, draw_no_tags,
)

# ============================================================
#  ZONE CONFIG
# ============================================================
TARGET_ZONE     = "green"
DEPOSIT_HEADING = 90         # face east (back into left wall at x=0)

# ============================================================
#  GAME TUNING
# ============================================================
GAME_DURATION    = 180
RETURN_TIME      = 30
ARRIVAL_DIST_MM  = 150
BOTTOM_THRESH    = 210
INTAKE_TIME      = 1.5
SPIN_SPEED       = 120
DRIVE_SPEED      = 200
ROTATE_KP        = 0.8
CENTER_DEAD      = 25
BACK_DURATION    = 1.5
BACK_SPEED       = 80
LIFT_UP_DURATION = 2.5
LIFT_SPEED       = 255       # Arduino: analogWrite(5, 255)

# ============================================================
#  CAMERA PARAMS
# ============================================================
TAG_CAM_W   = 1280
TAG_CAM_H   = 720
BALL_CAM_W  = 320
BALL_CAM_H  = 240
CAM_HEIGHT  = 150
CAM_TILT    = 15
BALL_H_FOV  = 90
BALL_V_FOV  = 50.625

BAUD = 115200


# ============================================================
#  MOTOR CONTROL
# ============================================================

def send_pwm(ser, m1, m2):
    ser.write(f"M1:{int(m1)},M2:{int(m2)}\n".encode())

def set_wheel_pwm(serA, serB, fl, fr, bl, br):
    send_pwm(serB, fl, -fr)
    send_pwm(serA, -br, bl)

def mecanum(serA, serB, fwd, strafe, rotate):
    fl =  fwd - strafe - rotate
    fr = -fwd - strafe + rotate
    bl = -fwd + strafe + rotate
    br =  fwd + strafe - rotate
    m = max(abs(fl), abs(fr), abs(bl), abs(br), 255.0)
    set_wheel_pwm(serA, serB, fl/m*255, fr/m*255, bl/m*255, br/m*255)

def stop_drive(serA, serB):
    set_wheel_pwm(serA, serB, 0, 0, 0, 0)

def set_lift(serA, speed):
    """L:speed → analogWrite(5, speed). One-direction only."""
    speed = max(0, min(255, int(speed)))
    serA.write(f"L:{speed}\n".encode())

def set_intake(serA, on: bool):
    """C:1 → pin4@128 + pin13@255.  C:0 → off."""
    serA.write(f"C:{1 if on else 0}\n".encode())


# ============================================================
#  BALL DETECTION
# ============================================================

def detect_ball_orange(frame, prev_circle=None):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 150, 150]), np.array([20, 255, 255]))
    res  = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 300,
                               param1=50, param2=20, minRadius=5, maxRadius=100)
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    chosen = None
    for c in circles[0, :]:
        if chosen is None:
            chosen = c
        elif prev_circle is not None:
            d_new = (int(c[0])-int(prev_circle[0]))**2 + (int(c[1])-int(prev_circle[1]))**2
            d_old = (int(chosen[0])-int(prev_circle[0]))**2 + (int(chosen[1])-int(prev_circle[1]))**2
            if d_new < d_old:
                chosen = c
    return chosen


# ============================================================
#  NAVIGATION HELPERS
# ============================================================

def drive_toward(serA, serB, distance_mm, rotation_deg, speed=DRIVE_SPEED):
    rad    = math.radians(rotation_deg)
    fwd    = speed * math.cos(rad)
    strafe = speed * math.sin(rad)
    rot    = max(-150.0, min(150.0, rotation_deg * 2.5))
    mecanum(serA, serB, fwd, strafe, rot)

def turn_toward(serA, serB, robot_h, target_h, speed=120, tol=5):
    err = target_h - robot_h
    if err > 180:  err -= 360
    if err < -180: err += 360
    if abs(err) <= tol:
        stop_drive(serA, serB)
        return True
    mecanum(serA, serB, 0, 0, speed if err > 0 else -speed)
    return False


# ============================================================
#  MAIN
# ============================================================

def run(port_a, port_b, tag_cam, ball_cam):
    serA = serial.Serial(port_a, BAUD, timeout=0.05)
    serB = serial.Serial(port_b, BAUD, timeout=0.05)
    time.sleep(2)
    for ser in (serA, serB):
        while ser.in_waiting:
            ser.readline()
    print(f"[INIT] Arduinos on {port_a} and {port_b}")

    tcap = cv2.VideoCapture(tag_cam)
    tcap.set(cv2.CAP_PROP_FRAME_WIDTH,  TAG_CAM_W)
    tcap.set(cv2.CAP_PROP_FRAME_HEIGHT, TAG_CAM_H)
    ret, tst = tcap.read()
    th, tw = tst.shape[:2] if ret else (TAG_CAM_H, TAG_CAM_W)
    focal_px = (tw / 2) / math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2))
    tcx, tcy = tw // 2, th // 2

    bcap = cv2.VideoCapture(ball_cam)
    bcap.set(cv2.CAP_PROP_FRAME_WIDTH,  BALL_CAM_W)
    bcap.set(cv2.CAP_PROP_FRAME_HEIGHT, BALL_CAM_H)
    bcap.set(cv2.CAP_PROP_EXPOSURE, -5)

    print(f"[INIT] Zone: {TARGET_ZONE.upper()} | Deposit hdg: {DEPOSIT_HEADING}°")

    robot_x = robot_y = robot_h = None
    ball = prev_ball = None
    balls_collected = 0
    game_start  = None
    game_state  = "seek_ball"
    state_start = 0.0
    turn_target = DEPOSIT_HEADING
    tframe = bframe = None

    try:
        while True:
            ret, tframe = tcap.read()
            if ret:
                gray = cv2.cvtColor(tframe, cv2.COLOR_BGR2GRAY)
                dets, raw_c, raw_ids = detect_tags(gray, tcx, tcy, focal_px, tw)
                if dets:
                    cv2.aruco.drawDetectedMarkers(tframe, raw_c, raw_ids, (0, 255, 0))
                    robot_x, robot_y, robot_h, per_tag = get_robot_position(dets)
                    draw_tag_info(tframe, dets, per_tag)
                    draw_robot_position(tframe, robot_x, robot_y, robot_h, len(dets))
                else:
                    robot_x = robot_y = robot_h = None
                    draw_no_tags(tframe)

            ret, bframe = bcap.read()
            if ret:
                ball = detect_ball_orange(bframe, prev_ball)
                if ball is not None:
                    prev_ball = ball
                    cv2.circle(bframe, (int(ball[0]), int(ball[1])), int(ball[2]),
                               (0, 165, 255), 2)

            if game_start is None:
                game_start = time.time()
                print(f"[GAME] Started — {GAME_DURATION}s, returning at T-{RETURN_TIME}s")
            elapsed   = time.time() - game_start
            remaining = GAME_DURATION - elapsed

            if remaining <= 0 and game_state != "done":
                game_state = "done"
                stop_drive(serA, serB)
                set_lift(serA, 0)
                set_intake(serA, False)
                print("[GAME] Time's up")

            elif remaining <= RETURN_TIME and game_state in ("seek_ball", "intake_commit"):
                game_state  = "return_zone"
                state_start = time.time()
                stop_drive(serA, serB)
                set_intake(serA, False)
                print(f"[GAME] {remaining:.0f}s left — heading to {TARGET_ZONE}!")

            # ================================================================
            if game_state == "seek_ball":
                set_intake(serA, True)
                if ball is not None:
                    x_err = int(ball[0]) - (BALL_CAM_W // 2)
                    if int(ball[1]) > BOTTOM_THRESH:
                        game_state  = "intake_commit"
                        state_start = time.time()
                        balls_collected += 1
                        print(f"[BALL] Committing to intake #{balls_collected}")
                    else:
                        rot = max(-150, min(150, int(ROTATE_KP * x_err))) \
                              if abs(x_err) > CENTER_DEAD else 0
                        mecanum(serA, serB, DRIVE_SPEED, 0, rot)
                else:
                    mecanum(serA, serB, 0, 0, SPIN_SPEED)

            elif game_state == "intake_commit":
                set_intake(serA, True)
                if time.time() - state_start < INTAKE_TIME:
                    mecanum(serA, serB, DRIVE_SPEED, 0, 0)
                else:
                    stop_drive(serA, serB)
                    game_state = "seek_ball"

            elif game_state == "return_zone":
                set_intake(serA, False)
                if robot_x is not None and robot_h is not None:
                    dist, rot = direction_to_zone(robot_x, robot_y, robot_h, TARGET_ZONE)
                    if dist is not None and dist < ARRIVAL_DIST_MM:
                        game_state  = "turning"
                        turn_target = DEPOSIT_HEADING
                        stop_drive(serA, serB)
                        print(f"[NAV] At {TARGET_ZONE} — turning to {DEPOSIT_HEADING}°")
                    elif dist is not None:
                        drive_toward(serA, serB, dist, rot)
                else:
                    mecanum(serA, serB, 0, 0, SPIN_SPEED)

            elif game_state == "turning":
                set_intake(serA, False)
                if robot_h is not None:
                    if turn_toward(serA, serB, robot_h, turn_target):
                        game_state  = "backing"
                        state_start = time.time()
                        mecanum(serA, serB, -BACK_SPEED, 0, 0)
                        print("[NAV] Aligned — backing into net")
                else:
                    stop_drive(serA, serB)

            elif game_state == "backing":
                set_intake(serA, False)
                if time.time() - state_start >= BACK_DURATION:
                    stop_drive(serA, serB)
                    game_state  = "lift_up"
                    state_start = time.time()
                    set_lift(serA, LIFT_SPEED)
                    print("[LIFT] Raising lift (pin5 @ 255)")
                else:
                    mecanum(serA, serB, -BACK_SPEED, 0, 0)

            elif game_state == "lift_up":
                set_intake(serA, False)
                if time.time() - state_start >= LIFT_UP_DURATION:
                    set_lift(serA, 0)
                    print(f"[LIFT] Deposited {balls_collected} ball(s). T-{remaining:.0f}s")
                    balls_collected = 0
                    if remaining > RETURN_TIME + 15:
                        game_state = "seek_ball"
                        print("[GAME] Heading out for more balls")
                    else:
                        game_state = "done"
                        stop_drive(serA, serB)
                        print("[GAME] Parking")

            elif game_state == "done":
                stop_drive(serA, serB)
                set_lift(serA, 0)
                set_intake(serA, False)

            if tframe is not None:
                draw_crosshair(tframe, tcx, tcy)
                cv2.putText(tframe,
                            f"[{TARGET_ZONE.upper()}] {game_state} | "
                            f"balls={balls_collected} | T-{remaining:.0f}s",
                            (10, th - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2)
                cv2.imshow(f"Tags [{TARGET_ZONE.upper()}]", tframe)
            if bframe is not None:
                cv2.imshow(f"Ball [{TARGET_ZONE.upper()}]", bframe)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted")

    finally:
        stop_drive(serA, serB)
        set_lift(serA, 0)
        set_intake(serA, False)
        serA.close()
        serB.close()
        tcap.release()
        bcap.release()
        cv2.destroyAllWindows()
        print("[EXIT] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Unibots 2026 — {TARGET_ZONE.upper()} Zone"
    )
    parser.add_argument("--port-a", default="/dev/ttyUSB0")
    parser.add_argument("--port-b", default="/dev/ttyUSB1")
    parser.add_argument("--tag-cam",  type=int, default=0)
    parser.add_argument("--ball-cam", type=int, default=1)
    args = parser.parse_args()
    run(args.port_a, args.port_b, args.tag_cam, args.ball_cam)
