import cv2
import numpy as np
import time
from april_tag_func import (
    connect_arduino,
    disconnect_arduino,
    stop_motors,
    detect_ball,
    ball_position_relative,
    ball_position_room,
    direction_to_ball,
    direction_to_zone,
    direction_to_tag,
    direction_to_point,
    drive_to_zone,
    drive_to_tag,
    drive_to_ball,
    drive_to_point,
    drive_forward,
    drive_backward,
    strafe_left,
    strafe_right,
    rotate_cw,
    rotate_ccw,
    stop_motors,
    turn_around,
    is_robot_in_zone,
    get_robot_zone,
    CAMERA_FOV_HORIZONTAL,
)
from april_tag_camera import (
    detect_tags,
    get_robot_position,
    draw_crosshair,
    draw_tag_info,
    draw_robot_position,
    draw_no_tags,
)

# ============================================================
#  CONFIG
# ============================================================

SERIAL_PORT    = "/dev/ttyUSB0"
CAMERA_INDEX   = 1
CAPTURE_WIDTH  = 2560
CAPTURE_HEIGHT = 1440
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

# Ball camera (if separate, change index; if same camera, keep same)
BALL_CAM_INDEX = 1
BALL_CAM_W     = 320
BALL_CAM_H     = 240

# ============================================================
#  SETUP
# ============================================================

# Arduino
ser = connect_arduino(SERIAL_PORT)

# Tag camera
tag_cap = cv2.VideoCapture(CAMERA_INDEX)
tag_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
tag_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

ret, test = tag_cap.read()
if ret:
    fh, fw = test.shape[:2]
else:
    fw, fh = CAPTURE_WIDTH, CAPTURE_HEIGHT

cx = fw // 2
cy = fh // 2
focal_px = (fw / 2) / np.tan(np.radians(CAMERA_FOV_HORIZONTAL / 2))

# Ball camera (comment out if using same camera)
# ball_cap = cv2.VideoCapture(BALL_CAM_INDEX)
# ball_cap.set(cv2.CAP_PROP_FRAME_WIDTH, BALL_CAM_W)
# ball_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BALL_CAM_H)

print(f"[INIT] Camera {fw}x{fh} | focal={focal_px:.0f}px")
print("[INIT] Ready — add your logic below\n")

# ============================================================
#  STATE (available for your logic)
# ============================================================

robot_x   = None
robot_y   = None
robot_h   = None
ball      = None       # (x, y, r) in pixels or None
prev_ball = None

# Game timing
GAME_DURATION  = 180            # 3 minutes total
RETURN_TIME    = 30             # seconds before end to head back to zone
game_start     = None           # set on first loop iteration

# Navigation state machine
TARGET_ZONE    = "purple"       # change to whichever zone the robot should go to
ARRIVAL_DIST   = 100            # mm — close enough to count as "arrived"
BACK_DURATION  = 1.0            # seconds to reverse into wall
BACK_SPEED     = 80             # low PWM for the nudge

# Game states: "seek_ball" → "return_zone" → "turning" → "backing" → "done"
game_state     = "seek_ball"
turn_target_h  = None
back_start     = 0.0

# ============================================================
#  MAIN LOOP
# ============================================================

try:
    while True:
        ret, frame = tag_cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- AprilTag detection → robot position ----
        detections, raw_c, raw_ids = detect_tags(gray, cx, cy, focal_px, fw)

        if detections:
            cv2.aruco.drawDetectedMarkers(frame, raw_c, raw_ids, (0, 255, 0))
            robot_x, robot_y, robot_h, per_tag = get_robot_position(detections)
            draw_tag_info(frame, detections, per_tag)
            draw_robot_position(frame, robot_x, robot_y, robot_h, len(detections))
        else:
            robot_x = robot_y = robot_h = None
            draw_no_tags(frame)

        # ---- Ball detection (using tag camera) ----
        ball, mask = detect_ball(frame, prev_ball)
        if ball:
            prev_ball = ball

        # ---- Game timer ----
        if game_start is None:
            game_start = time.time()
            print(f"[GAME] Started! {GAME_DURATION}s total, returning at {GAME_DURATION - RETURN_TIME}s")

        elapsed = time.time() - game_start
        remaining = GAME_DURATION - elapsed

        # ---- Game state transitions ----
        if remaining <= 0 and game_state != "done":
            game_state = "done"
            stop_motors(ser)
            print("[GAME] Time's up — stopped")

        elif remaining <= RETURN_TIME and game_state == "seek_ball":
            game_state = "return_zone"
            stop_motors(ser)
            print(f"[GAME] {remaining:.0f}s left — returning to {TARGET_ZONE} zone!")

        # ---- State machine ----
        if game_state == "seek_ball":
            if ball is not None:
                dist, rot, *_ = drive_to_ball(ser, robot_h, ball)
                if dist is not None:
                    print(f"[BALL] dist={dist:.0f}mm rot={rot:.1f}°")
            else:
                # No ball visible — spin slowly to search
                rotate_cw(ser, 80)

        elif game_state == "return_zone":
            if robot_x is not None and robot_h is not None:
                dist, rot, *_ = drive_to_zone(ser, robot_x, robot_y, robot_h, TARGET_ZONE)
                if dist is not None and dist < ARRIVAL_DIST:
                    game_state = "turning"
                    turn_target_h = (robot_h + 180) % 360
                    stop_motors(ser)
                    print(f"[NAV] Arrived at {TARGET_ZONE} zone — turning 180°")

        elif game_state == "turning":
            if robot_h is not None:
                done = turn_around(ser, robot_h, turn_target_h)
                if done:
                    game_state = "backing"
                    back_start = time.time()
                    drive_backward(ser, BACK_SPEED)
                    print("[NAV] Turn complete — backing into wall")

        elif game_state == "backing":
            if time.time() - back_start >= BACK_DURATION:
                game_state = "done"
                stop_motors(ser)
                print("[NAV] Backed into wall — parked in zone!")
            else:
                drive_backward(ser, BACK_SPEED)

        elif game_state == "done":
            pass  # motors already stopped

        # ---- Display ----
        draw_crosshair(frame, cx, cy)
        display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Robot", display)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\n[EXIT] Shutting down...")

finally:
    stop_motors(ser)
    disconnect_arduino(ser)
    tag_cap.release()
    # ball_cap.release()
    cv2.destroyAllWindows()
    print("[EXIT] Done.")