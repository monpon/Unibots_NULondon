import numpy as np
import cv2
import math
import serial
import time

# ============================================================
#  ROOM AND TAG CONFIGURATION (2m x 2m room)
# ============================================================

ROOM_WIDTH = 2000   # mm
ROOM_HEIGHT = 2000  # mm
TAG_SIZE_MM = 100
CAMERA_FOV_HORIZONTAL = 117

TAG_POSITION = {
    # Top wall (y=0), facing down into room (0°)
    0:  {"x": 150,  "y": 0,    "facing": 0},
    1:  {"x": 450,  "y": 0,    "facing": 0},
    2:  {"x": 750,  "y": 0,    "facing": 0},
    3:  {"x": 1250, "y": 0,    "facing": 0},
    4:  {"x": 1550, "y": 0,    "facing": 0},
    5:  {"x": 1850, "y": 0,    "facing": 0},
    # Right wall (x=2000), facing left into room (270°)
    6:  {"x": 2000, "y": 150,  "facing": 270},
    7:  {"x": 2000, "y": 450,  "facing": 270},
    8:  {"x": 2000, "y": 750,  "facing": 270},
    9:  {"x": 2000, "y": 1250, "facing": 270},
    10: {"x": 2000, "y": 1550, "facing": 270},
    11: {"x": 2000, "y": 1850, "facing": 270},
    # Bottom wall (y=2000), facing up into room (180°)
    12: {"x": 1850, "y": 2000, "facing": 180},
    13: {"x": 1550, "y": 2000, "facing": 180},
    14: {"x": 1250, "y": 2000, "facing": 180},
    15: {"x": 750,  "y": 2000, "facing": 180},
    16: {"x": 450,  "y": 2000, "facing": 180},
    17: {"x": 150,  "y": 2000, "facing": 180},
    # Left wall (x=0), facing right into room (90°)
    18: {"x": 0, "y": 1850, "facing": 90},
    19: {"x": 0, "y": 1550, "facing": 90},
    20: {"x": 0, "y": 1250, "facing": 90},
    21: {"x": 0, "y": 750,  "facing": 90},
    22: {"x": 0, "y": 450,  "facing": 90},
    23: {"x": 0, "y": 150,  "facing": 90},
}

# ============================================================
#  ZONES
# ============================================================

ZONES = {
    "yellow": {
        "x1": 600,  "y1": 0,
        "x2": 1300, "y2": 300,
        "center": {"x": 950, "y": 150},
        "near_tags": [2, 3],
        "wall": "top",
    },
    "green": {
        "x1": 0,    "y1": 850,
        "x2": 300,  "y2": 1350,
        "center": {"x": 150, "y": 1100},
        "near_tags": [20, 21],
        "wall": "left",
    },
    "orange": {
        "x1": 1700, "y1": 750,
        "x2": 2000, "y2": 1250,
        "center": {"x": 1850, "y": 1000},
        "near_tags": [8, 9],
        "wall": "right",
    },
    "purple": {
        "x1": 600,  "y1": 1700,
        "x2": 1400, "y2": 2000,
        "center": {"x": 1000, "y": 1850},
        "near_tags": [14, 15],
        "wall": "bottom",
    },
}


# ============================================================
#  1. ROBOT POSITION  — "where am I in the room?"
# ============================================================

def calculate_robot_position(tag_id, distance_mm, angle_x):
    """
    Calculate robot (x, y, heading) from a single detected tag.

    Returns (robot_x, robot_y, robot_heading) in mm / degrees,
    or (None, None, None) if tag_id is unknown.
    """
    if tag_id not in TAG_POSITION:
        return None, None, None

    tag = TAG_POSITION[tag_id]
    tag_x, tag_y, tag_facing = tag["x"], tag["y"], tag["facing"]

    x_offset_mm = distance_mm * np.tan(np.radians(angle_x))
    forward_mm = np.sqrt(max(0, distance_mm**2 - x_offset_mm**2))

    robot_heading = (tag_facing + 180 - angle_x) % 360

    f_rad = np.radians(tag_facing)
    robot_x = tag_x + forward_mm * np.sin(f_rad) - x_offset_mm * np.cos(f_rad)
    robot_y = tag_y + forward_mm * np.cos(f_rad) + x_offset_mm * np.sin(f_rad)

    return robot_x, robot_y, robot_heading


def average_robot_position(detections):
    """
    Average multiple (x, y, heading) estimates.
    Uses circular mean for heading.

    Returns (avg_x, avg_y, avg_heading) or (None, None, None).
    """
    valid = [(x, y, h) for x, y, h in detections if x is not None]
    if not valid:
        return None, None, None

    xs, ys, hs = zip(*valid)

    sin_sum = sum(np.sin(np.radians(h)) for h in hs)
    cos_sum = sum(np.cos(np.radians(h)) for h in hs)
    avg_heading = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

    return np.mean(xs), np.mean(ys), avg_heading


# ============================================================
#  2. DIRECTION CALCULATION  — "how do I get there?"
# ============================================================

def calc_direction(robot_x, robot_y, robot_heading, target_x, target_y):
    """
    Calculate distance and rotation from robot to any target point.

    Returns (distance_mm, rotation_degrees).
    rotation: positive = turn right, negative = turn left.
    """
    dx = target_x - robot_x
    dy = target_y - robot_y

    distance = np.sqrt(dx**2 + dy**2)
    angle_to = np.degrees(np.arctan2(dx, dy)) % 360

    rotation = angle_to - robot_heading
    if rotation > 180:
        rotation -= 360
    elif rotation < -180:
        rotation += 360

    return distance, rotation


def direction_to_tag(robot_x, robot_y, robot_heading, target_tag_id):
    """
    Direction from robot to a specific tag.

    Returns (distance_mm, rotation_degrees) or (None, None).
    """
    if target_tag_id not in TAG_POSITION:
        return None, None

    t = TAG_POSITION[target_tag_id]
    return calc_direction(robot_x, robot_y, robot_heading, t["x"], t["y"])


def direction_to_point(robot_x, robot_y, robot_heading, target_x, target_y):
    """
    Direction from robot to any arbitrary (x, y) point in the room.

    Returns (distance_mm, rotation_degrees).
    """
    return calc_direction(robot_x, robot_y, robot_heading, target_x, target_y)


# ============================================================
#  3. ZONE NAVIGATION  — "where to come back / go to"
# ============================================================

def get_zone_center(zone_name):
    """Return (x, y) center of a zone, or (None, None)."""
    if zone_name not in ZONES:
        return None, None
    z = ZONES[zone_name]
    return z["center"]["x"], z["center"]["y"]


def is_robot_in_zone(robot_x, robot_y, zone_name):
    """Check if robot is inside a zone's bounding box."""
    if zone_name not in ZONES:
        return False
    z = ZONES[zone_name]
    return z["x1"] <= robot_x <= z["x2"] and z["y1"] <= robot_y <= z["y2"]


def get_robot_zone(robot_x, robot_y):
    """Return name of zone the robot is in, or None."""
    for name in ZONES:
        if is_robot_in_zone(robot_x, robot_y, name):
            return name
    return None


def direction_to_zone(robot_x, robot_y, robot_heading, zone_name):
    """
    Direction from robot to a zone's center.

    Returns (distance_mm, rotation_degrees) or (None, None).
    """
    tx, ty = get_zone_center(zone_name)
    if tx is None:
        return None, None
    return calc_direction(robot_x, robot_y, robot_heading, tx, ty)


# ============================================================
#  4. BALL DETECTION  — "where is the ball?"
# ============================================================

# ---- Ball camera config (separate from tag camera) ----
BALL_CAM_HEIGHT_MM    = 150       # camera height above ground
BALL_CAM_TILT_DEG     = 15        # downward tilt angle
BALL_CAM_HFOV         = 90        # horizontal field of view
BALL_CAM_VFOV         = 50.625    # vertical field of view
BALL_CAM_RES_W        = 320
BALL_CAM_RES_H        = 240

# ---- HSV colour range for the ball ----
BALL_HSV_LOWER = np.array([150, 0, 50])
BALL_HSV_UPPER = np.array([200, 200, 150])

# ---- Hough circle detector params ----
BALL_HOUGH_DP         = 1
BALL_HOUGH_MIN_DIST   = 300
BALL_HOUGH_PARAM1     = 50
BALL_HOUGH_PARAM2     = 20
BALL_HOUGH_MIN_RADIUS = 5
BALL_HOUGH_MAX_RADIUS = 100


def detect_ball(frame, prev_circle=None):
    """
    Detect a ball in a BGR frame using colour mask + Hough circles.

    Parameters
    ----------
    frame       : BGR image (should be BALL_CAM_RES_W x BALL_CAM_RES_H)
    prev_circle : previous (x, y, r) to prefer closest match

    Returns
    -------
    circle : (x, y, radius) in pixels, or None if not found
    mask   : the colour mask (useful for debug display)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        BALL_HOUGH_DP, BALL_HOUGH_MIN_DIST,
        param1=BALL_HOUGH_PARAM1, param2=BALL_HOUGH_PARAM2,
        minRadius=BALL_HOUGH_MIN_RADIUS, maxRadius=BALL_HOUGH_MAX_RADIUS,
    )

    if circles is None:
        return None, mask

    detected = np.asarray(np.uint16(np.around(circles[0])))
    chosen = None

    for c in detected:
        if chosen is None:
            chosen = c
        elif prev_circle is not None:
            d_new = (c[0] - prev_circle[0])**2 + (c[1] - prev_circle[1])**2
            d_old = (chosen[0] - prev_circle[0])**2 + (chosen[1] - prev_circle[1])**2
            if d_new < d_old:
                chosen = c

    if chosen is not None:
        return (int(chosen[0]), int(chosen[1]), int(chosen[2])), mask

    return None, mask


def ball_position_relative(circle):
    """
    Convert ball pixel position to real-world distance relative to the robot.

    Parameters
    ----------
    circle : (x, y, radius) in pixels from detect_ball()

    Returns
    -------
    (forward_mm, lateral_mm) — distance ahead and to the side
        forward  : + means in front of robot
        lateral  : + means to the right, - means to the left
    Or (None, None) if the ball is behind/above the camera horizon.
    """
    if circle is None:
        return None, None

    cx, cy, _ = circle
    center_x = BALL_CAM_RES_W / 2
    center_y = BALL_CAM_RES_H / 2

    x_angle = (cx - center_x) * (BALL_CAM_HFOV / BALL_CAM_RES_W)
    y_angle = (cy - center_y) * (BALL_CAM_VFOV / BALL_CAM_RES_H)

    tilt_rad = math.radians(BALL_CAM_TILT_DEG)
    effective_down = tilt_rad + math.radians(y_angle)

    if effective_down <= 0:
        return None, None

    forward_mm  = BALL_CAM_HEIGHT_MM / math.tan(effective_down)
    lateral_mm  = forward_mm * math.tan(math.radians(x_angle))

    return forward_mm, lateral_mm


def ball_position_room(robot_x, robot_y, robot_heading, circle):
    """
    Convert ball position from camera-relative to absolute room coordinates.

    Parameters
    ----------
    robot_x, robot_y   : robot position in room (mm), from AprilTags
    robot_heading       : robot heading (degrees), 0 = facing top wall
    circle              : (x, y, radius) from detect_ball()

    Returns
    -------
    (ball_x, ball_y) in room coordinates (mm), or (None, None).
    """
    forward_mm, lateral_mm = ball_position_relative(circle)
    if forward_mm is None:
        return None, None

    heading_rad = np.radians(robot_heading)

    ball_x = robot_x + forward_mm * np.sin(heading_rad) + lateral_mm * np.cos(heading_rad)
    ball_y = robot_y + forward_mm * np.cos(heading_rad) - lateral_mm * np.sin(heading_rad)

    return ball_x, ball_y


def direction_to_ball(robot_heading, circle):
    """
    Quick direction to ball using camera-relative position only.
    No AprilTags needed — just drive toward what the camera sees.

    Parameters
    ----------
    robot_heading : current heading (degrees), only used for return context
    circle        : (x, y, radius) from detect_ball()

    Returns
    -------
    (distance_mm, rotation_degrees) to the ball
        rotation: + turn right, - turn left
    Or (None, None) if ball not detected.
    """
    forward_mm, lateral_mm = ball_position_relative(circle)
    if forward_mm is None:
        return None, None

    distance = math.sqrt(forward_mm**2 + lateral_mm**2) # type: ignore
    rotation = math.degrees(math.atan2(lateral_mm, forward_mm)) # type: ignore

    return distance, rotation


# ============================================================
#  5. UTILITIES
# ============================================================

def is_position_in_room(x, y):
    """Check if a position is within the room."""
    return 0 <= x <= ROOM_WIDTH and 0 <= y <= ROOM_HEIGHT


def clamp_to_room(x, y):
    """Clamp a position to room boundaries."""
    return max(0, min(ROOM_WIDTH, x)), max(0, min(ROOM_HEIGHT, y))


def distance_between_tags(tag_id_a, tag_id_b):
    """Straight-line distance between two tags (mm), or None."""
    if tag_id_a not in TAG_POSITION or tag_id_b not in TAG_POSITION:
        return None
    a, b = TAG_POSITION[tag_id_a], TAG_POSITION[tag_id_b]
    return np.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2)


def nearest_tag_to_point(x, y):
    """Return (tag_id, distance) of closest tag to a point."""
    best_id, best_dist = None, float("inf")
    for tid, tag in TAG_POSITION.items():
        d = np.sqrt((tag["x"] - x)**2 + (tag["y"] - y)**2)
        if d < best_dist:
            best_dist = d
            best_id = tid
    return best_id, best_dist


# ============================================================
#  6. ARDUINO / MOTOR CONTROL
# ============================================================

# ---- Motor tuning ----
MAX_SPEED  = 220      # PWM cap
MIN_SPEED  = 60       # deadband — below this motors stall

# ---- Proportional gains ----
KP_FORWARD = 0.6      # power per mm of distance
KP_STRAFE  = 0.6      # power per mm of lateral error
KP_ROTATE  = 1.8      # power per degree of heading error


def connect_arduino(port="/dev/ttyUSB0", baud=115200, timeout=0.05):
    """
    Open serial connection to Arduino. Waits for boot and returns
    the serial object, or None on failure.
    """
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)
        while ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(f"[Arduino] {line}")
        print("[Arduino] Connected")
        return ser
    except serial.SerialException as e:
        print(f"[Arduino] Connection failed: {e}")
        return None


def disconnect_arduino(ser):
    """Stop motors and close the serial port."""
    if ser and ser.is_open:
        ser.write(b"S\n")
        ser.readline()
        ser.close()
        print("[Arduino] Disconnected")


# ---- Raw wheel command ----

def set_wheels(ser, fl, fr, rl, rr):
    """
    Send raw wheel powers to Arduino.
    Values clamped to [-255, 255].
    Returns the ack string.
    """
    if ser is None:
        return None
    cl = lambda v: max(-255, min(255, int(v)))
    cmd = f"M {cl(fl)} {cl(fr)} {cl(rl)} {cl(rr)}\n"
    ser.write(cmd.encode())
    return ser.readline().decode(errors="ignore").strip()


def stop_motors(ser):
    """Hard stop all wheels."""
    if ser is None:
        return None
    ser.write(b"S\n")
    return ser.readline().decode(errors="ignore").strip()


def query_wheels(ser):
    """Ask Arduino for current wheel powers."""
    if ser is None:
        return None
    ser.write(b"P\n")
    return ser.readline().decode(errors="ignore").strip()


# ---- Mecanum kinematics ----

def mecanum_mix(forward, strafe, rotate):
    """
    Convert (forward, strafe, rotate) into four wheel powers.

    forward : + forward  / - backward
    strafe  : + right    / - left
    rotate  : + clockwise / - counter-clockwise

    Returns (fl, fr, rl, rr) each in [-MAX_SPEED, MAX_SPEED].
    """
    fl = forward - strafe + rotate
    fr = forward + strafe - rotate
    rl = forward + strafe + rotate
    rr = forward - strafe - rotate

    peak = max(abs(fl), abs(fr), abs(rl), abs(rr), 1)
    if peak > MAX_SPEED:
        s = MAX_SPEED / peak
        fl *= s; fr *= s; rl *= s; rr *= s

    db = lambda v: 0 if abs(v) < MIN_SPEED else v
    return int(db(fl)), int(db(fr)), int(db(rl)), int(db(rr))


# ---- High-level drive commands ----

def drive_forward(ser, speed=150):
    """All wheels forward."""
    fl, fr, rl, rr = mecanum_mix(speed, 0, 0)
    return set_wheels(ser, fl, fr, rl, rr)


def drive_backward(ser, speed=150):
    """All wheels backward."""
    fl, fr, rl, rr = mecanum_mix(-speed, 0, 0)
    return set_wheels(ser, fl, fr, rl, rr)


def strafe_left(ser, speed=150):
    """Sideways left."""
    fl, fr, rl, rr = mecanum_mix(0, -speed, 0)
    return set_wheels(ser, fl, fr, rl, rr)


def strafe_right(ser, speed=150):
    """Sideways right."""
    fl, fr, rl, rr = mecanum_mix(0, speed, 0)
    return set_wheels(ser, fl, fr, rl, rr)


def rotate_cw(ser, speed=120):
    """Spin clockwise in place."""
    fl, fr, rl, rr = mecanum_mix(0, 0, speed)
    return set_wheels(ser, fl, fr, rl, rr)


def rotate_ccw(ser, speed=120):
    """Spin counter-clockwise in place."""
    fl, fr, rl, rr = mecanum_mix(0, 0, -speed)
    return set_wheels(ser, fl, fr, rl, rr)


def drive_diagonal(ser, forward=150, strafe=150):
    """Diagonal movement — combine forward + strafe."""
    fl, fr, rl, rr = mecanum_mix(forward, strafe, 0)
    return set_wheels(ser, fl, fr, rl, rr)


# ---- Navigation drive: go toward a target ----

def drive_toward(ser, distance_mm, rotation_deg, blend=True):
    """
    Drive toward a target given distance and rotation.
    Uses the output of calc_direction() / direction_to_zone() / direction_to_ball().

    Parameters
    ----------
    ser          : Arduino serial object
    distance_mm  : how far away the target is
    rotation_deg : how much to turn (+ right, - left)
    blend        : True = simultaneous fwd+strafe+rotate
                   False = rotate first, then drive straight

    Returns (fl, fr, rl, rr) that were sent.
    """
    if ser is None or distance_mm is None:
        return None

    if blend:
        rad = np.radians(rotation_deg)
        forward = KP_FORWARD * distance_mm * np.cos(rad)
        strafe  = KP_STRAFE  * distance_mm * np.sin(rad)
        rotate  = KP_ROTATE  * rotation_deg
    else:
        if abs(rotation_deg) > 8:
            forward, strafe = 0, 0
            rotate = KP_ROTATE * rotation_deg
        else:
            forward = KP_FORWARD * distance_mm
            strafe  = 0
            rotate  = KP_ROTATE * rotation_deg

    fl, fr, rl, rr = mecanum_mix(forward, strafe, rotate)
    set_wheels(ser, fl, fr, rl, rr)
    return fl, fr, rl, rr


def drive_to_zone(ser, robot_x, robot_y, robot_heading, zone_name, blend=True):
    """
    One-call: calculate direction to zone and drive toward it.

    Returns (distance, rotation, fl, fr, rl, rr) or None values if zone unknown.
    """
    dist, rot = direction_to_zone(robot_x, robot_y, robot_heading, zone_name)
    if dist is None:
        stop_motors(ser)
        return None, None, None, None, None, None
    wheels = drive_toward(ser, dist, rot, blend)
    if wheels:
        return dist, rot, *wheels
    return dist, rot, None, None, None, None


def turn_around(ser, robot_heading, target_heading, speed=120, tolerance=5):
    """
    Rotate in place toward a target heading.
    Call this every loop iteration — it returns True when done.

    Parameters
    ----------
    ser            : Arduino serial object
    robot_heading  : current heading (degrees), from AprilTags
    target_heading : desired heading (degrees)
    speed          : rotation PWM speed
    tolerance      : degrees of error to accept as "done"

    Returns
    -------
    True if heading is within tolerance (turn complete), False otherwise.
    """
    if robot_heading is None:
        stop_motors(ser)
        return False

    error = target_heading - robot_heading
    if error > 180:
        error -= 360
    elif error < -180:
        error += 360

    if abs(error) <= tolerance:
        stop_motors(ser)
        return True

    if error > 0:
        rotate_cw(ser, speed)
    else:
        rotate_ccw(ser, speed)
    return False


def drive_to_tag(ser, robot_x, robot_y, robot_heading, tag_id, blend=True):
    """
    One-call: calculate direction to tag and drive toward it.

    Returns (distance, rotation, fl, fr, rl, rr) or None values.
    """
    dist, rot = direction_to_tag(robot_x, robot_y, robot_heading, tag_id)
    if dist is None:
        stop_motors(ser)
        return None, None, None, None, None, None
    wheels = drive_toward(ser, dist, rot, blend)
    if wheels:
        return dist, rot, *wheels
    return dist, rot, None, None, None, None


def drive_to_ball(ser, robot_heading, circle, blend=True):
    """
    One-call: detect ball direction and drive toward it.

    Parameters
    ----------
    ser           : Arduino serial object
    robot_heading : current heading (degrees)
    circle        : (x, y, radius) from detect_ball()

    Returns (distance, rotation, fl, fr, rl, rr) or None values.
    """
    dist, rot = direction_to_ball(robot_heading, circle)
    if dist is None:
        stop_motors(ser)
        return None, None, None, None, None, None
    wheels = drive_toward(ser, dist, rot, blend)
    if wheels:
        return dist, rot, *wheels
    return dist, rot, None, None, None, None


def drive_to_point(ser, robot_x, robot_y, robot_heading, target_x, target_y, blend=True):
    """
    One-call: drive toward any arbitrary (x, y) point in the room.

    Returns (distance, rotation, fl, fr, rl, rr) or None values.
    """
    dist, rot = direction_to_point(robot_x, robot_y, robot_heading, target_x, target_y)
    if dist is None:
        stop_motors(ser)
        return None, None, None, None, None, None
    wheels = drive_toward(ser, dist, rot, blend)
    if wheels:
        return dist, rot, *wheels
    return dist, rot, None, None, None, None