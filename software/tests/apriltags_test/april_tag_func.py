import numpy as np

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
#  ZONES / NETS (bounding boxes in mm)
#  x1,y1 = top-left corner    x2,y2 = bottom-right corner
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
#  POSITION CALCULATION
# ============================================================

def calculate_robot_position(tag_id, distance_mm, angle_x):
    """
    Calculate robot (x, y) and heading from a single detected tag.

    Parameters
    ----------
    tag_id      : int   – detected tag ID
    distance_mm : float – straight-line distance to the tag (mm)
    angle_x     : float – horizontal angle from camera centre (degrees)
                          negative = left, positive = right

    Returns
    -------
    (robot_x, robot_y, robot_heading) in mm / degrees, or (None, None, None)
    """
    if tag_id not in TAG_POSITION:
        print(f"Error: Tag ID {tag_id} not found!")
        return None, None, None

    tag = TAG_POSITION[tag_id]
    tag_x = tag["x"]
    tag_y = tag["y"]
    tag_facing = tag["facing"]

    x_offset_mm = distance_mm * np.tan(np.radians(angle_x))
    forward_mm = np.sqrt(max(0, distance_mm**2 - x_offset_mm**2))

    robot_heading = (tag_facing + 180 - angle_x) % 360

    f_rad = np.radians(tag_facing)
    robot_x = tag_x + forward_mm * np.sin(f_rad) - x_offset_mm * np.cos(f_rad)
    robot_y = tag_y + forward_mm * np.cos(f_rad) + x_offset_mm * np.sin(f_rad)

    return robot_x, robot_y, robot_heading


def average_robot_position(detections):
    """
    Average multiple (x, y, heading) estimates for a stable result.
    Uses circular mean for heading to handle 359°/1° wrap.

    Parameters
    ----------
    detections : list of (robot_x, robot_y, robot_heading) tuples

    Returns
    -------
    (avg_x, avg_y, avg_heading) or (None, None, None)
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
#  NAVIGATION — TO A TAG
# ============================================================

def _calc_navigation(robot_x, robot_y, robot_heading, target_x, target_y):
    """
    Internal helper: distance and rotation from robot to any target point.

    Returns
    -------
    (distance, rotation_needed) in mm / degrees
    rotation: positive = turn right, negative = turn left
    """
    delta_x = target_x - robot_x
    delta_y = target_y - robot_y

    distance = np.sqrt(delta_x**2 + delta_y**2)
    angle_to = np.degrees(np.arctan2(delta_x, delta_y)) % 360

    rotation = angle_to - robot_heading
    if rotation > 180:
        rotation -= 360
    elif rotation < -180:
        rotation += 360

    return distance, rotation


def find_the_way(tag_id, distance_mm, angle_x, target_apriltag):
    """
    Navigate to a target tag from a single tag detection.

    Returns
    -------
    (distance_to_target, rotation_needed) in mm / degrees, or (None, None)
    """
    robot_x, robot_y, robot_heading = calculate_robot_position(tag_id, distance_mm, angle_x)
    if robot_x is None:
        return None, None

    if target_apriltag not in TAG_POSITION:
        print(f"Error: Target tag ID {target_apriltag} not found!")
        return None, None

    target_x = TAG_POSITION[target_apriltag]["x"]
    target_y = TAG_POSITION[target_apriltag]["y"]

    return _calc_navigation(robot_x, robot_y, robot_heading, target_x, target_y)


def find_the_way_from_position(robot_x, robot_y, robot_heading, target_apriltag):
    """
    Navigate to a target tag from a pre-computed (averaged) position.

    Returns
    -------
    (distance_to_target, rotation_needed) in mm / degrees, or (None, None)
    """
    if target_apriltag not in TAG_POSITION:
        print(f"Error: Target tag ID {target_apriltag} not found!")
        return None, None

    target_x = TAG_POSITION[target_apriltag]["x"]
    target_y = TAG_POSITION[target_apriltag]["y"]

    return _calc_navigation(robot_x, robot_y, robot_heading, target_x, target_y)


# ============================================================
#  NAVIGATION — TO A ZONE
# ============================================================

def get_zone_center(zone_name):
    """
    Return (x, y) center of a named zone.

    Returns
    -------
    (x, y) in mm, or (None, None)
    """
    if zone_name not in ZONES:
        print(f"Error: Zone '{zone_name}' not found! Options: {list(ZONES.keys())}")
        return None, None
    z = ZONES[zone_name]
    return z["center"]["x"], z["center"]["y"]


def is_robot_in_zone(robot_x, robot_y, zone_name):
    """Check if the robot is inside a given zone's bounding box."""
    if zone_name not in ZONES:
        return False
    z = ZONES[zone_name]
    return z["x1"] <= robot_x <= z["x2"] and z["y1"] <= robot_y <= z["y2"]


def get_robot_zone(robot_x, robot_y):
    """
    Return the name of the zone the robot is currently in, or None.
    """
    for name in ZONES:
        if is_robot_in_zone(robot_x, robot_y, name):
            return name
    return None


def find_way_to_zone(tag_id, distance_mm, angle_x, zone_name):
    """
    Navigate to a zone center from a single tag detection.

    Returns
    -------
    (distance_to_zone, rotation_needed) in mm / degrees, or (None, None)
    """
    robot_x, robot_y, robot_heading = calculate_robot_position(tag_id, distance_mm, angle_x)
    if robot_x is None:
        return None, None

    target_x, target_y = get_zone_center(zone_name)
    if target_x is None:
        return None, None

    return _calc_navigation(robot_x, robot_y, robot_heading, target_x, target_y)


def find_way_to_zone_from_position(robot_x, robot_y, robot_heading, zone_name):
    """
    Navigate to a zone center from a pre-computed (averaged) position.

    Returns
    -------
    (distance_to_zone, rotation_needed) in mm / degrees, or (None, None)
    """
    target_x, target_y = get_zone_center(zone_name)
    if target_x is None:
        return None, None

    return _calc_navigation(robot_x, robot_y, robot_heading, target_x, target_y)


# ============================================================
#  UTILITIES
# ============================================================

def is_position_in_room(x, y):
    """Check if a position is within the room boundaries."""
    return 0 <= x <= ROOM_WIDTH and 0 <= y <= ROOM_HEIGHT


def clamp_to_room(x, y):
    """Clamp a position to stay within room boundaries."""
    return (
        max(0, min(ROOM_WIDTH, x)),
        max(0, min(ROOM_HEIGHT, y)),
    )


def distance_between_tags(tag_id_a, tag_id_b):
    """Return straight-line distance between two tags (mm), or None."""
    if tag_id_a not in TAG_POSITION or tag_id_b not in TAG_POSITION:
        return None
    a = TAG_POSITION[tag_id_a]
    b = TAG_POSITION[tag_id_b]
    return np.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2)


def nearest_tag_to_point(x, y):
    """Return the tag ID closest to an (x, y) point."""
    best_id = None
    best_dist = float("inf")
    for tid, tag in TAG_POSITION.items():
        d = np.sqrt((tag["x"] - x)**2 + (tag["y"] - y)**2)
        if d < best_dist:
            best_dist = d
            best_id = tid
    return best_id, best_dist