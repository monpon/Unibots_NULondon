import numpy as np

# ============================================================
#  INPUT YOUR VALUES HERE
# ============================================================

TAG_ID = 0              # Which tag are you looking at? (0, 1, 2, or 3)
DISTANCE_MM = 1000      # Distance from camera to tag (in mm)
ANGLE_X = -15.0           # Horizontal angle: negative=tag is left, positive=tag is right (degrees)

# ============================================================
#  ROOM AND TAG CONFIGURATION (2m x 2m room)
# ============================================================

TAGS_POSITION = {
    0: {"x": 1000, "y": 0,    "facing": 0},    # Top wall center, facing down
    1: {"x": 2000, "y": 1000, "facing": 270},  # Right wall center, facing left
    2: {"x": 1000, "y": 2000, "facing": 180},  # Bottom wall center, facing up
    3: {"x": 0,    "y": 1000, "facing": 90},   # Left wall center, facing right
}

ROOM_WIDTH = 2000
ROOM_HEIGHT = 2000

def calculate_robot_position(tag_id, distance_mm, angle_x):
    """Calculate robot position from tag detection data."""
    
    if tag_id not in TAGS_POSITION:
        print(f"Error: Tag ID {tag_id} not found!")
        return None, None, None
    
    tag = TAGS_POSITION[tag_id]
    tag_x = tag["x"]
    tag_y = tag["y"]
    tag_facing = tag["facing"]
    
    # Calculate horizontal offset
    x_offset_mm = distance_mm * np.tan(np.radians(angle_x))
    
    # Calculate forward distance (perpendicular to camera)
    y_adjustment = np.sqrt(max(0, distance_mm**2 - x_offset_mm**2))
    
    # Calculate robot heading
    robot_heading = (tag_facing + 180 - angle_x) % 360
    
    # Calculate robot position in room coordinates
    tag_facing_rad = np.radians(tag_facing)
    robot_x = tag_x + y_adjustment * np.sin(tag_facing_rad) - x_offset_mm * np.cos(tag_facing_rad)
    robot_y = tag_y + y_adjustment * np.cos(tag_facing_rad) + x_offset_mm * np.sin(tag_facing_rad)
    
    return robot_x, robot_y, robot_heading

def find_the_way(tag_id, distance_mm, angle_x, target_apriltag):

    robot_x, robot_y, robot_heading = calculate_robot_position(tag_id, distance_mm, angle_x)
    
    if robot_x is None:
        return None, None
    
    target_x = TAGS_POSITION[target_apriltag]["x"]
    target_y = TAGS_POSITION[target_apriltag]["y"]
    
    distance_to_target = np.sqrt((robot_x - target_x)**2 + (int(robot_y) - target_y)**2)
    
    delta_x = target_x - robot_x
    delta_y = target_y - int(robot_y)
    
    angle_to_target = np.degrees(np.arctan2(delta_x, delta_y)) % 360
    
    rotation_needed = angle_to_target - robot_heading
    
    if rotation_needed > 180:
        rotation_needed -= 360
    elif rotation_needed < -180:
        rotation_needed += 360
    
    print(f"=== Robot Status ===")
    print(f"Position: ({robot_x:.1f}, {robot_y:.1f}) mm")
    print(f"Current heading: {robot_heading:.1f}°")
    print(f"")
    print(f"=== Navigation to Tag {target_apriltag} ===")
    print(f"Target position: ({target_x}, {target_y}) mm")
    print(f"Distance to target: {distance_to_target:.1f} mm")
    print(f"Angle to target: {angle_to_target:.1f}°")
    print(f"Rotation needed: {rotation_needed:.1f}° {'(turn right)' if rotation_needed > 0 else '(turn left)'}")
    
    return distance_to_target, rotation_needed

find_the_way(TAG_ID, DISTANCE_MM, ANGLE_X, target_apriltag=2)