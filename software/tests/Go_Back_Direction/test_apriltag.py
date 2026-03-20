import numpy as np
import os

# ============================================================
#  INPUT YOUR VALUES HERE
# ============================================================

TAG_ID = 0              # Which tag are you looking at? (0, 1, 2, or 3)
DISTANCE_MM = 1000      # Distance from camera to tag (in mm)
ANGLE_X = -15.0           # Horizontal angle: negative=tag is left, positive=tag is right (degrees)

# ============================================================
#  ROOM AND TAG CONFIGURATION (2m x 2m room)
# ============================================================

ROOM_WIDTH = 2000   # mm
ROOM_HEIGHT = 2000  # mm

TAGS_POSITION = {
    0: {"x": 1000, "y": 0,    "facing": 0},    # Top wall center, facing down
    1: {"x": 2000, "y": 1000, "facing": 270},  # Right wall center, facing left
    2: {"x": 1000, "y": 2000, "facing": 180},  # Bottom wall center, facing up
    3: {"x": 0,    "y": 1000, "facing": 90},   # Left wall center, facing right
}

# ============================================================
#  ROBOT POSITION CALCULATION
# ============================================================

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

# ============================================================
#  BOARD DISPLAY
# ============================================================

BOARD_WIDTH = 40
BOARD_HEIGHT = 20

def create_board(robot_x=None, robot_y=None, robot_heading=None):
    """Create ASCII board visualization."""
    
    board = [[' ' for _ in range(BOARD_WIDTH + 1)] for _ in range(BOARD_HEIGHT + 1)]
    
    # Draw borders
    for x in range(BOARD_WIDTH + 1):
        board[0][x] = '-'
        board[BOARD_HEIGHT][x] = '-'
    for y in range(BOARD_HEIGHT + 1):
        board[y][0] = '|'
        board[y][BOARD_WIDTH] = '|'
    
    board[0][0] = '+'
    board[0][BOARD_WIDTH] = '+'
    board[BOARD_HEIGHT][0] = '+'
    board[BOARD_HEIGHT][BOARD_WIDTH] = '+'
    
    # Place tags
    for tid, tag in TAGS_POSITION.items():
        tx = int((tag["x"] / ROOM_WIDTH) * (BOARD_WIDTH - 2)) + 1
        ty = int((tag["y"] / ROOM_HEIGHT) * (BOARD_HEIGHT - 2)) + 1
        tx = max(1, min(BOARD_WIDTH - 1, tx))
        ty = max(1, min(BOARD_HEIGHT - 1, ty))
        board[ty][tx] = str(tid)
    
    # Place robot
    if robot_x is not None and robot_y is not None:
        rx = int((robot_x / ROOM_WIDTH) * (BOARD_WIDTH - 2)) + 1
        ry = int((robot_y / ROOM_HEIGHT) * (BOARD_HEIGHT - 2)) + 1
        rx = max(1, min(BOARD_WIDTH - 1, rx))
        ry = max(1, min(BOARD_HEIGHT - 1, ry))
        
        if robot_heading is not None:
            h = robot_heading % 360
            if 315 <= h or h < 45:
                robot_char = 'v'
            elif 45 <= h < 135:
                robot_char = '>'
            elif 135 <= h < 225:
                robot_char = '^'
            else:
                robot_char = '<'
        else:
            robot_char = 'R'
        
        board[ry][rx] = robot_char
    
    return board

def display_board(robot_x, robot_y, robot_heading, tag_id, distance_mm, angle_x):
    """Display board with all info."""
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    board = create_board(robot_x, robot_y, robot_heading)
    
    print("=" * 50)
    print("       ROBOT POSITION TESTER (2m x 2m)")
    print("=" * 50)
    print(f"  Tags: 0=Top, 1=Right, 2=Bottom, 3=Left")
    print(f"  Robot: ^v<> (shows heading direction)")
    print("-" * 50)
    
    for row in board:
        print(''.join(row))
    
    print("-" * 50)
    print("  INPUT VALUES:")
    print(f"    Tag ID:      {tag_id}")
    print(f"    Distance:    {distance_mm} mm ({distance_mm/10} cm)")
    print(f"    Angle X:     {angle_x}°")
    print("-" * 50)
    print("  CALCULATED ROBOT POSITION:")
    if robot_x is not None:
        print(f"    X Position:  {robot_x:.1f} mm ({robot_x/10:.1f} cm)")
        print(f"    Y Position:  {robot_y:.1f} mm ({robot_y/10:.1f} cm)")
        print(f"    Heading:     {robot_heading:.1f}°")
    else:
        print("    ERROR: Could not calculate position")
    print("=" * 50)

# ============================================================
#  RUN THE TEST
# ============================================================

if __name__ == "__main__":
    # Calculate robot position
    robot_x, robot_y, robot_heading = calculate_robot_position(TAG_ID, DISTANCE_MM, ANGLE_X)
    
    # Display the board
    display_board(robot_x, robot_y, robot_heading, TAG_ID, DISTANCE_MM, ANGLE_X)