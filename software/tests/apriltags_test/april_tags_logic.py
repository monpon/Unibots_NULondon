import cv2
import numpy as np
from april_tag_func import (
    find_the_way_from_position,
    find_way_to_zone_from_position,
    get_zone_center,
    is_robot_in_zone,
    CAMERA_FOV_HORIZONTAL,
)
from april_tag_camera import (
    detect_tags,
    get_robot_position,
    draw_crosshair,
    draw_tag_info,
    draw_robot_position,
    draw_spawn_info,
    draw_no_tags,
)

# ============================================================
#  SETTINGS — change these!
# ============================================================

SPAWN_ZONE = "purple"       # "yellow", "green", "orange", "purple"
CAMERA_INDEX = 1
CAPTURE_WIDTH = 2560
CAPTURE_HEIGHT = 1440
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# ============================================================
#  CAMERA INIT
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

ret, test_frame = cap.read()
if ret:
    frame_height, frame_width = test_frame.shape[:2]
else:
    frame_width, frame_height = CAPTURE_WIDTH, CAPTURE_HEIGHT

camera_center_x = frame_width // 2
camera_center_y = frame_height // 2
focal_length_pixels = (frame_width / 2) / np.tan(np.radians(CAMERA_FOV_HORIZONTAL / 2))

spawn_cx, spawn_cy = get_zone_center(SPAWN_ZONE)
print(f"Resolution: {frame_width}x{frame_height} | Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print(f"Spawn zone: {SPAWN_ZONE.upper()} — center at ({spawn_cx}, {spawn_cy}) mm")
print("Press 'q' to quit\n")

# ============================================================
#  MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    draw_crosshair(frame, camera_center_x, camera_center_y)

    # 1. Detect all visible tags
    detections, raw_corners, raw_ids = detect_tags(
        gray, camera_center_x, camera_center_y, focal_length_pixels, frame_width
    )

    if detections:
        cv2.aruco.drawDetectedMarkers(frame, raw_corners, raw_ids, (0, 255, 0))

        # 2. Calculate robot position
        avg_x, avg_y, avg_h, per_tag = get_robot_position(detections)

        # 3. Console log
        for tid, rx, ry, rh in per_tag:
            print(f"Tag {tid:2d} | Pos: ({rx:.0f}, {ry:.0f}) mm | Hdg: {rh:.1f}°")
        if avg_x is not None:
            print(f"  >> AVG | Pos: ({avg_x:.0f}, {avg_y:.0f}) mm | Hdg: {avg_h:.1f}°")

        # 4. Draw tag + position overlays
        draw_tag_info(frame, detections, per_tag)
        draw_robot_position(frame, avg_x, avg_y, avg_h, len(detections))

        # 5. Spawn zone status
        if avg_x is not None:
            in_spawn = is_robot_in_zone(avg_x, avg_y, SPAWN_ZONE)
            dist_z, rot_z = find_way_to_zone_from_position(avg_x, avg_y, avg_h, SPAWN_ZONE)
            draw_spawn_info(frame, avg_x, avg_y, avg_h, SPAWN_ZONE, in_spawn, dist_z, rot_z)

        # -------------------------------------------------------
        #  TODO: Add your navigation logic here!
        #  dist, rot = find_the_way_from_position(avg_x, avg_y, avg_h, target_apriltag=12)
        #  dist, rot = find_way_to_zone_from_position(avg_x, avg_y, avg_h, "yellow")
        # -------------------------------------------------------

    else:
        draw_no_tags(frame)

    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                               interpolation=cv2.INTER_LINEAR)
    cv2.imshow('AprilTag Detection', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()