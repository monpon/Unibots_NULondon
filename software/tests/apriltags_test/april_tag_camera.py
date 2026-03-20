import cv2
import numpy as np
from april_tag_func import (
    calculate_robot_position,
    average_robot_position,
    TAG_SIZE_MM, CAMERA_FOV_HORIZONTAL
)

# ============================================================
#  DETECTOR SETUP
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


# ============================================================
#  TAG DETECTION
# ============================================================

def detect_tags(gray, camera_center_x, camera_center_y, focal_length_pixels, frame_width):
    """
    Run ArUco detection and return a list of tag info dicts.

    Each dict contains:
        tag_id, corners, center_x, center_y,
        distance_mm, angle_x, angle_y, x_offset_mm, y_offset_mm
    """
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return [], corners, ids

    detections = []
    for i in range(len(ids)):
        tag_id = ids[i][0]
        mc = corners[i][0]

        cx = int(mc[:, 0].mean())
        cy = int(mc[:, 1].mean())

        pixel_offset_x = cx - camera_center_x
        pixel_offset_y = cy - camera_center_y

        left_x = (mc[0][0] + mc[3][0]) / 2
        right_x = (mc[1][0] + mc[2][0]) / 2
        tag_width_px = abs(right_x - left_x)

        dist_mm = (TAG_SIZE_MM * focal_length_pixels) / tag_width_px

        ang_x = CAMERA_FOV_HORIZONTAL / frame_width * pixel_offset_x
        ang_y = CAMERA_FOV_HORIZONTAL / frame_width * pixel_offset_y

        detections.append({
            "tag_id": tag_id,
            "corners": mc,
            "center_x": cx,
            "center_y": cy,
            "distance_mm": dist_mm,
            "angle_x": ang_x,
            "angle_y": ang_y,
            "x_offset_mm": dist_mm * np.tan(np.radians(ang_x)),
            "y_offset_mm": dist_mm * np.tan(np.radians(ang_y)),
        })

    return detections, corners, ids


# ============================================================
#  ROBOT POSITION FROM DETECTIONS
# ============================================================

def get_robot_position(detections):
    """
    From a list of tag detections, compute each per-tag robot position
    and return the averaged result.

    Returns
    -------
    (avg_x, avg_y, avg_heading, per_tag_positions)
    per_tag_positions is a list of (tag_id, robot_x, robot_y, heading)
    """
    estimates = []
    per_tag = []

    for d in detections:
        rx, ry, rh = calculate_robot_position(
            d["tag_id"], d["distance_mm"], d["angle_x"]
        )
        if rx is not None:
            estimates.append((rx, ry, rh))
            per_tag.append((d["tag_id"], rx, ry, rh))

    avg_x, avg_y, avg_h = average_robot_position(estimates)
    return avg_x, avg_y, avg_h, per_tag


# ============================================================
#  DRAWING HELPERS
# ============================================================

def draw_crosshair(frame, camera_center_x, camera_center_y):
    """Draw a small crosshair at the camera centre."""
    cv2.line(frame, (camera_center_x - 20, camera_center_y),
             (camera_center_x + 20, camera_center_y), (255, 255, 255), 1)
    cv2.line(frame, (camera_center_x, camera_center_y - 20),
             (camera_center_x, camera_center_y + 20), (255, 255, 255), 1)


def draw_tag_info(frame, detections, per_tag_positions):
    """Draw per-tag distance, offset, and position info on the frame."""
    pos_lookup = {tid: (rx, ry, rh) for tid, rx, ry, rh in per_tag_positions}

    for d in detections:
        tid = d["tag_id"]
        cx = d["center_x"]
        mc = d["corners"]
        dist_cm = d["distance_mm"] / 10

        cv2.circle(frame, (cx, d["center_y"]), 5, (0, 0, 255), -1)

        by = int(mc[:, 1].max()) + 20

        cv2.putText(frame, f"ID: {tid}", (cx - 40, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Dist: {dist_cm:.1f} cm", (cx - 70, by + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"X: {d['x_offset_mm']:+.1f} mm", (cx - 70, by + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Y: {d['y_offset_mm']:+.1f} mm", (cx - 70, by + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if tid in pos_lookup:
            rx, ry, rh = pos_lookup[tid]
            cv2.putText(frame, f"Pos: ({rx:.0f},{ry:.0f})", (cx - 90, by + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            cv2.putText(frame, f"Hdg: {rh:.1f} deg", (cx - 90, by + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)


def draw_robot_position(frame, avg_x, avg_y, avg_h, tag_count):
    """Draw the averaged robot position in the top-left corner."""
    cv2.putText(frame, f"Tags: {tag_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if avg_x is not None:
        cv2.putText(frame,
                    f"Robot: ({avg_x:.0f}, {avg_y:.0f}) mm  Hdg: {avg_h:.1f} deg",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame,
                    f"       ({avg_x/10:.1f}, {avg_y/10:.1f}) cm",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)


def draw_spawn_info(frame, avg_x, avg_y, avg_h, spawn_zone, in_spawn, dist_z, rot_z):
    """Draw spawn zone status on the frame."""
    if in_spawn:
        cv2.putText(frame, f"IN SPAWN [{spawn_zone.upper()}]",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif dist_z is not None:
        cv2.putText(frame,
                    f"Spawn [{spawn_zone.upper()}]: {dist_z:.0f}mm, rot {rot_z:+.1f} deg",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_no_tags(frame):
    """Draw 'no tags' message."""
    cv2.putText(frame, "No tags detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)