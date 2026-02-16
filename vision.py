"""
Vision Module for Unibots Mecanum Robot
========================================
Detects ping-pong balls, ball bearings, and other robots using
a Raspberry Pi camera and OpenCV.

Challenges this module addresses:
    - White ping-pong balls on a white floor (shadow/edge detection)
    - Tiny 20mm metallic ball bearings (specular highlights)
    - Other robots for collision avoidance
    - Converting pixel positions to arena coordinates using AprilTags

Dependencies:
    pip install opencv-python numpy pupil-apriltags

Hardware:
    - Raspberry Pi Camera Module (v2 or v3) or USB webcam
    - Mounted looking forward and/or downward

Usage:
    vision = VisionSystem(camera_index=0)
    vision.calibrate()  # One-time calibration
    while True:
        detections = vision.detect_all()
        balls = detections["balls"]
        robots = detections["robots"]
"""

import cv2
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

try:
    from pupil_apriltags import Detector as AprilTagDetector
    HAS_APRILTAGS = True
except ImportError:
    HAS_APRILTAGS = False
    print("Warning: pupil-apriltags not installed. AprilTag detection disabled.")
    print("Install with: pip install pupil-apriltags")


# ============================================================
# Data Structures
# ============================================================

class DetectedObjectType(Enum):
    PING_PONG = "ping_pong"
    BALL_BEARING = "ball_bearing"
    ROBOT = "robot"


@dataclass
class DetectedObject:
    """A detected object in the arena."""
    obj_type: DetectedObjectType
    pixel_x: int          # Pixel position in image
    pixel_y: int
    world_x: float = 0.0  # Arena position in mm (after transform)
    world_y: float = 0.0
    confidence: float = 0.0
    radius_px: int = 0     # Apparent size in pixels


@dataclass
class CameraCalibration:
    """Camera intrinsic parameters."""
    fx: float = 600.0   # Focal length x (pixels)
    fy: float = 600.0   # Focal length y (pixels)
    cx: float = 320.0   # Principal point x
    cy: float = 240.0   # Principal point y
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(5)
    )


# ============================================================
# Ball Detection
# ============================================================

class BallDetector:
    """
    Detects ping-pong balls and ball bearings in the arena.

    Strategy for white balls on white floor:
        - Use edge detection (Canny) to find circular outlines
        - Use shadow detection (balls cast shadows on the floor)
        - Combine with Hough circle detection
        - Filter by size to distinguish ping-pong (40mm) vs bearing (20mm)

    Strategy for ball bearings:
        - Metallic surfaces create specular highlights (bright spots)
        - They appear darker/greyer than the white floor
        - Much smaller than ping-pong balls
    """

    def __init__(self):
        # --- Ping-pong ball detection parameters ---
        # Expected radius range in pixels (adjust based on your camera
        # height and resolution). These are starting values for a camera
        # ~200mm above the floor at 640x480.
        self.pp_min_radius = 12
        self.pp_max_radius = 50

        # --- Ball bearing detection parameters ---
        self.bb_min_radius = 5
        self.bb_max_radius = 18

        # --- Edge detection ---
        self.canny_low = 30
        self.canny_high = 100

        # --- Shadow detection for white balls on white floor ---
        # Shadows appear as slightly darker regions around the ball
        self.shadow_threshold = 200  # Below this = potential shadow/ball edge

        # --- Specular highlight detection for ball bearings ---
        self.highlight_threshold = 240  # Above this = bright spot (metallic)

        # Background subtractor for detecting anything that isn't floor
        self.bg_subtractor = None
        self.floor_sample = None

    def calibrate_floor(self, frame: np.ndarray):
        """
        Sample the arena floor color to improve detection.
        Call this at the start of a match with the camera pointing
        at an empty area of floor.
        """
        # Take center region as floor sample
        h, w = frame.shape[:2]
        roi = frame[h//3:2*h//3, w//3:2*w//3]

        # Store floor color statistics
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.floor_mean = np.mean(hsv_roi, axis=(0, 1))
        self.floor_std = np.std(hsv_roi, axis=(0, 1))

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.floor_gray_mean = np.mean(gray_roi)
        self.floor_gray_std = np.std(gray_roi)

        print(f"Floor calibrated: gray mean={self.floor_gray_mean:.1f}, "
              f"std={self.floor_gray_std:.1f}")

    def detect(self, frame: np.ndarray,
               debug: bool = False) -> List[DetectedObject]:
        """
        Detect all balls in the frame.

        Args:
            frame: BGR image from camera
            debug: If True, display debug visualization

        Returns:
            List of DetectedObject for each ball found
        """
        detections = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- Method 1: Edge-based circle detection ---
        edge_detections = self._detect_circles_edge(gray, frame)
        detections.extend(edge_detections)

        # --- Method 2: Shadow-based detection (for white balls) ---
        shadow_detections = self._detect_by_shadow(gray, frame)
        detections.extend(shadow_detections)

        # --- Method 3: Specular highlight (for ball bearings) ---
        highlight_detections = self._detect_highlights(gray, frame)
        detections.extend(highlight_detections)

        # --- Merge overlapping detections ---
        detections = self._merge_detections(detections)

        if debug:
            self._draw_debug(frame, detections)

        return detections

    def _detect_circles_edge(self, gray: np.ndarray,
                              color: np.ndarray) -> List[DetectedObject]:
        """Use Canny edges + Hough circles to find balls."""
        detections = []

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Hough circle detection
        # dp=1.2: resolution ratio, minDist: min distance between circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=self.canny_high,
            param2=30,
            minRadius=self.bb_min_radius,
            maxRadius=self.pp_max_radius
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (cx, cy, r) in circles:
                # Classify by size
                if self.pp_min_radius <= r <= self.pp_max_radius:
                    obj_type = DetectedObjectType.PING_PONG
                    confidence = 0.6
                elif self.bb_min_radius <= r <= self.bb_max_radius:
                    obj_type = DetectedObjectType.BALL_BEARING
                    confidence = 0.5
                else:
                    continue

                # Verify: check if the region looks like a ball
                confidence = self._verify_ball(
                    gray, color, cx, cy, r, obj_type, confidence)

                if confidence > 0.3:
                    detections.append(DetectedObject(
                        obj_type=obj_type,
                        pixel_x=cx, pixel_y=cy,
                        confidence=confidence,
                        radius_px=r
                    ))

        return detections

    def _detect_by_shadow(self, gray: np.ndarray,
                           color: np.ndarray) -> List[DetectedObject]:
        """
        Detect white balls by their shadow on the white floor.

        White ping-pong balls on white floor are hard to see by color,
        but they cast a shadow and have a slight 3D shading gradient.
        We look for small dark crescents/rings below bright regions.
        """
        detections = []

        # Adaptive threshold highlights areas darker than local neighborhood
        # This catches shadows even on a white floor
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 8
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

        # Find contours of shadow regions
        contours, _ = cv2.findContours(
            adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by area (ping-pong ball shadow is roughly circular)
            if area < 100 or area > 5000:
                continue

            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)

            if circularity < 0.5:
                continue  # Not circular enough

            # Get bounding circle
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius = int(cx), int(cy), int(radius)

            # Classify by size
            if self.pp_min_radius <= radius <= self.pp_max_radius:
                obj_type = DetectedObjectType.PING_PONG
                confidence = 0.4 + 0.3 * circularity
            elif self.bb_min_radius <= radius <= self.bb_max_radius:
                obj_type = DetectedObjectType.BALL_BEARING
                confidence = 0.3 + 0.3 * circularity
            else:
                continue

            detections.append(DetectedObject(
                obj_type=obj_type,
                pixel_x=cx, pixel_y=cy,
                confidence=confidence,
                radius_px=radius
            ))

        return detections

    def _detect_highlights(self, gray: np.ndarray,
                            color: np.ndarray) -> List[DetectedObject]:
        """
        Detect ball bearings by their metallic specular highlights.

        Steel ball bearings are shiny and create bright spots (reflections)
        surrounded by a darker metallic body. This is distinctive on a
        white floor.
        """
        detections = []

        # Find very bright spots (specular highlights)
        _, bright_mask = cv2.threshold(
            gray, self.highlight_threshold, 255, cv2.THRESH_BINARY
        )

        # Also find darker-than-floor regions (the metallic body)
        if self.floor_gray_mean is not None:
            threshold = self.floor_gray_mean - 2 * self.floor_gray_std
            _, dark_mask = cv2.threshold(
                gray, int(max(threshold, 80)), 255, cv2.THRESH_BINARY_INV
            )
        else:
            _, dark_mask = cv2.threshold(
                gray, 160, 255, cv2.THRESH_BINARY_INV
            )

        # Look for small bright spots near small dark regions
        # Dilate bright spots to connect with nearby dark body
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bright_dilated = cv2.dilate(bright_mask, kernel)

        # Combine: regions that are both bright-nearby AND darker
        combined = cv2.bitwise_and(bright_dilated, dark_mask)

        # Clean up
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 800:  # Ball bearings are small
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy, radius = int(cx), int(cy), int(radius)

            if self.bb_min_radius <= radius <= self.bb_max_radius:
                # Check for the characteristic bright spot
                roi_mask = np.zeros_like(gray)
                cv2.circle(roi_mask, (cx, cy), radius + 3, 255, -1)
                max_val = np.max(gray[roi_mask > 0]) if np.any(roi_mask > 0) else 0

                if max_val > self.highlight_threshold:
                    confidence = 0.6
                else:
                    confidence = 0.3

                detections.append(DetectedObject(
                    obj_type=DetectedObjectType.BALL_BEARING,
                    pixel_x=cx, pixel_y=cy,
                    confidence=confidence,
                    radius_px=radius
                ))

        return detections

    def _verify_ball(self, gray, color, cx, cy, r, obj_type,
                     base_confidence) -> float:
        """
        Additional verification checks on a candidate ball detection.
        Returns adjusted confidence.
        """
        h, w = gray.shape
        confidence = base_confidence

        # Check bounds
        if cx - r < 0 or cx + r >= w or cy - r < 0 or cy + r >= h:
            return confidence * 0.5  # Partially off-screen

        # Create circular ROI mask
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        # Get pixel values inside the circle
        pixels = gray[mask > 0]
        if len(pixels) == 0:
            return 0.0

        mean_val = np.mean(pixels)
        std_val = np.std(pixels)

        if obj_type == DetectedObjectType.PING_PONG:
            # Ping-pong balls are white/bright with some shading
            if mean_val > 180:
                confidence += 0.15
            # Should have some gradient (3D shading), not flat
            if 5 < std_val < 50:
                confidence += 0.1

        elif obj_type == DetectedObjectType.BALL_BEARING:
            # Ball bearings are metallic grey, not white
            if 80 < mean_val < 200:
                confidence += 0.15
            # High contrast from specular highlight
            if std_val > 20:
                confidence += 0.1

        return min(confidence, 1.0)

    def _merge_detections(self, detections: List[DetectedObject],
                           distance_threshold: int = 25) -> List[DetectedObject]:
        """Merge overlapping detections, keeping highest confidence."""
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        merged = []
        used = [False] * len(detections)

        for i, det in enumerate(detections):
            if used[i]:
                continue

            # Find all overlapping detections
            group = [det]
            used[i] = True

            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                dist = math.sqrt(
                    (det.pixel_x - detections[j].pixel_x) ** 2 +
                    (det.pixel_y - detections[j].pixel_y) ** 2
                )
                if dist < distance_threshold:
                    group.append(detections[j])
                    used[j] = True

            # Keep the highest-confidence detection from the group
            best = max(group, key=lambda d: d.confidence)
            # Boost confidence if multiple methods agreed
            if len(group) > 1:
                best.confidence = min(best.confidence + 0.2, 1.0)
            merged.append(best)

        return merged

    def _draw_debug(self, frame: np.ndarray,
                     detections: List[DetectedObject]):
        """Draw detection results on frame for debugging."""
        debug_frame = frame.copy()

        for det in detections:
            if det.obj_type == DetectedObjectType.PING_PONG:
                color = (0, 255, 0)   # Green for ping-pong
                label = f"PP {det.confidence:.2f}"
            else:
                color = (0, 165, 255)  # Orange for bearing
                label = f"BB {det.confidence:.2f}"

            cv2.circle(debug_frame, (det.pixel_x, det.pixel_y),
                       det.radius_px, color, 2)
            cv2.putText(debug_frame, label,
                        (det.pixel_x - 20, det.pixel_y - det.radius_px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("Ball Detection", debug_frame)


# ============================================================
# Robot Detection (for collision avoidance)
# ============================================================

class RobotDetector:
    """
    Detects other robots in the arena for collision avoidance.

    Strategies:
        1. Color/shape: Robots have a visible team number board (50x50mm+)
           and are generally dark/colored objects on a white floor
        2. Motion: Background subtraction detects anything moving
        3. Size: Robots are 200-300mm, much larger than balls
        4. AprilTag boards: Some robots may have visible markers
    """

    def __init__(self):
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=40,
            detectShadows=True
        )

        # Minimum contour area to be considered a robot (in pixels)
        # Adjust based on camera resolution and distance
        self.min_robot_area = 2000
        self.max_robot_area = 50000

        # Tracking: keep history to smooth detections
        self.tracked_robots: List[TrackedRobot] = []
        self.next_id = 0

    def detect(self, frame: np.ndarray,
               debug: bool = False) -> List[DetectedObject]:
        """
        Detect other robots in the frame.

        Uses a combination of:
        - Background subtraction (motion)
        - Color filtering (robots are darker than white floor)
        - Size filtering (robots are large)
        """
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Method 1: Background subtraction (motion) ---
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (marked as 127 by MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # --- Method 2: Dark object detection ---
        # Robots are typically darker than the white floor
        _, dark_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Combine motion and darkness
        combined = cv2.bitwise_or(fg_mask, dark_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.min_robot_area or area > self.max_robot_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)

            # Robots are roughly square-ish (200x200 to 300x300mm)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue  # Too elongated, probably not a robot

            # Center of detected robot
            cx = x + w // 2
            cy = y + h // 2

            # Confidence based on size and shape
            confidence = 0.5
            if 0.6 < aspect_ratio < 1.6:
                confidence += 0.2  # Good aspect ratio
            if area > self.min_robot_area * 2:
                confidence += 0.1  # Good size

            detections.append(DetectedObject(
                obj_type=DetectedObjectType.ROBOT,
                pixel_x=cx, pixel_y=cy,
                confidence=confidence,
                radius_px=max(w, h) // 2
            ))

        # Update tracking
        self._update_tracking(detections)

        if debug:
            self._draw_debug(frame, detections, combined)

        return detections

    def _update_tracking(self, detections: List[DetectedObject]):
        """Simple tracking to smooth robot detections across frames."""
        MATCH_DISTANCE = 80  # pixels

        matched = [False] * len(detections)

        # Try to match new detections to existing tracks
        for track in self.tracked_robots:
            best_dist = float('inf')
            best_idx = -1

            for i, det in enumerate(detections):
                if matched[i]:
                    continue
                dist = math.sqrt(
                    (track.x - det.pixel_x)**2 +
                    (track.y - det.pixel_y)**2
                )
                if dist < best_dist and dist < MATCH_DISTANCE:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                # Update track with exponential moving average
                alpha = 0.3
                track.x = int(alpha * detections[best_idx].pixel_x +
                              (1 - alpha) * track.x)
                track.y = int(alpha * detections[best_idx].pixel_y +
                              (1 - alpha) * track.y)
                track.frames_seen += 1
                track.frames_missing = 0
                matched[best_idx] = True
            else:
                track.frames_missing += 1

        # Remove stale tracks
        self.tracked_robots = [
            t for t in self.tracked_robots if t.frames_missing < 10
        ]

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if not matched[i]:
                self.tracked_robots.append(TrackedRobot(
                    id=self.next_id,
                    x=det.pixel_x,
                    y=det.pixel_y
                ))
                self.next_id += 1

    def get_tracked_positions(self) -> List[Tuple[int, int]]:
        """Get smoothed positions of tracked robots (pixel coords)."""
        return [
            (t.x, t.y)
            for t in self.tracked_robots
            if t.frames_seen >= 3  # Only report stable tracks
        ]

    def _draw_debug(self, frame, detections, mask):
        debug = frame.copy()
        for det in detections:
            cv2.rectangle(
                debug,
                (det.pixel_x - det.radius_px, det.pixel_y - det.radius_px),
                (det.pixel_x + det.radius_px, det.pixel_y + det.radius_px),
                (0, 0, 255), 2
            )
            cv2.putText(debug, f"ROBOT {det.confidence:.2f}",
                        (det.pixel_x - 30, det.pixel_y - det.radius_px - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Show tracked robots
        for track in self.tracked_robots:
            if track.frames_seen >= 3:
                cv2.circle(debug, (track.x, track.y), 5, (255, 0, 0), -1)
                cv2.putText(debug, f"ID:{track.id}",
                            (track.x + 10, track.y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.imshow("Robot Detection", debug)
        cv2.imshow("Detection Mask", mask)


@dataclass
class TrackedRobot:
    id: int
    x: int
    y: int
    frames_seen: int = 1
    frames_missing: int = 0


# ============================================================
# AprilTag Localization + Pixel-to-World Transform
# ============================================================

class Localizer:
    """
    Uses AprilTags on arena walls to:
    1. Determine the robot's own position and heading in the arena
    2. Transform pixel coordinates of detected objects to arena coordinates

    AprilTag layout (from rulebook):
        North wall (top):    IDs 0-5
        East wall (right):   IDs 6-11
        South wall (bottom): IDs 12-17
        West wall (left):    IDs 18-23

    Tags are 100x100mm, spaced as shown in Appendix A.
    """

    # Known AprilTag positions in arena coordinates (center of each tag)
    # Format: tag_id -> (x_mm, y_mm, z_mm, wall)
    # z_mm is height of tag center (approx half wall height = 75mm)
    TAG_POSITIONS = {
        # North wall (y = 2000)
        0:  (150,  2000, 75, "north"),
        1:  (450,  2000, 75, "north"),
        2:  (750,  2000, 75, "north"),
        3:  (1250, 2000, 75, "north"),
        4:  (1550, 2000, 75, "north"),
        5:  (1850, 2000, 75, "north"),
        # East wall (x = 2000)
        6:  (2000, 1850, 75, "east"),
        7:  (2000, 1350, 75, "east"),
        8:  (2000, 1050, 75, "east"),
        9:  (2000, 750,  75, "east"),
        10: (2000, 450,  75, "east"),
        11: (2000, 150,  75, "east"),
        # South wall (y = 0)
        12: (1850, 0, 75, "south"),
        13: (1550, 0, 75, "south"),
        14: (1250, 0, 75, "south"),
        15: (750,  0, 75, "south"),
        16: (450,  0, 75, "south"),
        17: (150,  0, 75, "south"),
        # West wall (x = 0)
        18: (0, 150,  75, "west"),
        19: (0, 450,  75, "west"),
        20: (0, 750,  75, "west"),
        21: (0, 1050, 75, "west"),
        22: (0, 1350, 75, "west"),
        23: (0, 1850, 75, "west"),
    }

    def __init__(self, camera_params: CameraCalibration):
        self.camera_params = camera_params

        if HAS_APRILTAGS:
            self.detector = AprilTagDetector(
                families="tag36h11",
                nthreads=2,
                quad_decimate=1.0,
                quad_sigma=0.0,
                decode_sharpening=0.25
            )
        else:
            self.detector = None

        # Camera-to-robot transform (adjust for your mounting)
        # How far the camera is from robot center, in mm
        self.camera_offset_x = 0    # mm forward from robot center
        self.camera_offset_y = 0    # mm left from robot center
        self.camera_height = 180    # mm above floor

        # Latest pose estimate
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_heading = 0.0  # degrees, 0=north CW positive
        self.pose_valid = False

        # Homography for top-down pixel-to-world mapping
        self.homography = None

    def update_pose(self, gray_frame: np.ndarray) -> bool:
        """
        Detect AprilTags and estimate robot pose.

        Args:
            gray_frame: Grayscale image

        Returns:
            True if pose was successfully estimated
        """
        if self.detector is None:
            return False

        results = self.detector.detect(
            gray_frame,
            estimate_tag_pose=True,
            camera_params=(
                self.camera_params.fx,
                self.camera_params.fy,
                self.camera_params.cx,
                self.camera_params.cy
            ),
            tag_size=0.1  # 100mm = 0.1m
        )

        if not results:
            self.pose_valid = False
            return False

        # Use all visible tags for a more robust estimate
        pose_estimates = []

        for detection in results:
            tag_id = detection.tag_id
            if tag_id not in self.TAG_POSITIONS:
                continue

            tag_world = self.TAG_POSITIONS[tag_id]

            # The pose_t gives translation from camera to tag
            # The pose_R gives rotation from camera to tag
            if detection.pose_t is None:
                continue

            t = detection.pose_t.flatten()  # [x, y, z] in meters
            R = detection.pose_R

            # Distance to this tag
            dist_to_tag = np.linalg.norm(t)

            # Camera position relative to tag (invert the transform)
            # Camera in tag frame
            cam_in_tag = -R.T @ t

            # Convert to arena coordinates
            tx, ty, tz, wall = tag_world

            if wall == "north":
                robot_x = tx + cam_in_tag[0] * 1000
                robot_y = ty - cam_in_tag[2] * 1000
                heading_offset = 0
            elif wall == "south":
                robot_x = tx - cam_in_tag[0] * 1000
                robot_y = ty + cam_in_tag[2] * 1000
                heading_offset = 180
            elif wall == "east":
                robot_x = tx - cam_in_tag[2] * 1000
                robot_y = ty + cam_in_tag[0] * 1000
                heading_offset = 90
            elif wall == "west":
                robot_x = tx + cam_in_tag[2] * 1000
                robot_y = ty - cam_in_tag[0] * 1000
                heading_offset = 270

            # Extract heading from rotation matrix
            heading = math.degrees(math.atan2(R[0][2], R[2][2]))
            heading = (heading + heading_offset) % 360

            # Weight by inverse distance (closer tags = more accurate)
            weight = 1.0 / max(dist_to_tag, 0.1)

            pose_estimates.append((robot_x, robot_y, heading, weight))

        if not pose_estimates:
            self.pose_valid = False
            return False

        # Weighted average of all pose estimates
        total_weight = sum(w for _, _, _, w in pose_estimates)
        self.robot_x = sum(x * w for x, _, _, w in pose_estimates) / total_weight
        self.robot_y = sum(y * w for _, y, _, w in pose_estimates) / total_weight

        # Average heading (handle wraparound)
        sin_sum = sum(math.sin(math.radians(h)) * w
                      for _, _, h, w in pose_estimates)
        cos_sum = sum(math.cos(math.radians(h)) * w
                      for _, _, h, w in pose_estimates)
        self.robot_heading = math.degrees(
            math.atan2(sin_sum, cos_sum)) % 360

        self.pose_valid = True
        return True

    def pixel_to_world(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Convert a pixel position to arena coordinates.

        Uses the robot's known pose and camera geometry to project
        a point on the floor plane.

        Args:
            pixel_x, pixel_y: Pixel position in the image

        Returns:
            (world_x, world_y) in mm arena coordinates
        """
        if not self.pose_valid:
            return 0.0, 0.0

        # Ray from camera through pixel
        fx = self.camera_params.fx
        fy = self.camera_params.fy
        cx = self.camera_params.cx
        cy = self.camera_params.cy

        # Direction in camera frame
        ray_x = (pixel_x - cx) / fx
        ray_y = (pixel_y - cy) / fy
        ray_z = 1.0

        # Camera tilt (assumes camera points forward, tilted down)
        # Adjust camera_tilt_deg based on your mounting angle
        camera_tilt_deg = 30  # Degrees below horizontal
        tilt_rad = math.radians(camera_tilt_deg)

        # Rotate ray by camera tilt
        cos_t = math.cos(tilt_rad)
        sin_t = math.sin(tilt_rad)
        ray_y_tilted = ray_y * cos_t - ray_z * sin_t
        ray_z_tilted = ray_y * sin_t + ray_z * cos_t

        # Intersect with floor plane (z = 0 in robot frame,
        # camera is at height camera_height)
        if ray_z_tilted <= 0.01:
            return 0.0, 0.0  # Ray points upward, no floor intersection

        t = self.camera_height / ray_z_tilted
        floor_x_local = ray_x * t  # mm right of camera
        floor_y_local = ray_y_tilted * t  # mm forward of camera

        # Account for camera offset from robot center
        robot_frame_x = floor_y_local + self.camera_offset_x
        robot_frame_y = -floor_x_local + self.camera_offset_y

        # Rotate to arena frame using robot heading
        heading_rad = math.radians(self.robot_heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        # Robot heading: 0=north means forward=+Y in arena
        arena_dx = robot_frame_x * sin_h + robot_frame_y * cos_h
        arena_dy = robot_frame_x * cos_h - robot_frame_y * sin_h

        world_x = self.robot_x + arena_dx
        world_y = self.robot_y + arena_dy

        return world_x, world_y


# ============================================================
# Collision Avoidance System
# ============================================================

class CollisionAvoidance:
    """
    Real-time collision avoidance that modifies velocity commands.

    Sits between your navigation planner and the motor controller.
    Takes the desired velocity and adjusts it to avoid detected obstacles.

    Uses a time-to-collision approach:
    - Predict where the robot will be in the near future
    - If a collision is likely, modify the velocity to avoid it
    - Prioritize lateral avoidance (strafe around) over stopping

    This is a non-contact sport (Rule 3.2) and collisions trigger
    a reset (Rule 1.11), so avoidance is critical.
    """

    def __init__(self):
        # Distances in mm
        self.robot_radius = 150       # Half of 300mm max
        self.danger_zone = 300        # Start avoiding at this distance
        self.emergency_zone = 180     # Emergency stop/dodge distance
        self.avoidance_strength = 0.8 # How aggressively to avoid (0-1)

        # Ultrasonic/IR sensor integration (optional)
        self.sensor_readings: Dict[str, float] = {}

    def adjust_velocity(self,
                        desired_vx: float,
                        desired_vy: float,
                        desired_omega: float,
                        robot_pos: Tuple[float, float],
                        robot_heading: float,
                        obstacles: List[Tuple[float, float]]
                        ) -> Tuple[float, float, float]:
        """
        Adjust desired velocity to avoid collisions.

        Args:
            desired_vx, desired_vy, desired_omega: What navigation wants
            robot_pos: Current (x, y) in mm
            robot_heading: Current heading in degrees
            obstacles: List of (x, y) positions of other robots in mm

        Returns:
            (adjusted_vx, adjusted_vy, adjusted_omega) safe velocities
        """
        if not obstacles:
            return desired_vx, desired_vy, desired_omega

        vx, vy, omega = desired_vx, desired_vy, desired_omega
        heading_rad = math.radians(robot_heading)

        for obs_x, obs_y in obstacles:
            # Vector from robot to obstacle (global frame)
            dx = obs_x - robot_pos[0]
            dy = obs_y - robot_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > self.danger_zone:
                continue  # Too far to worry about

            # Convert to robot's local frame
            cos_h = math.cos(heading_rad)
            sin_h = math.sin(heading_rad)
            local_x = dx * sin_h + dy * cos_h     # Forward
            local_y = -dx * cos_h + dy * sin_h    # Left

            # --- Emergency zone: strong evasive action ---
            if distance < self.emergency_zone:
                # Which side is the obstacle on?
                if abs(local_y) < 0.01:
                    # Directly ahead or behind — dodge left
                    dodge_vy = 0.8
                else:
                    # Dodge away from obstacle
                    dodge_vy = -0.8 * (1 if local_y > 0 else -1)

                # Slow down forward/backward motion toward obstacle
                if local_x > 0:
                    vx = min(vx, 0.1)   # Obstacle ahead, slow down
                elif local_x < 0:
                    vx = max(vx, -0.1)  # Obstacle behind

                vy = vy * 0.3 + dodge_vy * 0.7
                continue

            # --- Danger zone: proportional avoidance ---
            # How close are we (0 = edge of danger zone, 1 = emergency zone)
            proximity = 1.0 - (distance - self.emergency_zone) / \
                        (self.danger_zone - self.emergency_zone)
            proximity = max(0, min(1, proximity))

            # Avoidance force: push away from obstacle in local frame
            if distance > 0:
                avoid_x = -local_x / distance  # Push away (forward/back)
                avoid_y = -local_y / distance  # Push away (left/right)
            else:
                avoid_x, avoid_y = 0, 0

            strength = proximity * self.avoidance_strength

            # Blend avoidance with desired velocity
            vx = vx * (1 - strength * 0.5) + avoid_x * strength * 0.5
            vy = vy * (1 - strength) + avoid_y * strength

        # Clamp final values
        vx = max(-1.0, min(1.0, vx))
        vy = max(-1.0, min(1.0, vy))
        omega = max(-1.0, min(1.0, omega))

        return vx, vy, omega

    def update_sensor(self, sensor_name: str, distance_mm: float):
        """
        Update distance reading from a proximity sensor.

        Call this with data from ultrasonic or IR sensors if you have them.
        Sensor names: "front", "back", "left", "right", etc.
        """
        self.sensor_readings[sensor_name] = distance_mm

    def check_sensors(self) -> bool:
        """
        Quick check: is anything dangerously close based on sensors?
        Returns True if emergency stop is needed.
        """
        for sensor, dist in self.sensor_readings.items():
            if dist < self.emergency_zone:
                return True
        return False


# ============================================================
# Complete Vision System (ties everything together)
# ============================================================

class VisionSystem:
    """
    Complete vision pipeline for the Unibots robot.

    Runs the camera, detects balls and robots, localizes using
    AprilTags, and converts everything to arena coordinates.

    Usage:
        vision = VisionSystem(camera_index=0)
        vision.calibrate_floor()

        while match_running:
            detections = vision.process_frame()
            balls = detections["balls"]
            obstacles = detections["obstacles"]
            pose = detections["pose"]
    """

    def __init__(self, camera_index: int = 0,
                 width: int = 640, height: int = 480):
        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        # Camera calibration (default values — run proper calibration!)
        self.camera_params = CameraCalibration(
            fx=width * 0.9,  # Rough estimate
            fy=width * 0.9,
            cx=width / 2,
            cy=height / 2
        )

        # Sub-systems
        self.ball_detector = BallDetector()
        self.robot_detector = RobotDetector()
        self.localizer = Localizer(self.camera_params)
        self.collision_avoidance = CollisionAvoidance()

        # Processing rate
        self.last_frame_time = time.time()
        self.fps = 0

    def calibrate_floor(self):
        """Capture a floor sample for ball detection calibration."""
        print("Calibrating floor... Make sure camera sees empty floor.")
        time.sleep(1)
        ret, frame = self.cap.read()
        if ret:
            self.ball_detector.calibrate_floor(frame)
        else:
            print("Failed to capture calibration frame")

    def process_frame(self, debug: bool = False) -> dict:
        """
        Capture and process one frame.

        Returns:
            {
                "balls": [DetectedObject, ...],     # Ball positions (arena coords)
                "obstacles": [(x, y), ...],          # Robot positions (arena coords)
                "pose": (x, y, heading) or None,     # Robot's own pose
                "fps": float                          # Processing rate
            }
        """
        ret, frame = self.cap.read()
        if not ret:
            return {"balls": [], "obstacles": [], "pose": None, "fps": 0}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Localization ---
        pose_valid = self.localizer.update_pose(gray)
        pose = None
        if pose_valid:
            pose = (
                self.localizer.robot_x,
                self.localizer.robot_y,
                self.localizer.robot_heading
            )

        # --- Ball detection ---
        balls = self.ball_detector.detect(frame, debug=debug)

        # Convert ball pixel positions to arena coordinates
        if pose_valid:
            for ball in balls:
                wx, wy = self.localizer.pixel_to_world(
                    ball.pixel_x, ball.pixel_y)
                ball.world_x = wx
                ball.world_y = wy

        # --- Robot detection ---
        robots = self.robot_detector.detect(frame, debug=debug)

        # Convert robot pixel positions to arena coordinates
        obstacle_positions = []
        if pose_valid:
            for robot in robots:
                wx, wy = self.localizer.pixel_to_world(
                    robot.pixel_x, robot.pixel_y)
                robot.world_x = wx
                robot.world_y = wy
                obstacle_positions.append((wx, wy))

        # --- FPS calculation ---
        now = time.time()
        self.fps = 1.0 / max(now - self.last_frame_time, 0.001)
        self.last_frame_time = now

        if debug:
            self._draw_hud(frame, balls, robots, pose)
            cv2.waitKey(1)

        return {
            "balls": balls,
            "obstacles": obstacle_positions,
            "pose": pose,
            "fps": self.fps
        }

    def _draw_hud(self, frame, balls, robots, pose):
        """Draw heads-up display with all detections."""
        hud = frame.copy()

        # Pose info
        if pose:
            text = f"Pose: ({pose[0]:.0f}, {pose[1]:.0f}) {pose[2]:.1f}deg"
            cv2.putText(hud, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Detection counts
        pp_count = sum(1 for b in balls
                       if b.obj_type == DetectedObjectType.PING_PONG)
        bb_count = sum(1 for b in balls
                       if b.obj_type == DetectedObjectType.BALL_BEARING)
        cv2.putText(hud, f"PP: {pp_count}  BB: {bb_count}  Robots: {len(robots)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # FPS
        cv2.putText(hud, f"FPS: {self.fps:.1f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Unibots Vision", hud)

    def release(self):
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    import sys

    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    debug_mode = "--debug" in sys.argv

    print(f"Starting vision system on camera {camera_idx}")
    print("Press 'q' to quit, 'c' to calibrate floor")

    vision = VisionSystem(camera_index=camera_idx)

    try:
        while True:
            result = vision.process_frame(debug=True)

            if result["pose"]:
                x, y, h = result["pose"]
                print(f"\rPose: ({x:.0f}, {y:.0f}) {h:.1f}°  "
                      f"Balls: {len(result['balls'])}  "
                      f"Obstacles: {len(result['obstacles'])}  "
                      f"FPS: {result['fps']:.1f}   ", end="")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                vision.calibrate_floor()

    except KeyboardInterrupt:
        pass
    finally:
        vision.release()
        print("\nVision system stopped.")
