"""
Reactive Navigation for Unibots Mecanum Robot
===============================================
Built around the principle: act on what you can SEE, not what you PLANNED.

Architecture (layered, highest priority first):
    Layer 1 — Emergency:  Collision imminent → dodge/stop
    Layer 2 — Reactive:   See a ball → go grab it
    Layer 3 — Deliberate: No balls visible → explore to find some
    Layer 4 — Endgame:    Time running out → deposit/park

Each control cycle (~50Hz), the robot picks the highest-priority
layer that has something to say, and acts on it. No long-term plans
that go stale.

Key insight: You don't NEED to know where all the balls are.
You only need to know what's in front of you RIGHT NOW.

Dependencies:
    pip install numpy
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


# ============================================================
# Data Structures
# ============================================================

class BallType(Enum):
    PING_PONG = "ping_pong"
    BALL_BEARING = "ball_bearing"


class RobotState(Enum):
    EXPLORING = "exploring"          # Looking for balls
    PURSUING_BALL = "pursuing_ball"  # Moving toward a visible ball
    COLLECTING = "collecting"        # Actuating collection mechanism
    RETURNING = "returning"          # Going back to deposit
    DEPOSITING = "depositing"        # At net, releasing balls
    PARKING = "parking"              # End of match, touching wall
    EMERGENCY_DODGE = "dodging"      # Avoiding collision
    STOPPED = "stopped"


@dataclass
class VisibleBall:
    """A ball the robot can currently see."""
    x: float            # Arena mm (from vision)
    y: float
    ball_type: BallType
    confidence: float
    last_seen: float    # timestamp

    @property
    def points(self) -> int:
        return 4 if self.ball_type == BallType.PING_PONG else 3

    @property
    def age(self) -> float:
        """Seconds since last seen."""
        return time.time() - self.last_seen


@dataclass
class VisibleRobot:
    """Another robot the system can currently see."""
    x: float
    y: float
    last_seen: float
    # Estimated velocity (built up over multiple sightings)
    vx: float = 0.0
    vy: float = 0.0

    @property
    def age(self) -> float:
        return time.time() - self.last_seen

    def predicted_position(self, dt: float) -> Tuple[float, float]:
        """Where will this robot be in dt seconds?"""
        return (self.x + self.vx * dt, self.y + self.vy * dt)


@dataclass
class ScoringZone:
    wall: str
    net_center_x: float
    net_center_y: float
    approach_x: float
    approach_y: float
    approach_heading: float


SCORING_ZONES = {
    "north": ScoringZone("north", 1000, 2000, 1000, 1800, 0),
    "south": ScoringZone("south", 1000, 0, 1000, 200, 180),
    "east":  ScoringZone("east", 2000, 1000, 1800, 1000, 90),
    "west":  ScoringZone("west", 0, 1000, 200, 1000, 270),
}


# ============================================================
# World Model (short-term memory of what we've seen)
# ============================================================

class WorldModel:
    """
    Keeps a short-term memory of detected objects.

    NOT a static map. Objects fade out if not re-detected.
    This handles the constantly changing field:
        - Balls disappear when collected (by us or others)
        - Robots move around
        - New balls may appear (returned to arena by volunteers)

    The key timeout values:
        - Balls: forget after 2 seconds unseen (they get collected fast)
        - Robots: forget after 1 second unseen (they move fast)
    """

    def __init__(self):
        self.balls: Dict[int, VisibleBall] = {}  # id -> ball
        self.robots: Dict[int, VisibleRobot] = {}
        self.next_ball_id = 0
        self.next_robot_id = 0

        # How long to remember things (seconds)
        self.ball_timeout = 2.0
        self.robot_timeout = 1.0

        # Merge distance: if a new detection is this close to an
        # existing one, update rather than create new
        self.ball_merge_dist = 80    # mm
        self.robot_merge_dist = 200  # mm

        # Areas we've recently explored (to avoid re-searching)
        self.explored_cells: Dict[Tuple[int, int], float] = {}
        self.explore_cell_size = 250  # mm
        self.explore_timeout = 15.0   # seconds before re-exploring

    def update_balls(self, detections: List[dict]):
        """
        Update ball memory with new detections from vision.

        Args:
            detections: List of {"x": float, "y": float,
                                  "type": BallType, "confidence": float}
        """
        now = time.time()

        for det in detections:
            matched = False
            for bid, ball in self.balls.items():
                dist = _dist((ball.x, ball.y), (det["x"], det["y"]))
                if dist < self.ball_merge_dist:
                    # Update existing ball with exponential moving average
                    alpha = 0.4
                    ball.x = alpha * det["x"] + (1 - alpha) * ball.x
                    ball.y = alpha * det["y"] + (1 - alpha) * ball.y
                    ball.confidence = max(ball.confidence, det["confidence"])
                    ball.last_seen = now
                    matched = True
                    break

            if not matched:
                self.balls[self.next_ball_id] = VisibleBall(
                    x=det["x"], y=det["y"],
                    ball_type=det["type"],
                    confidence=det["confidence"],
                    last_seen=now
                )
                self.next_ball_id += 1

        # Expire old balls
        expired = [bid for bid, b in self.balls.items()
                   if b.age > self.ball_timeout]
        for bid in expired:
            del self.balls[bid]

    def update_robots(self, detections: List[dict]):
        """
        Update robot memory with new detections.

        Args:
            detections: List of {"x": float, "y": float}
        """
        now = time.time()

        for det in detections:
            matched = False
            for rid, robot in self.robots.items():
                dist = _dist((robot.x, robot.y), (det["x"], det["y"]))
                if dist < self.robot_merge_dist:
                    # Estimate velocity from position change
                    dt = now - robot.last_seen
                    if dt > 0.01:
                        alpha = 0.5
                        new_vx = (det["x"] - robot.x) / dt
                        new_vy = (det["y"] - robot.y) / dt
                        robot.vx = alpha * new_vx + (1 - alpha) * robot.vx
                        robot.vy = alpha * new_vy + (1 - alpha) * robot.vy

                    robot.x = det["x"]
                    robot.y = det["y"]
                    robot.last_seen = now
                    matched = True
                    break

            if not matched:
                self.robots[self.next_robot_id] = VisibleRobot(
                    x=det["x"], y=det["y"], last_seen=now
                )
                self.next_robot_id += 1

        # Expire old robots
        expired = [rid for rid, r in self.robots.items()
                   if r.age > self.robot_timeout]
        for rid in expired:
            del self.robots[rid]

    def mark_explored(self, x: float, y: float):
        """Mark an area as recently explored."""
        cell = (int(x // self.explore_cell_size),
                int(y // self.explore_cell_size))
        self.explored_cells[cell] = time.time()

    def is_explored(self, x: float, y: float) -> bool:
        """Has this area been explored recently?"""
        cell = (int(x // self.explore_cell_size),
                int(y // self.explore_cell_size))
        if cell in self.explored_cells:
            return (time.time() - self.explored_cells[cell]) < self.explore_timeout
        return False

    def get_visible_balls(self) -> List[VisibleBall]:
        """Get all currently-remembered balls, sorted by confidence."""
        return sorted(self.balls.values(),
                      key=lambda b: b.confidence, reverse=True)

    def get_visible_robots(self) -> List[VisibleRobot]:
        """Get all currently-remembered robots."""
        return list(self.robots.values())

    def get_unexplored_direction(self, robot_x: float, robot_y: float,
                                  arena_size: float = 2000) -> Tuple[float, float]:
        """
        Suggest a direction to explore (area not recently visited).
        Returns a target (x, y) to move toward.
        """
        now = time.time()
        best_target = None
        best_score = -1

        # Check grid cells across the arena
        for gx in range(0, int(arena_size), self.explore_cell_size):
            for gy in range(0, int(arena_size), self.explore_cell_size):
                cell = (gx // self.explore_cell_size,
                        gy // self.explore_cell_size)
                cx = gx + self.explore_cell_size / 2
                cy = gy + self.explore_cell_size / 2

                # How long since explored (higher = more interesting)
                if cell in self.explored_cells:
                    staleness = now - self.explored_cells[cell]
                    if staleness < self.explore_timeout:
                        continue  # Recently explored, skip
                else:
                    staleness = self.explore_timeout  # Never explored

                # Prefer closer unexplored areas
                dist = _dist((robot_x, robot_y), (cx, cy))
                if dist < 50:
                    continue  # Already here

                # Score: prefer close + stale areas
                score = staleness / max(dist, 1)

                if score > best_score:
                    best_score = score
                    best_target = (cx, cy)

        if best_target is None:
            # Everything explored — go to center
            best_target = (arena_size / 2, arena_size / 2)

        return best_target


# ============================================================
# Layer 1: Emergency Collision Avoidance
# ============================================================

class EmergencyLayer:
    """
    Highest priority. Prevents physical collisions.

    Collisions are BAD in Unibots:
        - Rule 1.11: Collision = both robots get reset to starting zone
        - Rule 3.2: Intentional contact = possible disqualification
        - You lose time, position, and momentum

    This layer takes over when another robot is dangerously close.
    It uses the robot's velocity estimate to predict collisions.
    """

    def __init__(self):
        self.emergency_distance = 200   # mm — must react
        self.warning_distance = 350     # mm — start adjusting
        self.prediction_time = 1.0      # seconds to look ahead

    def check(self, robot_pos: Tuple[float, float],
              robot_heading: float,
              robot_velocity: Tuple[float, float],
              other_robots: List[VisibleRobot]) -> Optional[Tuple[float, float, float]]:
        """
        Check for imminent collisions.

        Returns:
            (vx, vy, omega) emergency velocity, or None if no danger
        """
        if not other_robots:
            return None

        most_urgent = None
        min_time_to_collision = float('inf')

        for other in other_robots:
            # Current distance
            dist = _dist(robot_pos, (other.x, other.y))

            if dist > self.warning_distance:
                continue

            # Predict future positions
            other_future = other.predicted_position(self.prediction_time)
            my_future = (
                robot_pos[0] + robot_velocity[0] * self.prediction_time,
                robot_pos[1] + robot_velocity[1] * self.prediction_time
            )
            future_dist = _dist(my_future, other_future)

            # Are we getting closer?
            if future_dist < dist:
                # Estimate time to collision
                closing_speed = (dist - future_dist) / self.prediction_time
                if closing_speed > 0:
                    ttc = dist / closing_speed
                    if ttc < min_time_to_collision:
                        min_time_to_collision = ttc
                        most_urgent = other

            # Already too close — emergency regardless of prediction
            if dist < self.emergency_distance:
                most_urgent = other
                min_time_to_collision = 0

        if most_urgent is None:
            return None

        # Calculate dodge direction (perpendicular to obstacle)
        dx = most_urgent.x - robot_pos[0]
        dy = most_urgent.y - robot_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1:
            dist = 1

        # Perpendicular dodge: rotate obstacle vector 90 degrees
        # Choose the direction that's more aligned with our current heading
        dodge_x1, dodge_y1 = -dy / dist, dx / dist
        dodge_x2, dodge_y2 = dy / dist, -dx / dist

        heading_rad = math.radians(robot_heading)
        forward_x = math.sin(heading_rad)
        forward_y = math.cos(heading_rad)

        # Pick dodge direction more aligned with where we're going
        dot1 = dodge_x1 * forward_x + dodge_y1 * forward_y
        dot2 = dodge_x2 * forward_x + dodge_y2 * forward_y

        if dot1 > dot2:
            dodge_gx, dodge_gy = dodge_x1, dodge_y1
        else:
            dodge_gx, dodge_gy = dodge_x2, dodge_y2

        # Also back away from the obstacle
        away_x = -dx / dist
        away_y = -dy / dist

        # Blend dodge and retreat based on distance
        if dist < self.emergency_distance:
            # Very close: mostly retreat + strong dodge
            gx = away_x * 0.4 + dodge_gx * 0.6
            gy = away_y * 0.4 + dodge_gy * 0.6
            speed = 0.9
        else:
            # Warning zone: mostly dodge, slight retreat
            gx = away_x * 0.2 + dodge_gx * 0.8
            gy = away_y * 0.2 + dodge_gy * 0.8
            speed = 0.6

        # Convert global velocity to local frame
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        vx_local = (gx * sin_h + gy * cos_h) * speed
        vy_local = (-gx * cos_h + gy * sin_h) * speed

        return (vx_local, vy_local, 0.0)


# ============================================================
# Layer 2: Reactive Ball Pursuit
# ============================================================

class PursuitLayer:
    """
    When balls are visible, go get them.

    Priority logic:
        1. Ping-pong balls (4 pts) preferred over bearings (3 pts)
        2. Closer balls preferred
        3. Balls that aren't near other robots preferred
        4. Balls near our scoring zone slightly preferred (shorter return trip)

    No path planning needed — just drive toward the best ball,
    and let Layer 1 handle any robots in the way.
    """

    def __init__(self, scoring_zone: ScoringZone):
        self.zone = scoring_zone
        self.arrival_distance = 60  # mm — close enough to trigger collection

    def choose_target(self,
                      robot_pos: Tuple[float, float],
                      balls: List[VisibleBall],
                      other_robots: List[VisibleRobot],
                      balls_held: int,
                      capacity: int) -> Optional[VisibleBall]:
        """
        Pick the best ball to go after right now.

        Returns the target ball, or None if no good target.
        """
        if not balls or balls_held >= capacity:
            return None

        scored = []
        for ball in balls:
            if ball.confidence < 0.35:
                continue

            dist_to_ball = _dist(robot_pos, (ball.x, ball.y))

            # Skip balls that are basically on top of us (probably
            # already being collected or a ghost detection)
            if dist_to_ball < 20:
                continue

            # Base score: points per millimeter of travel
            score = ball.points / max(dist_to_ball, 1)

            # Penalty: ball near another robot (risky, might cause collision)
            for robot in other_robots:
                robot_dist = _dist((ball.x, ball.y), (robot.x, robot.y))
                if robot_dist < 300:
                    score *= 0.3  # Heavy penalty — not worth the collision risk
                elif robot_dist < 500:
                    score *= 0.7

            # Bonus: ball closer to our scoring zone (shorter return trip)
            dist_to_net = _dist((ball.x, ball.y),
                                (self.zone.approach_x, self.zone.approach_y))
            # Normalize: arena diagonal is ~2828mm
            net_bonus = 1.0 + 0.3 * (1.0 - dist_to_net / 2828)
            score *= net_bonus

            # Bonus: fresher detections (seen more recently)
            freshness = max(0, 1.0 - ball.age / 2.0)
            score *= (0.5 + 0.5 * freshness)

            scored.append((score, ball))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def compute_velocity(self,
                         robot_pos: Tuple[float, float],
                         robot_heading: float,
                         target: VisibleBall) -> Tuple[float, float, float]:
        """
        Compute velocity to drive toward the target ball.

        Returns (vx_local, vy_local, omega) for the mecanum controller.
        """
        dx = target.x - robot_pos[0]
        dy = target.y - robot_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 1:
            return (0, 0, 0)

        # Speed: proportional to distance, but capped
        speed = min(0.7, max(0.2, dist / 500))

        # Global direction to target
        global_angle = math.atan2(dx, dy)  # 0 = north

        # Convert to robot local frame
        heading_rad = math.radians(robot_heading)
        local_angle = global_angle - heading_rad

        vx_local = speed * math.cos(local_angle)
        vy_local = speed * math.sin(local_angle)

        # Rotate to face the ball (helps with collection mechanism)
        target_heading = math.degrees(global_angle) % 360
        heading_error = _angle_diff(target_heading, robot_heading)
        omega = max(-0.5, min(0.5, heading_error * 0.015))

        return (vx_local, vy_local, -omega)


# ============================================================
# Layer 3: Exploration (when no balls are visible)
# ============================================================

class ExploreLayer:
    """
    When no balls are visible, systematically search the arena.

    Strategy: drive a pattern that maximizes camera coverage.
    Avoids areas recently explored. Prefers areas near the center
    (where balls are more likely to be after initial scramble).

    The arena is 2000x2000mm with 40 balls scattered around.
    With a forward-facing camera, the robot needs to physically
    drive around to see most of the arena.
    """

    def __init__(self, scoring_zone: ScoringZone):
        self.zone = scoring_zone

        # Exploration pattern: a series of waypoints that give good
        # arena coverage. Generated dynamically based on what's unexplored.
        self.current_target: Optional[Tuple[float, float]] = None
        self.target_reached_dist = 150  # mm

    def compute_velocity(self,
                         robot_pos: Tuple[float, float],
                         robot_heading: float,
                         world: WorldModel) -> Tuple[float, float, float]:
        """
        Compute velocity to explore the arena.

        Returns (vx_local, vy_local, omega).
        """
        # Mark current position as explored
        world.mark_explored(robot_pos[0], robot_pos[1])

        # Check if we've reached current target
        if self.current_target is not None:
            dist = _dist(robot_pos, self.current_target)
            if dist < self.target_reached_dist:
                self.current_target = None

        # Pick new exploration target if needed
        if self.current_target is None:
            self.current_target = world.get_unexplored_direction(
                robot_pos[0], robot_pos[1])

        # Drive toward exploration target
        tx, ty = self.current_target
        dx = tx - robot_pos[0]
        dy = ty - robot_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 1:
            return (0, 0, 0.3)  # Spin in place to look around

        speed = min(0.5, max(0.2, dist / 600))

        global_angle = math.atan2(dx, dy)
        heading_rad = math.radians(robot_heading)
        local_angle = global_angle - heading_rad

        vx_local = speed * math.cos(local_angle)
        vy_local = speed * math.sin(local_angle)

        # Slowly rotate while exploring to scan with camera
        target_heading = math.degrees(global_angle) % 360
        heading_error = _angle_diff(target_heading, robot_heading)
        omega = max(-0.4, min(0.4, heading_error * 0.012))

        return (vx_local, vy_local, -omega)


# ============================================================
# Layer 4: Return & Deposit
# ============================================================

class ReturnLayer:
    """
    Navigate back to scoring zone and deposit balls.

    Triggered when:
        - Robot is at capacity
        - Time is running low
        - Strategic decision to bank points
    """

    def __init__(self, scoring_zone: ScoringZone):
        self.zone = scoring_zone
        self.approach_tolerance = 80  # mm

    def should_return(self, balls_held: int, capacity: int,
                      time_remaining: float,
                      robot_pos: Tuple[float, float]) -> bool:
        """Decide whether to return to deposit now."""
        dist_to_net = _dist(robot_pos,
                            (self.zone.approach_x, self.zone.approach_y))
        travel_time = dist_to_net / 300 + 5  # 300mm/s + 5s buffer

        # Must return: time pressure
        if time_remaining < travel_time + 5:
            return True

        # Should return: at capacity
        if balls_held >= capacity:
            return True

        # Consider returning: have some balls and getting far from net
        if balls_held >= 2 and time_remaining < 60:
            return True

        return False

    def compute_velocity(self,
                         robot_pos: Tuple[float, float],
                         robot_heading: float) -> Tuple[float, float, float]:
        """Drive toward the scoring zone approach point."""
        dx = self.zone.approach_x - robot_pos[0]
        dy = self.zone.approach_y - robot_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < self.approach_tolerance:
            # We're at the net — face it for deposit
            heading_error = _angle_diff(
                self.zone.approach_heading, robot_heading)
            omega = max(-0.5, min(0.5, heading_error * 0.02))
            return (0.1, 0, -omega)  # Creep forward into wall

        speed = min(0.8, max(0.3, dist / 400))

        global_angle = math.atan2(dx, dy)
        heading_rad = math.radians(robot_heading)
        local_angle = global_angle - heading_rad

        vx_local = speed * math.cos(local_angle)
        vy_local = speed * math.sin(local_angle)

        # Gradually turn to face the net as we approach
        progress = max(0, 1.0 - dist / 1000)
        if progress > 0.5:
            target_heading = self.zone.approach_heading
        else:
            target_heading = math.degrees(global_angle) % 360

        heading_error = _angle_diff(target_heading, robot_heading)
        omega = max(-0.5, min(0.5, heading_error * 0.015))

        return (vx_local, vy_local, -omega)

    def is_at_net(self, robot_pos: Tuple[float, float]) -> bool:
        """Are we close enough to the net to deposit?"""
        dist = _dist(robot_pos,
                     (self.zone.approach_x, self.zone.approach_y))
        return dist < self.approach_tolerance


# ============================================================
# Wall Avoidance
# ============================================================

class WallAvoidance:
    """
    Soft repulsion from arena walls.
    Prevents the robot from driving into walls while pursuing balls
    or exploring. Blends with the current velocity command.
    """

    def __init__(self, arena_size: float = 2000):
        self.arena_size = arena_size
        self.wall_buffer = 120       # mm — start pushing away
        self.wall_strength = 0.6     # How much to override velocity

    def adjust(self, vx: float, vy: float, omega: float,
               robot_pos: Tuple[float, float],
               robot_heading: float) -> Tuple[float, float, float]:
        """Add wall repulsion to velocity commands."""
        push_gx, push_gy = 0.0, 0.0

        x, y = robot_pos
        buf = self.wall_buffer

        # Push away from each wall proportionally
        if x < buf:
            push_gx += self.wall_strength * (1 - x / buf)
        if x > self.arena_size - buf:
            push_gx -= self.wall_strength * (1 - (self.arena_size - x) / buf)
        if y < buf:
            push_gy += self.wall_strength * (1 - y / buf)
        if y > self.arena_size - buf:
            push_gy -= self.wall_strength * (1 - (self.arena_size - y) / buf)

        if abs(push_gx) < 0.01 and abs(push_gy) < 0.01:
            return vx, vy, omega

        # Convert push to local frame
        heading_rad = math.radians(robot_heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        push_local_x = push_gx * sin_h + push_gy * cos_h
        push_local_y = -push_gx * cos_h + push_gy * sin_h

        # Blend with existing velocity
        vx = vx * (1 - self.wall_strength) + push_local_x
        vy = vy * (1 - self.wall_strength) + push_local_y

        return vx, vy, omega


# ============================================================
# Main Reactive Controller
# ============================================================

class ReactiveController:
    """
    The brain. Runs every control cycle and picks the right action.

    This replaces the old A*-based GameController with something that
    actually works in a dynamic environment.

    Usage:
        controller = ReactiveController(robot, "north")

        # In your main loop:
        while match_running:
            # Feed in what the camera sees
            controller.update_vision(balls, robots)
            # Get motor command
            vx, vy, omega = controller.tick()
            # Send to Arduino
            robot.arduino.send_velocity(vx, vy, omega)
    """

    def __init__(self, mecanum_robot, zone: str = "north",
                 capacity: int = 5):
        self.robot = mecanum_robot
        self.zone = SCORING_ZONES[zone]
        self.capacity = capacity

        # Layers
        self.emergency = EmergencyLayer()
        self.pursuit = PursuitLayer(self.zone)
        self.explore = ExploreLayer(self.zone)
        self.return_layer = ReturnLayer(self.zone)
        self.wall_avoidance = WallAvoidance()

        # State
        self.world = WorldModel()
        self.state = RobotState.EXPLORING
        self.balls_held = 0
        self.match_start = None
        self.current_target_ball: Optional[VisibleBall] = None

        # Velocity estimate (for collision prediction)
        self.last_pos = None
        self.last_time = None
        self.velocity = (0.0, 0.0)

    @property
    def time_remaining(self) -> float:
        if self.match_start is None:
            return 180.0
        return max(0, 180.0 - (time.time() - self.match_start))

    def start_match(self):
        self.match_start = time.time()

    def update_vision(self, ball_detections: List[dict],
                      robot_detections: List[dict]):
        """
        Feed in current vision detections.

        Args:
            ball_detections: [{"x", "y", "type": BallType, "confidence"}, ...]
            robot_detections: [{"x", "y"}, ...]
        """
        self.world.update_balls(ball_detections)
        self.world.update_robots(robot_detections)

    def tick(self) -> Tuple[float, float, float]:
        """
        One control cycle. Call at ~50Hz.

        Returns:
            (vx, vy, omega) to send to the mecanum controller
        """
        pos = (self.robot.x, self.robot.y)
        heading = self.robot.heading

        # Update velocity estimate
        self._update_velocity(pos)

        balls = self.world.get_visible_balls()
        robots = self.world.get_visible_robots()

        # ========== LAYER 1: Emergency ==========
        emergency_cmd = self.emergency.check(
            pos, heading, self.velocity, robots)

        if emergency_cmd is not None:
            self.state = RobotState.EMERGENCY_DODGE
            vx, vy, omega = emergency_cmd
            # DON'T apply wall avoidance during emergency
            # (might push us into the robot we're dodging)
            return self._clamp(vx, vy, omega)

        # ========== LAYER 4: Endgame ==========
        if self.time_remaining < 10:
            return self._handle_endgame(pos, heading)

        # ========== LAYER: Return to deposit ==========
        if self.return_layer.should_return(
                self.balls_held, self.capacity,
                self.time_remaining, pos):
            self.state = RobotState.RETURNING

            if self.return_layer.is_at_net(pos):
                self.state = RobotState.DEPOSITING
                # TODO: trigger your deposit mechanism
                self.balls_held = 0
                return (0.1, 0, 0)  # Nudge into wall

            vx, vy, omega = self.return_layer.compute_velocity(pos, heading)
            vx, vy, omega = self.wall_avoidance.adjust(
                vx, vy, omega, pos, heading)
            return self._clamp(vx, vy, omega)

        # ========== LAYER 2: Pursue visible ball ==========
        target = self.pursuit.choose_target(
            pos, balls, robots, self.balls_held, self.capacity)

        if target is not None:
            self.state = RobotState.PURSUING_BALL
            self.current_target_ball = target

            # Check if we've reached the ball
            dist = _dist(pos, (target.x, target.y))
            if dist < self.pursuit.arrival_distance:
                self.state = RobotState.COLLECTING
                # TODO: trigger your collection mechanism
                self.balls_held += 1
                # Remove from world model
                to_remove = [bid for bid, b in self.world.balls.items()
                             if _dist((b.x, b.y), (target.x, target.y)) < 100]
                for bid in to_remove:
                    del self.world.balls[bid]
                return (0, 0, 0)

            vx, vy, omega = self.pursuit.compute_velocity(
                pos, heading, target)
            vx, vy, omega = self.wall_avoidance.adjust(
                vx, vy, omega, pos, heading)
            return self._clamp(vx, vy, omega)

        # ========== LAYER 3: Explore ==========
        self.state = RobotState.EXPLORING
        self.current_target_ball = None

        vx, vy, omega = self.explore.compute_velocity(
            pos, heading, self.world)
        vx, vy, omega = self.wall_avoidance.adjust(
            vx, vy, omega, pos, heading)
        return self._clamp(vx, vy, omega)

    def _handle_endgame(self, pos, heading) -> Tuple[float, float, float]:
        """Last 10 seconds: deposit if possible, otherwise just park."""
        dist_to_net = _dist(pos,
                            (self.zone.approach_x, self.zone.approach_y))
        can_reach = dist_to_net / 400 < self.time_remaining

        if can_reach:
            self.state = RobotState.PARKING
            vx, vy, omega = self.return_layer.compute_velocity(pos, heading)
            return self._clamp(vx, vy, omega)
        else:
            self.state = RobotState.STOPPED
            return (0, 0, 0)

    def _update_velocity(self, pos: Tuple[float, float]):
        """Estimate robot velocity from position changes."""
        now = time.time()
        if self.last_pos is not None and self.last_time is not None:
            dt = now - self.last_time
            if dt > 0.01:
                self.velocity = (
                    (pos[0] - self.last_pos[0]) / dt,
                    (pos[1] - self.last_pos[1]) / dt
                )
        self.last_pos = pos
        self.last_time = now

    @staticmethod
    def _clamp(vx, vy, omega) -> Tuple[float, float, float]:
        """Clamp all values to [-1, 1]."""
        return (
            max(-1, min(1, vx)),
            max(-1, min(1, vy)),
            max(-1, min(1, omega))
        )


# ============================================================
# Utility
# ============================================================

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _angle_diff(target_deg: float, current_deg: float) -> float:
    diff = (target_deg - current_deg) % 360
    if diff > 180:
        diff -= 360
    return diff


# ============================================================
# Updated main loop example
# ============================================================

if __name__ == "__main__":
    """
    Example of how the reactive controller runs.
    This replaces the old A*-based main.py approach.
    """
    print("""
    Reactive Navigation Demo
    =========================
    This would normally run with real hardware. Here's what happens:

    1. Vision thread detects balls and robots every 20ms
    2. Main thread calls controller.tick() every 20ms
    3. tick() checks layers in priority order:
       - Emergency dodge?  → dodge
       - Time to deposit?  → return to net
       - Ball visible?     → pursue it
       - Nothing visible?  → explore
    4. Wall avoidance adjusts the output
    5. Command sent to Arduino

    No A* planning. No static maps. Just react to what you see.
    """)

    # Simulated example
    class FakeRobot:
        def __init__(self):
            self.x = 200.0
            self.y = 200.0
            self.heading = 0.0

    robot = FakeRobot()
    controller = ReactiveController(robot, zone="south", capacity=4)
    controller.start_match()

    # Simulate seeing some balls
    controller.update_vision(
        ball_detections=[
            {"x": 500, "y": 400, "type": BallType.PING_PONG, "confidence": 0.8},
            {"x": 800, "y": 600, "type": BallType.BALL_BEARING, "confidence": 0.6},
        ],
        robot_detections=[
            {"x": 700, "y": 300},  # Another robot nearby
        ]
    )

    # Run a few ticks
    for i in range(5):
        vx, vy, omega = controller.tick()
        print(f"Tick {i}: state={controller.state.value:15s} "
              f"vx={vx:+.2f} vy={vy:+.2f} omega={omega:+.2f}")

    print(f"\nTarget ball: ping-pong at (500, 400)")
    print(f"Avoiding robot at (700, 300)")
