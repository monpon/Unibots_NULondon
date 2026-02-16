"""
Main Robot Controller - Unibots Competition
=============================================
Integrates vision, navigation, and motor control into a single loop.

This is the entry point you run on the Raspberry Pi when the match starts.

Usage:
    python3 main.py --zone north --serial /dev/ttyUSB0 --camera 0
"""

import argparse
import time
import threading
import math
from vision import VisionSystem, DetectedObjectType
from navigation import (
    GameController, Ball, BallType, CollisionAvoidance,
    SCORING_ZONES
)
from mecanum_robot import MecanumRobot


class UnibotMain:
    """
    Top-level controller that runs everything.

    Architecture:
        - Vision thread: captures frames, detects balls/robots, localizes
        - Main thread: runs game strategy, navigation, motor commands
        - Collision avoidance: filters motor commands before sending
    """

    def __init__(self, zone: str, serial_port: str, camera_index: int):
        print(f"Initializing Unibot — Zone: {zone}")

        # Motor control
        self.robot = MecanumRobot(serial_port=serial_port)

        # Vision
        self.vision = VisionSystem(camera_index=camera_index)

        # Navigation + game strategy
        self.game = GameController(self.robot, zone=zone)

        # Collision avoidance (wraps motor commands)
        self.collision_avoidance = CollisionAvoidance()

        # Shared state (updated by vision thread)
        self.latest_balls = []
        self.latest_obstacles = []
        self.pose_valid = False
        self.running = False

        # Override the robot's Arduino send to inject collision avoidance
        self._original_send = self.robot.arduino.send_velocity
        self.robot.arduino.send_velocity = self._safe_send_velocity

    def _safe_send_velocity(self, vx, vy, omega):
        """
        Wrapper that applies collision avoidance before sending
        velocity commands to the Arduino.
        """
        if self.pose_valid and self.latest_obstacles:
            vx, vy, omega = self.collision_avoidance.adjust_velocity(
                vx, vy, omega,
                (self.robot.x, self.robot.y),
                self.robot.heading,
                self.latest_obstacles
            )
        return self._original_send(vx, vy, omega)

    def _vision_loop(self):
        """Runs in a separate thread. Updates pose, balls, and obstacles."""
        self.vision.calibrate_floor()

        while self.running:
            result = self.vision.process_frame(debug=False)

            # Update robot pose
            if result["pose"]:
                x, y, heading = result["pose"]
                self.robot.update_pose(x, y, heading)
                self.pose_valid = True
            else:
                self.pose_valid = False

            # Update detected balls (convert to navigation Ball objects)
            nav_balls = []
            for det in result["balls"]:
                if det.confidence < 0.4:
                    continue
                if det.obj_type == DetectedObjectType.PING_PONG:
                    ball_type = BallType.PING_PONG
                else:
                    ball_type = BallType.BALL_BEARING
                nav_balls.append(Ball(
                    x=det.world_x,
                    y=det.world_y,
                    ball_type=ball_type
                ))
            self.latest_balls = nav_balls

            # Update obstacle positions
            self.latest_obstacles = result["obstacles"]

            # Feed into game controller
            self.game.update_balls(self.latest_balls)
            self.game.update_obstacles(self.latest_obstacles)

            time.sleep(0.02)  # ~50 Hz

    def run(self):
        """
        Main entry point. Call this when the match starts
        (i.e., when you press the physical start button).
        """
        self.running = True

        # Start vision in background thread
        vision_thread = threading.Thread(
            target=self._vision_loop, daemon=True
        )
        vision_thread.start()

        # Wait for first valid pose
        print("Waiting for localization...")
        timeout = time.time() + 10
        while not self.pose_valid and time.time() < timeout:
            time.sleep(0.1)

        if not self.pose_valid:
            print("WARNING: No AprilTags detected. Running without localization.")

        # Run the game strategy
        print("Starting match!")
        try:
            self.game.run()
        except Exception as e:
            print(f"Error during match: {e}")
        finally:
            self.stop()

    def stop(self):
        """Emergency stop everything."""
        self.running = False
        self.robot.stop()
        self.vision.release()
        self.robot.shutdown()
        print("Robot stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unibots Robot Controller")
    parser.add_argument("--zone", type=str, default="north",
                        choices=["north", "south", "east", "west"],
                        help="Your scoring zone")
    parser.add_argument("--serial", type=str, default="/dev/ttyUSB0",
                        help="Arduino serial port")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    args = parser.parse_args()

    bot = UnibotMain(
        zone=args.zone,
        serial_port=args.serial,
        camera_index=args.camera
    )

    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nMatch interrupted")
    finally:
        bot.stop()
