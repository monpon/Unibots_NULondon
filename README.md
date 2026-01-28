# Unibots UK Competition Robot

An autonomous robot system designed for the Unibots UK competition, featuring mecanum drive omnidirectional movement, AprilTag-based localization, and autonomous ball collection and launching capabilities.

## Competition Overview

The robot competes in a timed arena-based challenge to:
- Collect magnetic steel ball bearings
- Launch ping-pong balls into scoring nets
- Navigate autonomously with collision avoidance
- Operate within strict size constraints

## System Architecture

### Hardware Components

**Main Controller**
- Raspberry Pi 3 - Primary processing unit

**Motor Control**
- Arduino Uno - Motor control interface with FPID control loops
- 4x Motor encoders - Precise speed feedback for mecanum drive
- ESC units - Electronic speed controllers
- Mecanum wheel drive system - Omnidirectional movement

**Vision System**
- 2x USB webcams:
  - Camera 1: AprilTag localization
  - Camera 2: Ball/gamepiece detection
- AprilTag markers - Localization reference points

**Actuators**
- Ball collection mechanism
- Ping-pong ball launcher

### Software Stack

**Control Architecture**
```
Raspberry Pi 3 (Python)
    ↓ Position Targets via Serial
Arduino Uno (C++)
    ├─ FPID Control Loops (4 motors)
    ├─ Encoder Feedback
    ↓ PWM Signals
ESC Controllers
    ↓ Power
Brushless/DC Motors with Encoders
```

**Key Features**
- Dual camera system: AprilTag localization and object detection
- Encoder-based FPID control for precise mecanum drive
- Distributed control: Pi handles navigation, Arduino handles drive control
- Real-time position targeting with continuous updates
- Autonomous path planning and obstacle avoidance

## Getting Started

### Prerequisites

```bash
# Raspberry Pi dependencies
sudo apt-get update
sudo apt-get install python3-opencv python3-numpy
pip3 install apriltag pyserial

# Arduino IDE for motor control firmware
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd unibots-robot/software

# Install Python dependencies
pip3 install -r requirements.txt

# Upload Arduino firmware
# Open arduino/motor_control/motor_control.ino in Arduino IDE
# Select Arduino Uno and upload
```

### Configuration

1. **Camera Calibration**: Calibrate each camera for lens distortion
2. **AprilTag Mapping**: Record arena AprilTag positions
3. **Encoder Calibration**: Calibrate encoder counts per revolution
4. **FPID Tuning**: Tune FPID parameters for each motor
5. **Serial Protocol**: Configure position target communication format

## Project Structure

```
unibots-robot/
├── README.md
├── software/
│   ├── requirements.txt
│   ├── config/
│   │   ├── camera_params.yaml
│   │   ├── apriltag_positions.yaml
│   │   ├── motor_calibration.yaml
│   │   └── fpid_params.yaml
│   ├── src/
│   │   ├── main.py
│   │   ├── vision/
│   │   │   ├── apriltag_detector.py
│   │   │   ├── camera_manager.py
│   │   │   └── localization.py
│   │   ├── navigation/
│   │   │   ├── path_planner.py
│   │   │   ├── obstacle_avoidance.py
│   │   │   └── position_targeting.py
│   │   ├── control/
│   │   │   └── arduino_interface.py
│   │   └── utils/
│   │       └── geometry.py
│   ├── arduino/
│   │   └── motor_control/
│   │       ├── motor_control.ino
│   │       ├── fpid.h
│   │       └── encoder.h
│   ├── tests/
│   │   └── test_localization.py
│   ├── tools/
│   │   ├── calibrate_cameras.py
│   │   ├── calibrate_motors.py
│   │   └── tune_fpid.py
│   └── docs/
│       ├── architecture.md
│       ├── serial_protocol.md
│       ├── calibration_guide.md
│       ├── fpid_tuning.md
│       └── competition_rules.md
└── hardware/
    ├── chassis/
    ├── motor_mounts/
    ├── camera_mounts/
    ├── ball_collector/
    ├── lifter/
    └── assembly/
```

## Usage

### Running the Robot

```bash
# Navigate to software directory
cd software

# Start the main control system
python3 src/main.py

# Run with debug visualization
python3 src/main.py --debug

# Test individual components
python3 tests/test_localization.py
```

### Calibration

```bash
# Camera calibration
python3 tools/calibrate_cameras.py

# Motor calibration
python3 tools/calibrate_motors.py

# FPID tuning
python3 tools/tune_fpid.py
```

## Development

### Key Design Decisions

- **Mecanum Drive**: Chosen for omnidirectional movement and precise positioning
- **Encoder-Based Control**: Four motor encoders provide precise speed feedback for synchronized mecanum drive
- **FPID Control**: Arduino handles real-time FPID loops for each motor, ensuring accurate speed matching
- **Distributed Architecture**: Raspberry Pi 3 handles high-level navigation and sends position targets; Arduino Uno executes low-level drive control
- **Camera-Based Odometry**: AprilTag localization with dedicated camera for position tracking

### Known Issues

- Motor selection: Need motors with integrated or attachable encoders
- Arduino Uno memory constraints with FPID for 4 motors + encoder processing
- Serial communication latency for position target updates
- Camera processing optimization needed for real-time performance with dual cameras

## Competition Constraints

- Initial size limitations must be met
- Full autonomy required (no remote control)
- Collision avoidance mandatory
- AprilTags positioned around arena walls for localization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Contact

Monesh - moneshpon@gmail.com