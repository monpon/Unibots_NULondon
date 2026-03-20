// =============================================================
// Arc Motion + Trapezoidal Profile + PID for 4 Encoder Motors
// =============================================================
// Motor: 12V DC 300RPM with Encoder
// Controller: Arduino (Mega recommended for 4 interrupt pins)
//
// Supports:
//   - Arc motion (turn + drive simultaneously)
//   - Straight line motion
//   - Point turn (rotate in place)
//   - Automatic selection based on angle
// =============================================================

#include <Arduino.h>

// ─── CONFIGURATION ───────────────────────────────────────────
#define NUM_MOTORS 4

// Encoder specs — ADJUST TO YOUR MOTOR DATASHEET
#define ENCODER_PPR        11
#define GEAR_RATIO         30
#define TICKS_PER_REV      (ENCODER_PPR * GEAR_RATIO * 2)

// Robot dimensions — MEASURE YOUR ROBOT
#define WHEEL_DIAMETER_MM  65.0   // Wheel diameter
#define WHEEL_CIRCUMF_MM   (PI * WHEEL_DIAMETER_MM)
#define TRACK_WIDTH_MM     200.0  // Center-to-center, left to right wheels

#define TICKS_PER_MM       (TICKS_PER_REV / WHEEL_CIRCUMF_MM)

// Motion limits — START CONSERVATIVE, INCREASE LATER
#define MAX_VELOCITY_TICKS 800.0   // ticks/sec
#define ACCELERATION_TICKS 400.0   // ticks/sec²

// Control loop
#define PID_LOOP_HZ        100
#define PID_DT             (1.0 / PID_LOOP_HZ)
#define POSITION_TOLERANCE 5       // ticks

// Arc vs point turn threshold (degrees)
#define ARC_MAX_ANGLE      120.0   // Above this, do point turn instead

// ─── PIN DEFINITIONS ─────────────────────────────────────────
struct MotorPins {
  uint8_t pwm;
  uint8_t dir1;
  uint8_t dir2;
  uint8_t encA;
  uint8_t encB;
};

//                        PWM  DIR1 DIR2 ENCA ENCB
MotorPins motorPins[NUM_MOTORS] = {
  { 5,  22, 23, 2,  26 },   // Motor 0 (front-left)
  { 6,  24, 25, 3,  27 },   // Motor 1 (front-right)
  { 7,  28, 29, 18, 30 },   // Motor 2 (rear-left)
  { 8,  31, 32, 19, 33 },   // Motor 3 (rear-right)
};

// ─── ENCODER STATE ───────────────────────────────────────────
volatile long encoderTicks[NUM_MOTORS] = {0, 0, 0, 0};

void encoderISR0() {
  encoderTicks[0] += digitalRead(motorPins[0].encB) ? -1 : 1;
}
void encoderISR1() {
  encoderTicks[1] += digitalRead(motorPins[1].encB) ? -1 : 1;
}
void encoderISR2() {
  encoderTicks[2] += digitalRead(motorPins[2].encB) ? -1 : 1;
}
void encoderISR3() {
  encoderTicks[3] += digitalRead(motorPins[3].encB) ? -1 : 1;
}

void (*encoderISRs[NUM_MOTORS])() = {
  encoderISR0, encoderISR1, encoderISR2, encoderISR3
};

// ─── PID CONTROLLER ──────────────────────────────────────────
struct PIDController {
  float Kp = 2.0;
  float Ki = 0.5;
  float Kd = 0.1;

  float integral    = 0.0;
  float prevError   = 0.0;
  float outputMin   = -255.0;
  float outputMax   =  255.0;
  float integralMax =  200.0;

  float compute(float error, float dt) {
    integral += error * dt;
    integral = constrain(integral, -integralMax, integralMax);
    float derivative = (dt > 0) ? (error - prevError) / dt : 0.0;
    prevError = error;
    float output = (Kp * error) + (Ki * integral) + (Kd * derivative);
    return constrain(output, outputMin, outputMax);
  }

  void reset() {
    integral  = 0.0;
    prevError = 0.0;
  }
};

PIDController pid[NUM_MOTORS];

// ─── TRAPEZOIDAL MOTION PROFILE ──────────────────────────────
struct TrapezoidalProfile {
  float maxVelocity;
  float acceleration;
  float accelDistance;
  float decelDistance;
  float totalDistance;
  float cruiseDistance;
  float accelTime;
  float cruiseTime;
  float decelTime;
  float totalTime;
  int   direction;       // +1 or -1

  void plan(float distTicks, float maxVel, float accel) {
    direction    = (distTicks >= 0) ? 1 : -1;
    totalDistance = fabs(distTicks);
    maxVelocity  = maxVel;
    acceleration = accel;

    accelTime    = maxVelocity / acceleration;
    accelDistance = 0.5 * acceleration * accelTime * accelTime;
    decelDistance = accelDistance;

    if (accelDistance + decelDistance > totalDistance) {
      // Triangular profile
      accelDistance  = totalDistance / 2.0;
      decelDistance  = totalDistance / 2.0;
      accelTime     = sqrt(2.0 * accelDistance / acceleration);
      decelTime     = accelTime;
      cruiseDistance = 0;
      cruiseTime    = 0;
      maxVelocity   = acceleration * accelTime;
    } else {
      // Trapezoidal profile
      cruiseDistance = totalDistance - accelDistance - decelDistance;
      cruiseTime    = cruiseDistance / maxVelocity;
      decelTime     = accelTime;
    }
    totalTime = accelTime + cruiseTime + decelTime;
  }

  // Target position at time t (signed)
  float getPosition(float t) {
    float pos;
    if (t <= 0) {
      pos = 0;
    } else if (t < accelTime) {
      pos = 0.5 * acceleration * t * t;
    } else if (t < accelTime + cruiseTime) {
      float tc = t - accelTime;
      pos = accelDistance + maxVelocity * tc;
    } else if (t < totalTime) {
      float td = t - accelTime - cruiseTime;
      pos = accelDistance + cruiseDistance
            + maxVelocity * td
            - 0.5 * acceleration * td * td;
    } else {
      pos = totalDistance;
    }
    return pos * direction;
  }

  bool isComplete(float t) {
    return t >= totalTime;
  }
};

// ─── MOTOR OUTPUT ────────────────────────────────────────────
void setMotorPWM(uint8_t i, int pwm) {
  MotorPins &m = motorPins[i];
  if (pwm > 0) {
    digitalWrite(m.dir1, HIGH);
    digitalWrite(m.dir2, LOW);
    analogWrite(m.pwm, constrain(pwm, 0, 255));
  } else if (pwm < 0) {
    digitalWrite(m.dir1, LOW);
    digitalWrite(m.dir2, HIGH);
    analogWrite(m.pwm, constrain(-pwm, 0, 255));
  } else {
    digitalWrite(m.dir1, LOW);
    digitalWrite(m.dir2, LOW);
    analogWrite(m.pwm, 0);
  }
}

void stopAllMotors() {
  for (int i = 0; i < NUM_MOTORS; i++) setMotorPWM(i, 0);
}

void resetEncodersAndPIDs() {
  noInterrupts();
  for (int i = 0; i < NUM_MOTORS; i++) {
    encoderTicks[i] = 0;
    pid[i].reset();
  }
  interrupts();
}

// ─── CORE DUAL-PROFILE EXECUTOR ──────────────────────────────
// Runs two time-synchronized profiles:
//   leftProfile  → motors 0, 2
//   rightProfile → motors 1, 3
// Both profiles share the SAME total time so they start and
// stop together, even though distances may differ.

bool executeDualProfile(TrapezoidalProfile &leftProf,
                        TrapezoidalProfile &rightProf) {

  float totalTime = max(leftProf.totalTime, rightProf.totalTime);

  Serial.print("Executing move: ");
  Serial.print(totalTime, 2);
  Serial.println(" sec");
  Serial.print("  Left dist:  ");
  Serial.print(leftProf.totalDistance * leftProf.direction);
  Serial.print(" ticks | Right dist: ");
  Serial.println(rightProf.totalDistance * rightProf.direction);

  resetEncodersAndPIDs();

  unsigned long startTime = millis();
  unsigned long loopInterval = 1000 / PID_LOOP_HZ;
  unsigned long nextLoop = startTime;

  while (true) {
    unsigned long now = millis();
    if (now < nextLoop) continue;
    nextLoop += loopInterval;

    float t = (now - startTime) / 1000.0;

    // Get target positions from each profile
    float leftTarget  = leftProf.getPosition(t);
    float rightTarget = rightProf.getPosition(t);

    // PID for left motors (0, 2)
    for (int i = 0; i < NUM_MOTORS; i += 2) {
      long curr;
      noInterrupts();
      curr = encoderTicks[i];
      interrupts();
      float err = leftTarget - (float)curr;
      setMotorPWM(i, (int)pid[i].compute(err, PID_DT));
    }

    // PID for right motors (1, 3)
    for (int i = 1; i < NUM_MOTORS; i += 2) {
      long curr;
      noInterrupts();
      curr = encoderTicks[i];
      interrupts();
      float err = rightTarget - (float)curr;
      setMotorPWM(i, (int)pid[i].compute(err, PID_DT));
    }

    // Check completion
    if (t >= totalTime) {
      delay(50);  // Settling time

      bool settled = true;
      float leftFinal  = leftProf.totalDistance * leftProf.direction;
      float rightFinal = rightProf.totalDistance * rightProf.direction;

      for (int i = 0; i < NUM_MOTORS; i += 2) {
        if (fabs(leftFinal - encoderTicks[i]) > POSITION_TOLERANCE)
          settled = false;
      }
      for (int i = 1; i < NUM_MOTORS; i += 2) {
        if (fabs(rightFinal - encoderTicks[i]) > POSITION_TOLERANCE)
          settled = false;
      }

      if (settled || t > totalTime + 1.0) break;
    }

    // Safety timeout
    if (t > totalTime * 2.0 + 2.0) {
      Serial.println("ERROR: Move timeout!");
      break;
    }
  }

  stopAllMotors();

  Serial.println("Move complete:");
  for (int i = 0; i < NUM_MOTORS; i++) {
    Serial.print("  Motor ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(encoderTicks[i]);
  }
  return true;
}

// ─── TIME SYNCHRONIZATION HELPER ─────────────────────────────
// Given two distances, compute profiles that take the SAME
// total time. The shorter side gets a lower max velocity
// so both finish together.

void planSynchronized(TrapezoidalProfile &prof,
                      float distTicks, float totalTime) {
  // We need to find the max velocity that makes this profile
  // fit exactly in totalTime. For a symmetric trapezoid:
  //   totalTime = 2 * accelTime + cruiseTime
  //   distance  = maxVel * (totalTime - accelTime)
  // With accelTime = maxVel / accel:
  //   distance = maxVel * (totalTime - maxVel/accel)
  // Solving: accel * totalTime * maxVel - maxVel^2 = accel * dist
  // Quadratic in maxVel.

  float a = 1.0;
  float b = -(ACCELERATION_TICKS * totalTime);
  float c = ACCELERATION_TICKS * fabs(distTicks);
  float discriminant = b * b - 4 * a * c;

  if (discriminant < 0) {
    // Fallback: just use regular planning
    prof.plan(distTicks, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);
    return;
  }

  float maxVel = (-b - sqrt(discriminant)) / (2 * a);
  maxVel = min(maxVel, MAX_VELOCITY_TICKS);

  prof.plan(distTicks, maxVel, ACCELERATION_TICKS);
}

// =============================================================
// PUBLIC MOVE FUNCTIONS
// =============================================================

// ─── ARC MOTION ──────────────────────────────────────────────
// Drives an arc that turns angleDeg while moving forward.
// angleDeg > 0 = turn right, angleDeg < 0 = turn left
// forwardMM = chord-like forward distance (straight-line
//             component toward the target)
//
// The arc radius is computed from the angle and forward
// distance, then each wheel's arc length is derived from it.

bool moveArc(float angleDeg, float forwardMM) {
  if (fabs(angleDeg) < 1.0) {
    // Basically straight — avoid division by zero
    return moveStraight(forwardMM);
  }

  float angleRad = angleDeg * PI / 180.0;

  // Compute arc radius from forward distance and angle
  // Forward distance ≈ chord, so: R = forward / (2 * sin(angle/2))
  // For the center path of the robot.
  float radius = fabs(forwardMM / (2.0 * sin(angleRad / 2.0)));

  // Arc length along the robot center
  float centerArc = radius * fabs(angleRad);

  // Each side's arc length depends on which way we turn
  float leftArc, rightArc;
  float halfTrack = TRACK_WIDTH_MM / 2.0;

  if (angleDeg > 0) {
    // Turning right: left is outer (longer), right is inner
    leftArc  = (radius + halfTrack) * fabs(angleRad);
    rightArc = (radius - halfTrack) * fabs(angleRad);
  } else {
    // Turning left: right is outer (longer), left is inner
    leftArc  = (radius - halfTrack) * fabs(angleRad);
    rightArc = (radius + halfTrack) * fabs(angleRad);
  }

  // Handle case where inner radius < 0 (very tight turn)
  // In this case inner wheels go backward
  // The sign is already handled naturally by the math

  float leftTicks  = leftArc  * TICKS_PER_MM;
  float rightTicks = rightArc * TICKS_PER_MM;

  Serial.println("=== ARC MOTION ===");
  Serial.print("  Angle: ");
  Serial.print(angleDeg);
  Serial.print(" deg, Forward: ");
  Serial.print(forwardMM);
  Serial.println(" mm");
  Serial.print("  Radius: ");
  Serial.print(radius);
  Serial.print(" mm, Center arc: ");
  Serial.print(centerArc);
  Serial.println(" mm");
  Serial.print("  Left arc: ");
  Serial.print(leftArc);
  Serial.print(" mm, Right arc: ");
  Serial.print(rightArc);
  Serial.println(" mm");

  // Plan the longer side at full speed
  TrapezoidalProfile leftProf, rightProf;

  if (fabs(leftTicks) >= fabs(rightTicks)) {
    leftProf.plan(leftTicks, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);
    planSynchronized(rightProf, rightTicks, leftProf.totalTime);
  } else {
    rightProf.plan(rightTicks, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);
    planSynchronized(leftProf, leftTicks, rightProf.totalTime);
  }

  return executeDualProfile(leftProf, rightProf);
}

// ─── STRAIGHT LINE MOTION ────────────────────────────────────
bool moveStraight(float distanceMM) {
  float ticks = distanceMM * TICKS_PER_MM;

  Serial.println("=== STRAIGHT MOTION ===");
  Serial.print("  Distance: ");
  Serial.print(distanceMM);
  Serial.println(" mm");

  TrapezoidalProfile leftProf, rightProf;
  leftProf.plan(ticks, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);
  rightProf.plan(ticks, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);

  return executeDualProfile(leftProf, rightProf);
}

// ─── POINT TURN (ROTATE IN PLACE) ───────────────────────────
// angleDeg > 0 = turn right, < 0 = turn left
bool pointTurn(float angleDeg) {
  float angleRad = fabs(angleDeg) * PI / 180.0;
  float arcPerSide = (TRACK_WIDTH_MM / 2.0) * angleRad;
  float ticksPerSide = arcPerSide * TICKS_PER_MM;

  Serial.println("=== POINT TURN ===");
  Serial.print("  Angle: ");
  Serial.print(angleDeg);
  Serial.println(" deg");

  TrapezoidalProfile leftProf, rightProf;

  if (angleDeg > 0) {
    // Turn right: left forward, right backward
    leftProf.plan(ticksPerSide, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);
    rightProf.plan(-ticksPerSide, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);
  } else {
    // Turn left: left backward, right forward
    leftProf.plan(-ticksPerSide, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);
    rightProf.plan(ticksPerSide, MAX_VELOCITY_TICKS, ACCELERATION_TICKS);
  }

  return executeDualProfile(leftProf, rightProf);
}

// ─── SMART MOVE: AUTO-SELECT BEST STRATEGY ───────────────────
// Given an angle to turn and distance to travel, automatically
// picks the best approach:
//   - Small angle  → arc motion (fast, smooth)
//   - Large angle  → point turn first, then straight
//   - Zero angle   → straight line
//   - Zero distance → point turn

bool moveToTarget(float angleDeg, float distanceMM) {
  Serial.println("============================");
  Serial.print("Target: ");
  Serial.print(angleDeg);
  Serial.print(" deg, ");
  Serial.print(distanceMM);
  Serial.println(" mm");

  if (fabs(distanceMM) < 5.0) {
    // Just need to rotate
    return pointTurn(angleDeg);
  }

  if (fabs(angleDeg) < 1.0) {
    // Basically straight ahead
    return moveStraight(distanceMM);
  }

  if (fabs(angleDeg) <= ARC_MAX_ANGLE) {
    // Arc motion — turn and drive simultaneously
    Serial.println("Strategy: ARC MOTION");
    return moveArc(angleDeg, distanceMM);
  }

  // Large angle — point turn first, then drive straight
  Serial.println("Strategy: POINT TURN + STRAIGHT");
  pointTurn(angleDeg);
  delay(100);  // Brief pause between moves
  return moveStraight(distanceMM);
}

// ─── SETUP ───────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("Arc Motion Controller Init...");

  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(motorPins[i].pwm,  OUTPUT);
    pinMode(motorPins[i].dir1, OUTPUT);
    pinMode(motorPins[i].dir2, OUTPUT);
    pinMode(motorPins[i].encA, INPUT_PULLUP);
    pinMode(motorPins[i].encB, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(motorPins[i].encA),
                    encoderISRs[i], CHANGE);
  }

  stopAllMotors();
  Serial.println("Ready. Commands:");
  Serial.println("  arc [angle] [distance]  - arc motion");
  Serial.println("  go [distance]           - straight");
  Serial.println("  turn [angle]            - point turn");
  Serial.println("  move [angle] [distance] - auto-select");
  Serial.println("  stop                    - emergency stop");
}

// ─── MAIN LOOP ───────────────────────────────────────────────
void loop() {
  // =====================================================
  // INTEGRATE WITH YOUR NAVIGATION CODE:
  //
  //   float angle = getHeadingError();    // degrees
  //   float dist  = getDistanceToTarget(); // mm
  //   moveToTarget(angle, dist);
  //
  // That's it! moveToTarget() handles everything.
  // =====================================================

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("arc ")) {
      int sp = cmd.indexOf(' ', 4);
      float angle = cmd.substring(4, sp).toFloat();
      float dist  = cmd.substring(sp + 1).toFloat();
      moveArc(angle, dist);
    }
    else if (cmd.startsWith("go ")) {
      float dist = cmd.substring(3).toFloat();
      moveStraight(dist);
    }
    else if (cmd.startsWith("turn ")) {
      float angle = cmd.substring(5).toFloat();
      pointTurn(angle);
    }
    else if (cmd.startsWith("move ")) {
      int sp = cmd.indexOf(' ', 5);
      float angle = cmd.substring(5, sp).toFloat();
      float dist  = cmd.substring(sp + 1).toFloat();
      moveToTarget(angle, dist);
    }
    else if (cmd == "stop") {
      stopAllMotors();
    }
  }
}