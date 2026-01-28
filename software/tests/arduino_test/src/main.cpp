#include <Arduino.h>

// Motor A (Left)
const int ENA = 9;
const int IN1 = 8;
const int IN2 = 7;

// Motor B (Right)
const int ENB = 10;
const int IN3 = 6;
const int IN4 = 5;

// Forward declarations
void setMotorA(int speed, bool forward);
void setMotorB(int speed, bool forward);
void driveForward(int speed);
void driveBackward(int speed);
void turnLeft(int speed);
void turnRight(int speed);
void stopMotors();

void setup() {
  // Motor A pins
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  
  Serial.begin(9600);
  Serial.println("Motor Control Ready");
  Serial.println("Waiting for commands...");
  
  // Start with motors stopped
}

void setMotorA(int speed, bool forward) {
  if (forward) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  } else {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  }
  analogWrite(ENA, abs(speed));
}

void setMotorB(int speed, bool forward) {
  if (forward) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  analogWrite(ENB, abs(speed));
}

void driveForward(int speed) {
  Serial.println("Moving Forward");
  setMotorA(speed, true);
  setMotorB(speed, true);
}

void driveBackward(int speed) {
  Serial.println("Moving Backward");
  setMotorA(speed, false);
  setMotorB(speed, false);
}

void turnLeft(int speed) {
  Serial.println("Turning Left");
  setMotorA(speed, false);  // Left motor backward
  setMotorB(speed, true);   // Right motor forward
}

void turnRight(int speed) {
  Serial.println("Turning Right");
  setMotorA(speed, true);   // Left motor forward
  setMotorB(speed, false);  // Right motor backward
}

void stopMotors() {
  Serial.println("Motors Stopped");
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    // Default speed (adjust as needed, 0-255)
    int speed = 255;  // ~40% speed, increase if motors are too slow
    
    switch(command) {
      case 'F':
      case 'f':
        driveForward(speed);
        break;
        
      case 'B':
      case 'b':
        driveBackward(speed);
        break;
        
      case 'L':
      case 'l':
        turnLeft(speed);
        break;
        
      case 'R':
      case 'r':
        turnRight(speed);
        break;
        
      case 'S':
      case 's':
        stopMotors();
        break;
        
      default:
        // Ignore unknown commands
        break;
    }
  }
}