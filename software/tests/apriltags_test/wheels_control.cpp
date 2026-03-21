#include <PinChangeInterrupt.h>

volatile long count1 = 0;
volatile long count2 = 0;

// Motor 1 pins
const int IN1 = 8;
const int IN2 = 9;
const int ENA = 10;

// Motor 2 pins
const int IN3 = 11;
const int IN4 = 12;
const int ENB = 6;

// Encoder 1 - hardware interrupts
const int ENC1A = 2;
const int ENC1B = 3;

// Encoder 2 - pin change interrupts on analog pins
const int ENC2A = A0;
const int ENC2B = A1;

unsigned long lastReport = 0;

void setup() {
  TCCR2B = TCCR2B & B11111000 | B00000001;
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT); pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT); pinMode(ENB, OUTPUT);

  pinMode(ENC1A, INPUT_PULLUP);
  pinMode(ENC1B, INPUT_PULLUP);
  pinMode(ENC2A, INPUT_PULLUP);
  pinMode(ENC2B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC1A), isr1A, RISING);
  attachPCINT(digitalPinToPCINT(ENC2A), isr2A, RISING);

  Serial.begin(115200);
}

// Motor 1 quadrature
void isr1A() {
  if (digitalRead(ENC1B)) count1++;
  else count1--;
}
void isr1B() {} // unused

// Motor 2 quadrature
void isr2A() {
  if (digitalRead(ENC2B)) count2++;
  else count2--;
}
void isr2B() {} // unused

void driveMotor(int in1, int in2, int en, int pwm) {
  if (pwm > 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(en, constrain(pwm, 0, 255));
  } else if (pwm < 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(en, constrain(-pwm, 0, 255));
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(en, 0);
  }
}

void parseSerial() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();
    int m1idx = msg.indexOf("M1:");
    int m2idx = msg.indexOf("M2:");
    if (m1idx != -1) {
      int end = msg.indexOf(',', m1idx);
      int val = msg.substring(m1idx + 3, end == -1 ? msg.length() : end).toInt();
      driveMotor(IN1, IN2, ENA, val);
    }
    if (m2idx != -1) {
      int val = msg.substring(m2idx + 3).toInt();
      driveMotor(IN3, IN4, ENB, val);
    }
  }
}

void loop() {
  parseSerial();

  if (millis() - lastReport >= 20) {
    noInterrupts();
    long c1 = count1;
    long c2 = count2;
    interrupts();
    Serial.print("T:");
    Serial.print(c1);
    Serial.print(",");
    Serial.println(c2);
    lastReport = millis();
  }
}