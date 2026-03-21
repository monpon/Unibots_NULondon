#include <Arduino.h>

// ============================================================
//  PIN CONFIGURATION — 4 independent motors
//  Assumes two L298N boards (or equivalent 4-channel driver)
// ============================================================

// Front-Left motor
const int EN_FL = 9;
const int IN1_FL = 8;
const int IN2_FL = 7;

// Front-Right motor
const int EN_FR = 10;
const int IN1_FR = 6;
const int IN2_FR = 5;

// Rear-Left motor
const int EN_RL = 3;
const int IN1_RL = 4;
const int IN2_RL = 2;

// Rear-Right motor
const int EN_RR = 11;
const int IN1_RR = 12;
const int IN2_RR = 13;

// ============================================================
//  STATE  (current power per wheel, -255 to +255)
// ============================================================

int pwrFL = 0;
int pwrFR = 0;
int pwrRL = 0;
int pwrRR = 0;

// ============================================================
//  SERIAL PROTOCOL  (115200 baud, newline-terminated)
//
//  M <FL> <FR> <RL> <RR>\n
//      Set each wheel independently.
//      Values: -255 to 255  (negative = backward)
//      Example: "M 200 200 200 200\n"       → forward
//      Example: "M 200 -200 -200 200\n"     → strafe right
//      Example: "M -150 150 -150 150\n"     → rotate CW
//
//  S\n
//      Hard stop all wheels.
//
//  P\n
//      Query — returns current power of all four wheels.
//
//  Arduino replies:
//      "OK <FL> <FR> <RL> <RR>\n"   after M / S
//      "P <FL> <FR> <RL> <RR>\n"    after P
//      "ERR <reason>\n"             on bad input
// ============================================================

// ============================================================
//  SINGLE-MOTOR HELPER
// ============================================================

void applyMotor(int enPin, int in1Pin, int in2Pin, int power)
{
    power = constrain(power, -255, 255);

    if (power > 0)
    {
        digitalWrite(in1Pin, LOW);
        digitalWrite(in2Pin, HIGH);
        analogWrite(enPin, power);
    }
    else if (power < 0)
    {
        digitalWrite(in1Pin, HIGH);
        digitalWrite(in2Pin, LOW);
        analogWrite(enPin, -power);
    }
    else
    { // 0 → active brake
        digitalWrite(in1Pin, LOW);
        digitalWrite(in2Pin, LOW);
        analogWrite(enPin, 0);
    }
}

// ============================================================
//  APPLY ALL FOUR WHEELS
// ============================================================

void applyAll(int fl, int fr, int rl, int rr)
{
    pwrFL = constrain(fl, -255, 255);
    pwrFR = constrain(fr, -255, 255);
    pwrRL = constrain(rl, -255, 255);
    pwrRR = constrain(rr, -255, 255);

    applyMotor(EN_FL, IN1_FL, IN2_FL, pwrFL);
    applyMotor(EN_FR, IN1_FR, IN2_FR, pwrFR);
    applyMotor(EN_RL, IN1_RL, IN2_RL, pwrRL);
    applyMotor(EN_RR, IN1_RR, IN2_RR, pwrRR);
}

void stopMotors()
{
    applyAll(0, 0, 0, 0);
}

// ============================================================
//  SERIAL RESPONSES
// ============================================================

void sendOK()
{
    Serial.print("OK ");
    Serial.print(pwrFL);
    Serial.print(" ");
    Serial.print(pwrFR);
    Serial.print(" ");
    Serial.print(pwrRL);
    Serial.print(" ");
    Serial.println(pwrRR);
}

void sendPower()
{
    Serial.print("P ");
    Serial.print(pwrFL);
    Serial.print(" ");
    Serial.print(pwrFR);
    Serial.print(" ");
    Serial.print(pwrRL);
    Serial.print(" ");
    Serial.println(pwrRR);
}

// ============================================================
//  COMMAND PARSING
// ============================================================

char buf[48];
uint8_t bufIdx = 0;

void processCommand(const char *cmd)
{
    // ---- STOP ----
    if (cmd[0] == 'S' || cmd[0] == 's')
    {
        stopMotors();
        sendOK();
        return;
    }

    // ---- QUERY ----
    if (cmd[0] == 'P' || cmd[0] == 'p')
    {
        sendPower();
        return;
    }

    // ---- MOTOR SET ----
    if (cmd[0] == 'M' || cmd[0] == 'm')
    {
        int fl = 0, fr = 0, rl = 0, rr = 0;
        int parsed = sscanf(cmd + 1, "%d %d %d %d", &fl, &fr, &rl, &rr);
        if (parsed == 4)
        {
            applyAll(fl, fr, rl, rr);
            sendOK();
        }
        else
        {
            Serial.println("ERR need 4 values: FL FR RL RR");
        }
        return;
    }

    Serial.println("ERR unknown cmd");
}

// ============================================================
//  PIN SETUP HELPER
// ============================================================

void setupMotorPins(int en, int in1, int in2)
{
    pinMode(en, OUTPUT);
    pinMode(in1, OUTPUT);
    pinMode(in2, OUTPUT);
}

// ============================================================
//  SETUP / LOOP
// ============================================================

void setup()
{
    setupMotorPins(EN_FL, IN1_FL, IN2_FL);
    setupMotorPins(EN_FR, IN1_FR, IN2_FR);
    setupMotorPins(EN_RL, IN1_RL, IN2_RL);
    setupMotorPins(EN_RR, IN1_RR, IN2_RR);

    stopMotors();

    Serial.begin(115200);
    Serial.println("READY 4WD");
}

void loop()
{
    while (Serial.available())
    {
        char c = Serial.read();

        if (c == '\n' || c == '\r')
        {
            if (bufIdx > 0)
            {
                buf[bufIdx] = '\0';
                processCommand(buf);
                bufIdx = 0;
            }
        }
        else if (bufIdx < sizeof(buf) - 1)
        {
            buf[bufIdx++] = c;
        }
    }
}