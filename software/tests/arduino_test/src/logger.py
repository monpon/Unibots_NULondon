import serial
import keyboard

ser = serial.Serial('COM3', 9600)  # Change to your port

print("Use arrow keys to control. Press ESC to exit.")

while True:
    if keyboard.is_pressed('up'):
        ser.write(b'F')  # Forward
    elif keyboard.is_pressed('down'):
        ser.write(b'B')  # Backward
    elif keyboard.is_pressed('left'):
        ser.write(b'L')  # Left
    elif keyboard.is_pressed('right'):
        ser.write(b'R')  # Right
    elif keyboard.is_pressed('space'):
        ser.write(b'S')  # Stop
    elif keyboard.is_pressed('esc'):
        break