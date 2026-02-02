import serial
import subprocess
import platform
import time

# Adjust this to your Arduino's port
# Windows: "COM3", "COM4", etc.
# Mac: "/dev/cu.usbmodem14101" or similar
# Linux: "/dev/ttyUSB0" or "/dev/ttyACM0"
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

# Distance thresholds and actions
ACTIONS = {
    4: r"C:\Users\lukas\OneDrive\Pictures\Cyberpunk 2077\photomode_12102022_171830.png",
    10: r"C:\Users\lukas\OneDrive\Pictures\Cyberpunk 2077\photomode_18102022_155908.png",
    15: r"C:\Users\lukas\OneDrive\Pictures\Cyberpunk 2077\photomode_18102022_165425.png",
}

TOLERANCE = 1  # cm tolerance for triggering
COOLDOWN = 2   # seconds before same action can trigger again

last_triggered = {}

def open_file(filepath):
    """Open a file with the default application"""
    system = platform.system()
    if system == "Windows":
        subprocess.run(["start", "", filepath], shell=True)
    elif system == "Darwin":  # macOS
        subprocess.run(["open", filepath])
    else:  # Linux
        subprocess.run(["xdg-open", filepath])

def main():
    print(f"Connecting to {SERIAL_PORT}...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        print("Connected! Monitoring distance...\n")
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                
                try:
                    distance = int(line)
                    print(f"Distance: {distance} cm", end="\r")
                    
                    # Check each action threshold
                    for threshold, filepath in ACTIONS.items():
                        if abs(distance - threshold) <= TOLERANCE:
                            current_time = time.time()
                            
                            # Check cooldown
                            if threshold not in last_triggered or \
                               current_time - last_triggered[threshold] > COOLDOWN:
                                print(f"\n→ Triggered at {distance}cm: Opening {filepath}")
                                open_file(filepath)
                                last_triggered[threshold] = current_time
                                
                except ValueError:
                    pass  # Ignore non-numeric lines
                    
    except serial.SerialException as e:
        print(f"Error: {e}")
        print("Check that the Arduino is connected and the port is correct.")

if __name__ == "__main__":
    main()