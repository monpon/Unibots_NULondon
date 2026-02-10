import cv2
import numpy as np
import math

prevCircle = None
circ1 = None
dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2
deltax = 0
deltay = 0
deltas = 0
frame_count = 0

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    # Process every other frame
    if frame_count % 2 != 0:
        if circ1 is not None:
            circ1[0] += deltax
            circ1[1] += deltay
            circ1[2] += deltas
        continue
    
    # Color masking
    lower_white = np.array([150,0,50])
    upper_white = np.array([200,200,150])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colormask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask = colormask)
    
    frame = res
    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (9,9), 0)
    
    # Hough circle detector
    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1, 300, param1=50, param2=20, minRadius=5, maxRadius=100)
    chosen = None
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            elif prevCircle is not None:
                if dist(i[0], i[1], prevCircle[0], prevCircle[1]) < dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]):
                    chosen = i
    
    if circ1 is not None:
        circ1[0] += deltax
        circ1[1] += deltay
        circ1[2] += deltas

        # Camera parameters
        camera_height = 150
        camera_tilt = 15
        horizontal_fov = 90
        vertical_fov = 50.625
        
        # Calculate angles (320x240 center is 160, 120)
        x_comp = circ1[0] - 160
        y_comp = circ1[1] - 120
        x_angle = x_comp * (horizontal_fov / 320)
        y_angle = y_comp * (vertical_fov / 240)
        
        # Calculate distances
        x_angle_rad = math.radians(x_angle)
        y_angle_rad = math.radians(y_angle)
        tilt_rad = math.radians(camera_tilt)
        effective_down_angle = tilt_rad + y_angle_rad
        
        if effective_down_angle > 0:
            forward_distance = camera_height / math.tan(effective_down_angle)
            lateral_distance = forward_distance * math.tan(x_angle_rad)
            
            print(f"X: {lateral_distance:.1f}mm, Y: {forward_distance:.1f}mm")

    if chosen is not None:
        circ1 = chosen
        prevCircle = chosen

cap.release()