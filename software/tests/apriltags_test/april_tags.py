import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

CAMERA_FOV_HORIZONTAL = 117  
TAG_SIZE_MM = 100  

# Capture at full resolution
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

ret, test_frame = cap.read()
if ret:
    frame_height, frame_width = test_frame.shape[:2]
    print(f"Capture resolution: {frame_width}x{frame_height}")
else:
    frame_width, frame_height = 2560, 1440

# Display resolution (720p)
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Calculate scale factor
scale_factor = DISPLAY_WIDTH / frame_width

camera_center_x = frame_width // 2
camera_center_y = frame_height // 2

focal_length_pixels = (frame_width / 2) / np.tan(np.radians(CAMERA_FOV_HORIZONTAL / 2))

print(f"Camera center: ({camera_center_x}, {camera_center_y})")
print(f"Focal length: {focal_length_pixels:.2f} pixels")
print(f"Horizontal FOV: {CAMERA_FOV_HORIZONTAL}°")
print(f"Display resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print(f"Scale factor: {scale_factor:.3f}")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    cv2.line(frame, (camera_center_x - 20, camera_center_y), 
             (camera_center_x + 20, camera_center_y), (255, 255, 255), 1)
    cv2.line(frame, (camera_center_x, camera_center_y - 20), 
             (camera_center_x, camera_center_y + 20), (255, 255, 255), 1)
    
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
        
        for i in range(len(ids)):
            tag_id = ids[i][0]
            marker_corners = corners[i][0]
            
            tag_center_x = int(marker_corners[:, 0].mean())
            tag_center_y = int(marker_corners[:, 1].mean())
            
            cv2.circle(frame, (tag_center_x, tag_center_y), 5, (0, 0, 255), -1)
            
            pixel_offset_x = tag_center_x - camera_center_x
            pixel_offset_y = tag_center_y - camera_center_y
            
            left_x = (marker_corners[0][0] + marker_corners[3][0]) / 2
            right_x = (marker_corners[1][0] + marker_corners[2][0]) / 2
            tag_width_pixels = abs(right_x - left_x)
            
            distance_mm = (TAG_SIZE_MM * focal_length_pixels) / tag_width_pixels
            distance_cm = distance_mm / 10
            
            angle_x = CAMERA_FOV_HORIZONTAL/1080 * pixel_offset_x 
            angle_y = CAMERA_FOV_HORIZONTAL/1080 * pixel_offset_y
            
            x_offset_mm = distance_mm * np.tan(np.radians(angle_x))
            y_offset_mm = distance_mm * np.tan(np.radians(angle_y))
            
            print(f"Tag ID {tag_id:2d} | "
                  f"Distance: {distance_cm:6.1f} cm | "
                  f"X offset: {x_offset_mm:+7.1f} mm | "
                  f"Y offset: {y_offset_mm:+7.1f} mm | "
                  f"Angle X: {angle_x:+6.2f}° | "
                  f"Angle Y: {angle_y:+6.2f}°")
            
            bottom_y = int(marker_corners[:, 1].max()) + 20
            center_x = tag_center_x
            
            text1 = f"ID: {tag_id}"
            cv2.putText(frame, text1, (center_x - 40, bottom_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            text2 = f"Dist: {distance_cm:.1f} cm"
            cv2.putText(frame, text2, (center_x - 70, bottom_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            text3 = f"X: {x_offset_mm:+.1f} mm"
            cv2.putText(frame, text3, (center_x - 70, bottom_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            text4 = f"Y: {y_offset_mm:+.1f} mm"
            cv2.putText(frame, text4, (center_x - 70, bottom_y + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            text5 = f"Angles: {angle_x:+.1f}°, {angle_y:+.1f}°"
            cv2.putText(frame, text5, (center_x - 90, bottom_y + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Tags: {len(ids)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No tags detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Resize frame for display
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow('AprilTag Detection', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()