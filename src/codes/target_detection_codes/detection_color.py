import os
import cv2
import numpy as np

# ==== CONFIGURATION ====
video_path = "videos/first_video.mp4"
REAL_CROSS_WIDTH_M = 0.22    # meters
FOCAL_LENGTH_PX = 1053.95       # pixels (based on real calibration)

# ==== Load Video ====
abs_path = os.path.abspath(video_path)
print(f"📍 Full path: {abs_path}")
print(f"📁 File exists: {os.path.exists(abs_path)}")

cap = cv2.VideoCapture(abs_path)
if not cap.isOpened():
    print("❌ Failed to open video. Check path or codec.")
    exit()

# ==== Main Loop ====
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent processing (optional)
    # frame = cv2.resize(frame, (640, 480))

    # Convert to HSV and isolate red
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define mask for red through orange to yellow
    lower_range = np.array([0, 100, 100])   # start at red
    upper_range = np.array([35, 255, 255])  # end at yellow

    mask = cv2.inRange(hsv, lower_range, upper_range)


    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Largest contour
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 500:  # ignore small noise
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            # Estimate distance
            est_distance = (FOCAL_LENGTH_PX * REAL_CROSS_WIDTH_M) / w

            # Annotate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Center: ({cx}, {cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Est. distance: {est_distance:.2f} m", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show result
    cv2.imshow("Red Cross Detection", frame)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()