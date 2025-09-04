'''
import cv2 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("error")
else:
    print("Camera on")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed")
            break
        cv2.imshow('Camera feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
'''

import cv2
from datetime import datetime

cap = cv2.VideoCapture('rtsp://10.0.0.3:8554/main.264')

if not cap.isOpened():
    print("Error: Cannot open camera")
else:
    print("Camera on")

    recording = False
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Camera feed', frame)

        key = cv2.waitKey(1) & 0xFF

        # Start recording when 's' is pressed
        if key == ord('s') and not recording:
            filename = datetime.now().strftime("video_%Y%m%d_%H%M%S.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
            print(f"Started recording: {filename}")

        # If recording, write frames to file
        if recording and writer is not None:
            writer.write(frame)

        # Stop everything when 'q' is pressed
        if key == ord('q'):
            print("Exiting...")
            break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
