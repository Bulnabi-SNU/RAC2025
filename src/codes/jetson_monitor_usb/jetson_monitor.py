import cv2

# Specify the device index (use /dev/video1 or /dev/video2)
video_device = "/dev/video0"  # Or change this to /dev/video2

# Open the video capture device
cap = cv2.VideoCapture(video_device)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Set the video resolution (width x height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)  # Set width to 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)  # Set height to 720

# Check if the resolution is applied correctly
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual resolution: {actual_width}x{actual_height}")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
