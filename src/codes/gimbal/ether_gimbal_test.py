import cv2

# Try different RTSP URLs
urls = [
    'rtsp://10.0.0.3:8554/main.264',
]

cap = None
for url in urls:
    print(f"Trying: {url}")
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print(f"Successfully connected to: {url}")
        break
    cap.release()

if cap is None or not cap.isOpened():
    print("Failed to connect to camera")
    exit()

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('SIYI A8 Mini', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to read frame")
        break

cap.release()
cv2.destroyAllWindows()