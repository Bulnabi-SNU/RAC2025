#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

VIDEO_PATH = '/home/daniel/Downloads/test_video_250714.mp4'

# HSV 캘리브레이션 범위
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([20, 255, 255])

# 가까이용 최소 면적 기준 (픽셀 단위)
min_area = 500

def detect_baguni_close(frame):
    """가까이용: 면적이 충분한 가장 큰 컨투어 중심 반환"""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def detect_baguni_far(frame):
    """멀리용: 마스크된 모든 픽셀의 평균 좌표 반환"""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx = int(xs.mean())
    cy = int(ys.mean())
    return (cx, cy)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{VIDEO_PATH}'")
        return

    # 화면 크기 및 중앙 좌표
    ret, frame0 = cap.read()
    if not ret:
        print("Error: 첫 프레임을 읽을 수 없습니다.")
        return
    h, w = frame0.shape[:2]
    center = (w//2, h//2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 먼저 가까이용 시도
        pt = detect_baguni_close(frame)
        mode = 'close'
        if pt is None:
            # 가까이용 실패하면 멀리용 시도
            pt = detect_baguni_far(frame)
            mode = 'far' if pt else 'none'

        if pt is None:
            print("바구니 인식 실패")
        else:
            cx, cy = pt
            dx = cx - center[0]
            dy = center[1] - cy
            print(f"[{mode}] 바구니 위치 (상대좌표): ({dx}, {dy})")

            # 시각화
            color = (0,0,255) if mode=='close' else (0,128,255)
            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.circle(frame, center, 4, (255,255,0), -1)
            cv2.putText(frame, f"{mode}:({dx},{dy})", 
                        (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Baguni Detection", frame)
        # ESC 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
