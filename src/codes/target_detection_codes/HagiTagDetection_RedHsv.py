#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

VIDEO_PATH = 'your_video_path'

# 빨강 범위 (두 구간 합침)
lower_red1 = np.array([0, 100, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170,100, 50])
upper_red2 = np.array([180,255,255])

# 최소 면적 기준 (픽셀 단위)
min_area = 500

def detect_tag_close(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, lower_red1, upper_red1)
    m2   = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None, mask
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None, mask
    pt = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    return pt, mask

def detect_tag_far(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, lower_red1, upper_red1)
    m2   = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(m1, m2)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, mask
    pt = (int(xs.mean()), int(ys.mean()))
    return pt, mask

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{VIDEO_PATH}'")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = int(1000 / fps)

    ret, frame0 = cap.read()
    if not ret:
        print("Error: 첫 프레임을 읽을 수 없습니다.")
        return
    h, w    = frame0.shape[:2]
    center  = (w//2, h//2)
    frame   = frame0.copy()

    paused  = False
    last_pt = None

    while True:
        # 태그 검출 및 마스크 생성
        if not paused:
            res, mask = detect_tag_close(frame)
            mode = 'close'
            if res is None:
                res, mask = detect_tag_far(frame)
                mode = 'far' if res else 'none'
        else:
            # pause 중에도 마스크 비율용 계산
            _, mask = detect_tag_far(frame)
            res = last_pt
            mode = 'pause'

        # 마스크 비율 계산
        red_pixels   = np.count_nonzero(mask)
        total_pixels = mask.size
        ratio        = red_pixels / total_pixels

        # pause 상태 전환
        if not paused and ratio > 0.4:
            paused = True
            print(">>> 화면 40% 이상 빨강: 추적 일시정지")
        elif paused and ratio <= 0.4:
            paused = False
            print("<<< 빨강 비율 낮아짐: 추적 재개")

        # 결과 처리
        if res is None:
            print("하기 태그 인식 실패")
        else:
            last_pt = res
            cx, cy  = res
            dx, dy  = cx - center[0], center[1] - cy
            print(f"[{mode}] 하기 태그 위치: ({dx}, {dy})")
            color = (0,0,255) if mode=='close' else (0,128,255)
            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.circle(frame, center,    4, (255,255,0), -1)

        # 화면 표시
        cv2.imshow("Red Tag Detection", frame)

        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(delay) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
