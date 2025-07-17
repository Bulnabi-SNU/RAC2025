#!/usr/bin/env python3
"""
landing_tag_detector_cv.py

- camera_info.yaml 로부터 K, D 로드
- /camera/image_raw를 구독하여 AprilTag(aruco) 검출
- cv2.solvePnP로 Tag 6DoF Pose 계산
- LandingTagLocation 메시지(x,y,z,yaw,height) 퍼블리시
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from custom_msgs.msg import LandingTagLocation  # 패키지·네임스페이스 맞게 수정

import cv2
import numpy as np

class LandingTagDetector(Node):
    def __init__(self):
        super().__init__("landing_tag_detector_cv")

        # === 파라미터 선언 ===
        self.declare_parameter("image_topic", "/world/RAC_2025/model/standard_vtol_0/link/camera_link/sensor/imager/image")
        self.declare_parameter("tag_size_m", 0.5)           # Tag 한 변 길이 [m]
        self.declare_parameter("pub_topic", "/landing_tag/location")

        self.tag_sz = self.get_parameter("tag_size_m").get_parameter_value().double_value
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        pub_topic   = self.get_parameter("pub_topic").get_parameter_value().string_value

        # === 카메라 파라미터 로드 ===
        self.K = np.array([
            [1070.089695, 0.0, 1045.772015],
            [0.0, 1063.560096, 566.257075],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.D = np.array([-0.090292, 0.052332, 0.000171, 0.006618, 0.0], dtype=np.float64)

        # === OpenCV aruco 설정 (DICT_APRILTAG_36h10) ===
        self.dictionary   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        self.det_params   = cv2.aruco.DetectorParameters()
        # 검출기 생성 (OpenCV >=4.7)
        self.detector     = cv2.aruco.ArucoDetector(self.dictionary, self.det_params)

        # === ROS 통신 ===
        self.bridge    = CvBridge()
        self.sub_image = self.create_subscription(Image, image_topic,
                                                  self.image_cb, 10)
        self.pub_pose  = self.create_publisher(LandingTagLocation, pub_topic, 10)

        self.get_logger().info(f"✅ 노드 초기화 완료. 이미지 토픽: {image_topic}  →  {pub_topic}")

    # ---------- 이미지 콜백 ----------
    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge 오류: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === Tag 검출 ===
        corners, ids, rejected = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            self.get_logger().error(f"Not detected apriltag") # detect 까지는 됨
            return  # 미검출

        # 첫 번째 Tag만 사용 (필요 시 루프 돌려 모든 Tag 처리 가능)
        img_pts = corners[0].reshape(-1, 2).astype(np.float32)  # (4,2)

        # solvePnP용 3D 좌표계 정의 (태그 중심이 원점, 시계방향)
        s = self.tag_sz / 2.0
        obj_pts = np.array([[-s,  s, 0],
                            [ s,  s, 0],
                            [ s, -s, 0],
                            [-s, -s, 0]], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                           self.K, self.D,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            self.get_logger().warn("solvePnP 실패")
            return

        # 위치 (카메라 좌표계)
        x, y, z = tvec.flatten()

        # yaw 추출 (Z축 회전)
        R, _ = cv2.Rodrigues(rvec)
        yaw = np.arctan2(R[1, 0], R[0, 0])

        # 메시지 작성·퍼블리시
        loc = LandingTagLocation()
        loc.x = float(x)
        loc.y = float(y)
        loc.z = float(z)
        loc.yaw = float(yaw)
        loc.height = float(z)

        self.pub_pose.publish(loc)
        self.get_logger().debug(f"Tag pose: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = LandingTagDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
