#!/usr/bin/env python3

"""
Vision Processing Node
Handles video capture, target detection, and streaming in one process for efficiency.
For Gazebo: subscribes to ROS image messages
For hardware: directly captures from camera/RTSP and handles UDP streaming
"""

__author__ = "tkweon426"
__contact__ = "tkweon426@snu.ac.kr"

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from rcl_interfaces.msg import SetParametersResult

from custom_msgs.msg import VehicleState, TargetLocation

import threading
import time
import numpy as np
import cv2
from cv_bridge import CvBridge

# Import detection modules
from vision_processing_nodes.detection.landingtag import LandingTagDetector
from vision_processing_nodes.detection.casualty import CasualtyDetector
from vision_processing_nodes.detection.droptag import DropTagDetector

from vision_processing_nodes.detection.utils import pixel_to_fov

class VisionProcessorNode(Node):
    def __init__(self):
        super().__init__('vision_processor_node')
        
        self._declare_parameters()
        self._setup_qos()
        self._init_variables()
        self._setup_detection_modules()
        self._setup_ros_modules()
        
        # Start appropriate video source based on configuration
        if self.use_gazebo:
            self._setup_gazebo_mode()
        else:
            self._setup_hardware_mode()
            
        self.get_logger().info("Vision Processor Node initialized")

    def _declare_parameters(self):
        """Declare ROS parameters"""
        self.declare_parameters(
            namespace='',
            parameters=[
                ('use_gazebo', False),
                ('use_jetson', True),
                ('show_debug_stream', True),
                
                # Camera/streaming parameters
                ('camera.device', 0),
                ('camera.rtsp_url', 'rtsp://10.0.0.11:8554/main.264'),
                ('streaming.target_ip', '192.168.1.100'),
                ('streaming.port', 5000),
                ('streaming.width', 1280),
                ('streaming.height', 720),
                ('streaming.fps', 30),
                
                # Detection parameters - Casualty (red objects)
                ('detection.casualty.lower_red1', [0.0, 100.0, 50.0]),
                ('detection.casualty.upper_red1', [10.0, 255.0, 255.0]),
                ('detection.casualty.lower_red2', [170.0, 100.0, 50.0]),
                ('detection.casualty.upper_red2', [180.0, 255.0, 255.0]),
                ('detection.casualty.min_area', 500.0),
                
                # Detection parameters - Drop tag (red zones)
                ('detection.drop_tag.lower_red1', [0.0, 100.0, 50.0]),
                ('detection.drop_tag.upper_red1', [10.0, 255.0, 255.0]),
                ('detection.drop_tag.lower_red2', [170.0, 100.0, 50.0]),
                ('detection.drop_tag.upper_red2', [180.0, 255.0, 255.0]),
                ('detection.drop_tag.min_area', 500.0),
                ('detection.drop_tag.pause_threshold', 0.4),
                
                # Detection parameters - Landing tag (AprilTag)
                ('detection.landing_tag.tag_size', 0.5),
                ('detection.landing_tag.camera_matrix', [1070.089695, 0.0, 1045.772015,
                                                        0.0, 1063.560096, 566.257075,
                                                        0.0, 0.0, 1.0]),
                ('detection.landing_tag.distortion_coeffs', [-0.090292, 0.052332, 0.000171, 0.006618, 0.0]),
                
                # Camera FOV
                ('camera_fov.horizontal', 89.0),
                ('camera_fov.vertical', 60.0),
            ]
        )
        
        # Cache parameter values
        self.use_gazebo = self.get_parameter('use_gazebo').value
        self.use_jetson = self.get_parameter('use_jetson').value
        self.show_debug_stream = self.get_parameter('show_debug_stream').value
        
        self.h_fov = self.get_parameter('camera_fov.horizontal').value
        self.v_fov = self.get_parameter('camera_fov.vertical').value  
    
    def _param_update_callback(self, params):
        """Dynamic parameter update callback"""
        successful = True
        reason = ''
        
        for param in params:
            try:
                
                # Camera parameters
                if param.name == 'camera.device':
                    # Note: Camera device changes require restart
                    self.get_logger().warn("Camera device parameter changed - restart required for effect")
                    
                elif param.name == 'camera.rtsp_url':
                    # Note: RTSP URL changes require restart
                    self.get_logger().warn("RTSP URL parameter changed - restart required for effect")
                    
                # Streaming parameters
                elif param.name in ['streaming.target_ip', 'streaming.port', 'streaming.width', 
                                'streaming.height', 'streaming.fps']:
                    # Note: Streaming parameters require restart
                    self.get_logger().warn(f"Streaming parameter {param.name} changed - restart required for effect")
                    
                # Casualty detection parameters
                elif param.name == 'detection.casualty.lower_red1':
                    self.casualty_detector.update_param(lower_red1=np.array(param.value))
                    self.get_logger().info(f"Updated casualty lower_red1: {param.value}")
                    
                elif param.name == 'detection.casualty.upper_red1':
                    self.casualty_detector.update_param(upper_red1=np.array(param.value))
                    self.get_logger().info(f"Updated casualty upper_red1: {param.value}")
                    
                elif param.name == 'detection.casualty.lower_red2':
                    self.casualty_detector.update_param(lower_red2=np.array(param.value))
                    self.get_logger().info(f"Updated casualty lower_red2: {param.value}")
                    
                elif param.name == 'detection.casualty.upper_red2':
                    self.casualty_detector.update_param(upper_red2=np.array(param.value))
                    self.get_logger().info(f"Updated casualty upper_red2: {param.value}")
                    
                elif param.name == 'detection.casualty.min_area':
                    self.casualty_detector.update_param(min_area=param.value)
                    self.get_logger().info(f"Updated casualty min_area: {param.value}")
                    
                # Drop tag detection parameters
                elif param.name == 'detection.drop_tag.lower_red1':
                    self.drop_tag_detector.update_param(lower_red1=np.array(param.value))
                    self.get_logger().info(f"Updated drop_tag lower_red1: {param.value}")
                    
                elif param.name == 'detection.drop_tag.upper_red1':
                    self.drop_tag_detector.update_param(upper_red1=np.array(param.value))
                    self.get_logger().info(f"Updated drop_tag upper_red1: {param.value}")
                    
                elif param.name == 'detection.drop_tag.lower_red2':
                    self.drop_tag_detector.update_param(lower_red2=np.array(param.value))
                    self.get_logger().info(f"Updated drop_tag lower_red2: {param.value}")
                    
                elif param.name == 'detection.drop_tag.upper_red2':
                    self.drop_tag_detector.update_param(upper_red2=np.array(param.value))
                    self.get_logger().info(f"Updated drop_tag upper_red2: {param.value}")
                    
                elif param.name == 'detection.drop_tag.min_area':
                    self.drop_tag_detector.update_param(min_area=param.value)
                    self.get_logger().info(f"Updated drop_tag min_area: {param.value}")
                    
                elif param.name == 'detection.drop_tag.pause_threshold':
                    self.drop_tag_detector.update_param(pause_threshold=param.value)
                    self.get_logger().info(f"Updated drop_tag pause_threshold: {param.value}")
                    
                # Landing tag detection parameters
                elif param.name == 'detection.landing_tag.tag_size':
                    self.landing_tag_detector.update_param(tag_size=param.value)
                    self.get_logger().info(f"Updated landing_tag tag_size: {param.value}")
                    
                elif param.name == 'detection.landing_tag.camera_matrix':
                    K = np.reshape(np.array(param.value, dtype=np.float64), (3, 3))
                    self.landing_tag_detector.update_param(K=K)
                    self.get_logger().info(f"Updated landing_tag camera_matrix")
                    
                elif param.name == 'detection.landing_tag.distortion_coeffs':
                    D = np.array(param.value, dtype=np.float64)
                    self.landing_tag_detector.update_param(D=D)
                    self.get_logger().info(f"Updated landing_tag distortion_coeffs")
                    
                # Camera FOV parameters
                elif param.name  == 'camera_fov.horizontal':
                    self.h_fov = param.value
                    self.get_logger().info(f"Updated camera_fov.horizontal: {param.value}")
                elif param.name == 'camera_fov.vertical':
                    self.v_fov = param.value
                    self.get_logger().info(f"Updated camera_fov.vertical: {param.value}")
                    
                else:
                    self.get_logger().warn(f"Ignoring parameter: {param.name}")
                    continue
                    
            except Exception as e:
                self.get_logger().error(f"Failed to update parameter {param.name}: {e}")
                successful = False
                reason = f"Failed to update {param.name}: {str(e)}"
                break
        
        if successful:
            self.get_logger().info("[Parameter Update] Vision processing parameters updated successfully")
            
        return SetParametersResult(successful=successful, reason=reason)

    def _setup_qos(self):
        """Setup QoS profiles"""
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Gazebo images use different QoS
        self.image_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        ) if self.use_gazebo else self.qos_profile

    def _init_variables(self):
        """Initialize instance variables"""
        self.bridge = CvBridge()
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Vehicle state tracking
        self.vehicle_state = VehicleState()
        self.detection_status = -1
        self.detection_cx = 0
        self.detection_cy = 0
        
        # Threading components
        self.stream_frame = None
        self.stream_frame_lock = threading.Lock()
        self.running = True
        
    def _setup_detection_modules(self):
        """Initialize detection modules"""
        self.casualty_detector = CasualtyDetector(
            lower_red1=np.array(self.get_parameter('detection.casualty.lower_red1').value),
            upper_red1=np.array(self.get_parameter('detection.casualty.upper_red1').value),
            lower_red2=np.array(self.get_parameter('detection.casualty.lower_red2').value),
            upper_red2=np.array(self.get_parameter('detection.casualty.upper_red2').value),
            min_area=self.get_parameter('detection.casualty.min_area').value
        )
        
        self.drop_tag_detector = DropTagDetector(
            lower_red1=np.array(self.get_parameter('detection.drop_tag.lower_red1').value),
            upper_red1=np.array(self.get_parameter('detection.drop_tag.upper_red1').value),
            lower_red2=np.array(self.get_parameter('detection.drop_tag.lower_red2').value),
            upper_red2=np.array(self.get_parameter('detection.drop_tag.upper_red2').value),
            min_area=self.get_parameter('detection.drop_tag.min_area').value,
            pause_threshold=self.get_parameter('detection.drop_tag.pause_threshold').value
        )
        
        self.landing_tag_detector = LandingTagDetector(
            tag_size=self.get_parameter('detection.landing_tag.tag_size').value,
            K=np.reshape(np.array(self.get_parameter('detection.landing_tag.camera_matrix').value, dtype=np.float64), (3, 3)),
            D=np.array(self.get_parameter('detection.landing_tag.distortion_coeffs').value, dtype=np.float64)
        )

    def _setup_ros_modules(self):
        """Setup ROS publishers and subscribers"""
        # Subscribers
        self.vehicle_state_subscriber = self.create_subscription(
            VehicleState, '/vehicle_state', self._vehicle_state_callback, self.qos_profile
        )
        
        # Publishers
        self.target_publisher = self.create_publisher(
            TargetLocation, '/target_position', self.qos_profile
        )
        
        # Timers
        self.detection_timer = self.create_timer(0.05, self._detection_timer_callback)
        
        # Parameter callback
        self.add_on_set_parameters_callback(self._param_update_callback)

    def _setup_gazebo_mode(self):
        """Setup for Gazebo simulation mode"""
        self.get_logger().info("Setting up Gazebo mode")
        
        # Subscribe to Gazebo camera feed
        self.image_subscriber = self.create_subscription(
            Image,
            '/world/RAC_2025/model/standard_vtol_gimbal_0/link/camera_link/sensor/camera/image',
            self._gazebo_image_callback,
            self.image_qos_profile
        )
        
        if self.show_debug_stream:
            cv2.namedWindow("Vision Debug", cv2.WINDOW_NORMAL)
        

    def _setup_hardware_mode(self):
        """Setup for hardware mode with video capture and streaming"""
        self.get_logger().info("Setting up hardware mode")
        
        # Start video capture thread
        self.capture_thread = threading.Thread(target=self._video_capture_thread, daemon=True)
        self.capture_thread.start()
        
        # Start streaming thread
        self.streaming_thread = threading.Thread(target=self._streaming_thread, daemon=True)
        self.streaming_thread.start()
        
        if self.show_debug_stream:
            cv2.namedWindow("Vision Debug", cv2.WINDOW_NORMAL)
            

    def _video_capture_thread(self):
        """Video capture thread for hardware mode"""
        rtsp_url = self.get_parameter('camera.rtsp_url').value
        camera_device = self.get_parameter('camera.device').value
        
        # Check for RTSP URL first, then fallback to USB camera
        if rtsp_url:
            self.get_logger().info(f"Using RTSP stream: {rtsp_url}")
            if self.use_jetson:
                # Hardware decode RTSP stream on Jetson 
                
                # TODO: Use SIYI Assistant to change camera stream to h265, use the according pipeline
                # Camera's using h264 for now (h265 is too CPU heavy). Need to check with gst-launch 1.0 on a terminal first.
                
                # Example: # Direct pipeline for window display
                # gst-launch-1.0 rtspsrc location=rtspurl ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! nv3dsink
                # Maybe this? 
                
                # For OPENCV, we need to convert the YUV format to BGR for OpenCV
                # pipeline = (
                # "rtspsrc location=rtspurl ! "
                # "rtph264depay ! h265parse ! nvv4l2decoder ! "
                # "nvvidconv ! video/x-raw,format=BGRx ! "
                # "videoconvert ! video/x-raw,format=BGR ! "
                # "appsink emit-signals=true sync=false max-buffers=2 drop=true"
                # )
                gst_pipeline = (
                    f"rtspsrc location={rtsp_url} latency=0 buffer-mode=3 protocols=udp ! "
                    f"rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 disable-dpb=true ! "
                    f"nvvidconv ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1" 
                )

                print(gst_pipeline)
                cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            else:
                # Generic RTSP capture
                gst_pipeline = (
                    f"rtspsrc location={rtsp_url} latency=0 ! "
                    f"rtph264depay ! h264parse ! avdec_h264 ! "
                    f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
                )
                cap = cv2.VideoCapture(rtsp_url)
        else:
            self.get_logger().info(f"Using USB camera device: {camera_device}")
            cap = cv2.VideoCapture(camera_device)
        
        if not cap.isOpened():
            self.get_logger().error("Failed to open video source")
            return
            
        self.get_logger().info("Video capture started")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.get_logger().warn("Failed to read frame")

                continue
                
            # Update current frame with thread safety
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            time.sleep(1/30)  # Target 30 FPS
            
        cap.release()

    def _streaming_thread(self):
        """UDP point-to-point streaming thread"""
        target_ip = self.get_parameter('streaming.target_ip').value
        port = self.get_parameter('streaming.port').value
        width = self.get_parameter('streaming.width').value
        height = self.get_parameter('streaming.height').value
        fps = self.get_parameter('streaming.fps').value
    
        
        # Jetson Orin Nano has no NVENC so it's the same for all cases.
        
        gst_pipeline = (
            f"appsrc ! "
            f"videoconvert ! "
            f"video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 ! "
            f"x264enc bitrate=500 speed-preset=ultrafast tune=zerolatency ! "
            f"h264parse ! "
            f"rtph264pay config-interval=1 pt=96 mtu=1200 ! "
            f"udpsink host={target_ip} port={port}"
        )
       
        self.gst_writer = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height))
        
        if not self.gst_writer.isOpened():
            self.get_logger().error("Failed to open GStreamer pipeline")
            return
            
        platform = "Jetson" if self.use_jetson else "Generic"
        self.get_logger().info(f"{platform} streaming to {target_ip}:{port}")
        
        while self.running:
            with self.stream_frame_lock:
                if self.stream_frame is not None:
                    frame = self.stream_frame.copy()
                else:
                    time.sleep(1/fps)
                    continue
            
            resized = cv2.resize(frame, (width, height))
            self.gst_writer.write(resized)
            time.sleep(1/fps)
                
        self.gst_writer.release()

    def _gazebo_image_callback(self, msg):
        """Callback for Gazebo image messages"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.current_frame = frame
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def _vehicle_state_callback(self, msg):
        """Callback for vehicle state updates"""
        self.vehicle_state = msg

    def _detection_timer_callback(self):
        """Main detection processing timer"""
        with self.frame_lock:
            if self.current_frame is None:
                return
            frame = self.current_frame.copy()
        
        # Reset detection status
        self.detection_status = -1
        
        # Process based on target type
        if self.vehicle_state.detect_target_type == 1:
            self._handle_casualty_detection(frame)
        elif self.vehicle_state.detect_target_type == 2:
            self._handle_drop_tag_detection(frame)
        elif self.vehicle_state.detect_target_type == 3:
            self._handle_landing_tag_detection(frame)
        
        # Create annotated frame for streaming/debug
        annotated_frame = self._create_annotated_frame(frame)
        
        # Update stream frame
        with self.stream_frame_lock:
            self.stream_frame = annotated_frame
        
        # Publish target location
        self._publish_target_location(frame)
        
        # Show debug stream if enabled
        if self.show_debug_stream:
            cv2.imshow("Vision Debug", annotated_frame)
            cv2.waitKey(1)
    

    def _handle_casualty_detection(self, frame):
        """Handle casualty detection"""
        detection, _ = self.casualty_detector.detect_casualty(frame)
        if detection is not None:
            self.detection_status = 0
            self.detection_cx = int(detection[0])
            self.detection_cy = int(detection[1])

    def _handle_drop_tag_detection(self, frame):
        """Handle drop tag detection"""
        detection, _ = self.drop_tag_detector.detect_drop_tag(frame)
        if detection is not None:
            self.detection_status = 0
            self.detection_cx = int(detection[0])
            self.detection_cy = int(detection[1])

    def _handle_landing_tag_detection(self, frame):
        """Handle landing tag detection"""
        detection, _ = self.landing_tag_detector.detect_landing_tag(frame)
        if detection is not None:
            self.detection_status = 0
            self.detection_cx = int(detection[0])
            self.detection_cy = int(detection[1])

    def _publish_target_location(self, frame):
        """Publish target location message"""
        h, w = frame.shape[:2]
        h_fov = self.h_fov
        v_fov = self.v_fov
        
        target_msg = TargetLocation()
        target_msg.status = self.detection_status
        
        if self.detection_status == 0:
            target_msg.angle_x, target_msg.angle_y = pixel_to_fov(
                self.detection_cx, self.detection_cy, w, h, h_fov, v_fov
            )
        
        self.target_publisher.publish(target_msg)

    def _create_annotated_frame(self, frame):
        """Create annotated frame for streaming and debug display"""
        annotated = frame.copy()
        
        # Draw center crosshair
        h, w = annotated.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv2.line(annotated, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(annotated, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        
        # Draw vehicle state at top
        state_text = f"State: {self.vehicle_state.vehicle_state}"
        cv2.putText(annotated, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection type
        detect_type_names = {0: "None", 1: "Casualty", 2: "DropTag", 3: "LandingTag"}
        detect_type_text = f"Target: {detect_type_names.get(self.vehicle_state.detect_target_type, 'Unknown')}"
        cv2.putText(annotated, detect_type_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detection result
        if self.detection_status == 0:
            cv2.circle(annotated, (self.detection_cx, self.detection_cy), 10, (0, 0, 255), 2)
            
            detect_coords_text = f"Detection: ({self.detection_cx}, {self.detection_cy})"
            cv2.putText(annotated, detect_coords_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, "Detection: None", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        self.running = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        if hasattr(self, 'streaming_thread'):
            self.streaming_thread.join(timeout=2.0)
            
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
