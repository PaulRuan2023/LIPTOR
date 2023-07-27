#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import cv2
import torch
from PIL import Image as PILImage
import numpy as np

def main():
    rospy.init_node('object_detection_node', anonymous=True)
    rate = rospy.Rate(30)  # 设置ROS节点的运行频率

    # 加载YOLOv5模型
    model = torch.jit.load('/home/paul/catkin_ws/src/object_detection_pkg/weights/best_v2.pt')
    model.eval()

    # 配置相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # 获取相机内参
    profile = pipeline.get_active_profile()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    bridge = CvBridge()

    try:
        while not rospy.is_shutdown():
            # 获取相机帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 将颜色图像转换为OpenCV格式（array）
            color_image = np.asanyarray(color_frame.get_data())

            # 将图像转换为PIL Image格式，便于数据处理
            pil_image = PILImage.fromarray(color_image)

            # 进行目标检测
            with torch.no_grad():
                results = model(pil_image)
                results = results[0]

            # 获取检测结果
            boxes = results['boxes']
            scores = results['scores']
            labels = results['labels']

            # 绘制边界框和类别标签
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                class_name = "Class {}".format(label.item())
                confidence = "Confidence: {:.2f}".format(score.item())

                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(color_image, "{} {}".format(class_name, confidence), (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 显示检测结果
            cv2.imshow('RealSense YOLOv5', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()

    finally:
        # 停止相机管道并关闭窗口
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

        main()
    except rospy.ROSInterruptException:
        pass

