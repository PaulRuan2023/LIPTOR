#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
import numpy as np

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
model.conf = 0.5  # 设置置信度阈值

# 配置相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 获取相机内参
profile = pipeline.get_active_profile()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

try:
    while True:
        # 获取相机帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 将颜色图像转换为OpenCV格式（array）
        color_image = np.asanyarray(color_frame.get_data())

        # 将图像转换为PIL Image格式，便于数据处理
        pil_image = Image.fromarray(color_image)

        # 使用YOLOv5模型进行目标检测
        results = model(pil_image)

        # 初始化最大置信度和索引
        max_confidence = -1
        max_index = -1

        # 遍历检测结果，找到置信度最高的边界框
        for i, detection in enumerate(results.xyxy[0]):
            x1, y1, x2, y2, conf, cls = detection # 左上角和右下角的坐标信息、置信度信息、类别信息

            if conf > max_confidence:
                max_confidence = conf
                max_index = i

        # 检查是否找到置信度超过阈值的边界框
        if max_index != -1:
            # 获取置信度最高的边界框
            x1, y1, x2, y2, conf, cls = results.xyxy[0][max_index] 

            # 获取目标中心坐标
            object_center_x = int((x1 + x2) / 2)
            object_center_y = int((y1 + y2) / 2)

            # 获取目标深度信息
            object_depth = depth_frame.get_distance(object_center_x, object_center_y)

            # 计算物体的三维坐标， 将图像中的像素坐标转换为世界坐标系下的三维空间坐标
            object_position = rs.rs2_deproject_pixel_to_point(intrinsics, [object_center_x, object_center_y], object_depth) 

            # 打印目标类别、置信度、深度信息和位置
            class_name = model.names[int(cls)]
            print("目标：{}，置信度：{}，深度：{}，位置：{}".format(class_name, conf, object_depth, object_position))

            # 绘制边界框和类别标签
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(color_image, "{} {:.2f}".format(class_name, conf), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示检测结果
        cv2.imshow('RealSense YOLOv5', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止相机管道并关闭窗口
    pipeline.stop()
    cv2.destroyAllWindows()

