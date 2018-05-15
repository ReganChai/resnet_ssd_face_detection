# -*- coding = UTF-8 -*-
##  
##  调用 Caffe 框架以及训练好的残差神经网络进行人脸检测
##
##  Created on:  2018年4月24
##      Author: Regan_Chai
##      E-Mail: regan_chai@163.com
##

__author__ = 'Regan_Chai'

import numpy as np
import argparse
import cv2 as cv
try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

from cv2 import dnn

inWidth = 300
inHeight = 300
confThreshold = 0.5
count = 0

prototxt = 'face_detector/deploy.prototxt'    # 调用.caffemodel时的测试网络文件
caffemodel = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'  #包含实际图层权重的.caffemodel文件


def face_detection():
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    #cap = cv.VideoCapture(0)
    #cap = cv.VideoCapture("E:/视频库/srcImage/OneStopMoveEnter1cor.avi")
    frame = cv.imread("face01.jpg")

    while True:
        #ret, frame = cap.read()

        cols = frame.shape[1]
        rows = frame.shape[0]

        net.setInput(dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
        detections = net.forward()

        perf_stats = net.getPerfProfile()
        print('Inference time:  %.2f ms' % (perf_stats[0] / cv.getTickFrequency() * 1000))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                global count
                count += 1
                print(confidence)

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
                label = "face: %.4f" % confidence
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (xLeftBottom, yLeftBottom), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("detections", frame)
        print('the num of face: %d ' %count)
        if cv.waitKey(0) != -1:
            break


if __name__ == '__main__':
    face_detection()
  