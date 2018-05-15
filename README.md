# resnet_ssd_face_detection


## 说明
    
    用 OpenCV 调用 Caffe 框架以及训练好的残差神经网络进行人脸检测
    
## 流程

- 加载模型
    - .prototxt 为调用 .caffemodel 时的测试网络文件
    - .caffemodel 为包含实际图层权重的模型文件
