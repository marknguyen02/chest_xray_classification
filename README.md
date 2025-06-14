# Tuberculosis Chest X-ray Classification

Hệ thống phân loại ảnh X quang ngực sử dụng Deep Learning.

## Overview

Hệ thống phân loại tự động phân biệt giữa các trường hợp **bình thường** và **lao phổi** từ ảnh X-quang ngực sử dụng transfer learning với ResNet50.

## Requirements

- Python >= 3.10

- TensorFlow 2.18.0

- NumPy 1.26.4

- Pandas 2.2.3

- OpenCV 4.11.0

- Scikit Learn 1.2.2

- Matplotlib 3.7.2

- Seaborn 0.12.2

## Dataset

Bộ dữ liệu sử dụng trong dự án: [TB Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

## Model Architecture

```
Input (224×224×3) → ResNet50 (frozen) → GlobalAvgPool → Dense(128) → Dropout(0.3) → Dense(2)
```

## Pretrained Model

Model đã được huấn luyện từ trước: [TB Chest X-ray Model](https://www.kaggle.com/models/marknguyen02/xray_tuberculosis_detection)

## Explainable AI

Score-CAM giúp hiểu quá trình ra quyết định của mô hình:
- Trích xuất feature maps từ ResNet50
- Tạo attention heatmaps
- Overlay các vùng quan tâm lên ảnh gốc

## Metrics

Mô hình đạt độ chính xác **0.97** trên dữ liệu kiểm thử, với các thông số khác như sau:

| Lớp | Precision | Recall | F1-Score |
|-----|-----------|--------|----------|
| Normal | 0.96 | 0.98 | 0.97 |
| Tuberculosis | 0.98 | 0.96 | 0.97 |