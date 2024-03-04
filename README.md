# SideWalkTask

# 1번 과제 - 사람 추출, 배경 합성

## 1. task1_1.py
- **File:** task1_1.py
- **Input:** 1번과제용.mp4
- **Output:** change_background_video_1.mp4
- **Process:** 외부 tflite모델 (selfie_multiclass_256x256) 을 사용하여 영상 내 사람 인식, 배경 이미지 합성.

## 2. task1_2.py
- **File:** task1_2.py
- **Input:** 1번과제용.mp4
- **Output:** change_background_video_2.mp4
- **Process:** mediapipe 내장 함수 (mp_selfie_segmentation) 를 사용하여 영상 내 사람 인식, 배경 이미지 합성.


# 2번 과제 - 배경 / 물체 구분 물체에 색 부여

## 1. task2.py
- **File:** task2.py
- **Input:** 2번과제용.mp4
- **Output:** recognize_object_video.mp4
- **Process:** 외부 tflite모델 (ssd_mobilenet_v2) 을 사용하여 영상 물체 인식. 인식 너비가 화면의 1/4이상일 경우 배경으로 간주


# 3번 과제 - 사람 얼굴 모자이크

## 1. task3.py
- **File:** task3.py
- **Input:** 3번과제용.mp4
- **Output:** blurred_face_video.mp4
- **Process:** 외부 tflite모델 (selfie_multiclass_256x256) 을 사용하여 영상 내 사람 인식 및 얼굴인식, 가우시안 블러 적용


# 4번 과제 - UCF101 데이터 학습
## 모델의 구조 및 데이터 사이즈에 따른 변인을 완벽히 통제하지 못함(컴퓨팅 파워 제한)

## 1. EfficientNet B0 (Train Accuracy: 0.99 / Test Accuracy: 0.87) (사용영상: 5050개)
## 2. Conv2D LSTM (Train Accuracy: 0.73 / Test Accuracy: 0.59) (사용영상: 13320개)
## 3. Context-LSTM (Train Accuracy: 0.83 / Test Accuracy: 0.56) (사용영상: 13320개)
