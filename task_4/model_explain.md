# 모델 요약

## 1. EfficientNetB0
https://colab.research.google.com/drive/11vGKPCEjzFe-yEq1eYmnOdtMB7v4khqy?usp=sharing
### 텐서플로 공식 홈페이지 내용 일부 변형
### Encoder:
- **EfficientNetB0**: 사전 훈련된 컨볼루션 신경망으로, `include_top=False`를 사용하여 최종 분류 계층을 제거하고 다른 출력 크기에 재사용.

### Decoder:
- EfficientNetB0에 의해 추출된 특징을 분류를 위해 처리.
- **구성 요소**:
  - **Rescaling**: 입력 데이터를 255로 스케일링.
  - **TimeDistributed(net)**: 입력 비디오 시퀀스의 각 프레임에 EfficientNetB0 모델을 적용.
  - **Dense(101)**: 분류 작업을 위한 101개 유닛.
  - **GlobalAveragePooling3D()**: 시간 차원뿐만 아니라 공간 차원에 대해서도 평균을 내어 비디오 프레임에서 추출된 정보를 단일 표현으로 압축하는 데 사용.

### 추가 사항:
- **모델 컴파일**:
  - Adam 최적화 함수와 클래스가 상호 배타적인 다중 클래스 분류 작업에 적합한 Sparse Categorical Crossentropy 손실 함수를 사용함.

## 2. Conv2D LSTM
https://colab.research.google.com/drive/1z-1yDcFST_2mF42msmPLhLTq1QCKG0aD?usp=sharing
### Encoder:
- `TimeDistributed` 레이어 시리즈를 사용하여 구성되며, `Conv2D` 및 `MaxPooling2D` 연산을 비디오 시퀀스의 각 프레임에 걸쳐 프레임별로 적용.
- **구성 요소**:
  - **TimeDistributed(Conv2D)**: 각 프레임에 독립적으로 컨볼루션 레이어를 적용함.
  - **TimeDistributed(MaxPooling2D)**: 컨볼루션 레이어 출력의 공간 차원을 줄임.
  - **TimeDistributed(Dropout)**: 각 프레임에 드롭아웃을 적용하여 정규화를 수행하고 과적합을 줄임.

### Decoder:
- 디코더 구성 요소는 비교적 간단하며, LSTM 레이어 다음에 분류를 위한 softmax 활성화 함수가 있는 밀집 레이어로 구성됨.
- **구성 요소**:
  - **LSTM(32)**: 인코더에 의해 추출된 시퀀스 데이터를 처리하며, 시간적 의존성을 포착함.
  - **Dense(101, activation='softmax')**: LSTM 출력을 101개 클래스에 대한 확률 분포로 매핑함.

### 추가 사항:
- **데이터 준비 및 로딩**:
  - 파일(`ucf101_dataset.npz`)에서 데이터 로드.
  - 제너레이터 함수(`generator`)가 데이터의 배치를 생성하며, 이는 효율적인 로딩 및 훈련을 위해 `tf.data.Dataset.from_generator`에 공급.
- **모델 훈련**:
  - Adam 최적화 함수와 다중 클래스 분류에 적합한 범주형 크로스엔트로피 손실로 모델을 컴파일함.
  

## 3. Context LSTM 
https://colab.research.google.com/drive/14YH8U9zRfvOLOzkaI2apXvb__ScNHvgP?usp=sharing
### 1번의 데이터 설계구조 Context LSTM에 맞게 수정
### Encoder:
- **구성 요소**:
  - ImageNet에 사전 훈련된 ResNet152를 `TimeDistributed` 레이어를 통해 비디오 프레임을 처리하도록 적응시킴. 이 설정을 통해 비디오 시퀀스의 각 프레임을 개별적으로 처리하여 밀집 레이어와 배치 정규화를 거쳐 특징을 추출함.

### Decoder:
-  디코더 미 존재. 외 기타 요소
- **구성 요소**:
  - **LSTMModel**: 인코더에 의해 추출된 순차 데이터를 처리하기 위해 양방향 또는 단방향 LSTM 레이어를 사용함.
  - **AttentionModule**: LSTM 출력에 주의 기법을 적용하여 모델이 비디오 시퀀스의 특정 부분에 초점을 맞춤.
  - **ConvLSTM**: 인코더와 LSTMModel을 통합하고 선택적으로 AttentionModule을 적용하여 비디오 시퀀스를 처리하고 클래스 예측을 수행함.
  - **ConvClassifier**: 인코더와 일련의 밀집 레이어를 사용하여 시간적 역동성 처리 없이 분류를 수행하는 더 단순한 모델.

### 추가 사항:
- **모델 훈련 및 컴파일**:
  - ConvLSTM 모델은 Adam 최적화 함수와 Sparse Categorical Crossentropy 손실로 컴파일.
  - 최종 레이어의 softmax 활성화 함수와 SparseCategoricalCrossentropy의 사용을 통해 다중 클래스 분류를 목표로 함.
