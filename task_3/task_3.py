import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf


# Gpu 설정
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# ImageSegmenter 설정
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

video_file = '3번과제용.mp4'
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('blurred_face_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


# ImageSegmenter 인스턴스 생성
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite'),
    running_mode=VisionRunningMode.VIDEO,
    output_category_mask=True)

with ImageSegmenter.create_from_options(options) as segmenter:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        segmented_masks = segmenter.segment_for_video(mp_image, frame_timestamp_ms)
        category_mask = segmented_masks.category_mask

        # "face-skin" 카테고리에 대한 조건 생성
        condition = (category_mask.numpy_view() == 3)
        condition_stacked = np.stack((condition,) * 3, axis=-1)  # 3-channel mask

        # 조건에 따라 처리
        output_image = np.where(condition_stacked, cv2.GaussianBlur(frame_rgb, (99,99), 0), frame_rgb)

        # OpenCV는 BGR 포맷을 사용하므로 다시 변환
        output_image_bgr = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(output_image_bgr)

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()