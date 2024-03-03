import numpy as np
import mediapipe as mp
import cv2


video_file = '1번과제용.mp4'
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('change_background_video_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Background 이미지 설정
bg_image = cv2.imread('background.jpg')
bg_image = cv2.resize(bg_image, (width, height))
bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

# ImageSegmenter 설정
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ImageSegmenter 가중치 파일 설정
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
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

      segmented_masks = segmenter.segment_for_video(mp_image, frame_timestamp_ms)
      category_mask = segmented_masks.category_mask

       # Apply effects
      condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.4
      output_image = np.where(condition, frame_rgb, bg_image)

      output_image_bgr = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
      out.write(output_image_bgr)

# 자원 해제
  cap.release()
  out.release()
  cv2.destroyAllWindows()