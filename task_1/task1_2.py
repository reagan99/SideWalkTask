import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# 배경 이미지 로드 및 조정 함수
def load_and_resize_background(background_path, target_width, target_height):
    background = cv2.imread(background_path)
    background = cv2.resize(background, (target_width, target_height))
    return background

# 사람 분리 및 배경 합성 함수
def segment_and_composite(video_path, background_path, output_path):
    # Mediapipe 설정
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video")
            cap.release()
            return
        
        h, w, _ = frame.shape
        background = load_and_resize_background(background_path, w, h)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 사람 인식 및 분리
            results = selfie_segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mask = results.segmentation_mask > 0.4
            
            # 마스크를 3채널로 변환
            mask_3c = np.dstack([mask]*3)
            
            # 배경과 합성
            composite_frame = np.where(mask_3c, frame, background)
            
            out.write(composite_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

# 함수 호출 예시
segment_and_composite('1번과제용.mp4', 'background.jpg', 'change_background_video_2.mp4')
