import mediapipe as mp
import cv2

video_file = '2번과제용.mp4'
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('recognize_object_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

mp_drawing = mp.solutions.drawing_utils

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
color_index = 0

# ObjectDetector 가중치 파일 설정
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='ssd_mobilenet_v2.tflite'),
    max_results=8,
    running_mode=VisionRunningMode.VIDEO)

detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

# 물체 시각화
def visualize(image, detection_result):
    for detection in detection_result.detections:

        bbox = detection.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        # 1/4 이상이면 배경으로 간주(오 인식 방지)
        if w > width / 4:
            continue

        cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_index % len(colors)], 2)

        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255, 100), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    return image

# 비디오 프레임별로 물체 인식
with detector as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        annotated_image = visualize(frame_rgb, detection_result)

        bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        out.write(bgr_annotated_image)

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
