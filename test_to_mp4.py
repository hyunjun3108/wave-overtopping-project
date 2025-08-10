import cv2
import os
from glob import glob

# 이미지가 저장된 디렉토리 경로
image_dir = '/ultralytics/runs/detect/predict_V_H08.00T1.8h37'
# 저장할 비디오 파일 경로
output_video_path = '/workspace/runs/predict/predict_V_H08.00T1.8h37.mp4'

# 이미지 파일들을 정렬된 리스트로 가져오기
image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))

# 첫 번째 이미지로 해상도 확인
first_frame = cv2.imread(image_paths[0])
height, width, _ = first_frame.shape

# 비디오 코덱 설정 및 VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID'
fps = 20  # 초당 프레임 수
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 이미지들을 순서대로 영상에 기록
for img_path in image_paths:
    frame = cv2.imread(img_path)
    video_writer.write(frame)

# 자원 해제
video_writer.release()
print(f"영상 저장 완료: {output_video_path}")
