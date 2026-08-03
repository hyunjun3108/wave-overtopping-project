import cv2
from pathlib import Path
import re
import numpy as np

def create_video_from_labels(video_path, labels_folder, output_video_path, max_frames=None):
    """
    필터링된 labels 폴더를 기반으로 시각화 영상 생성
    - YOLO tracking 결과와 동일한 형태로 박스와 ID 표시
    """
    print("="*70)
    print("🎬 CREATING VIDEO FROM FILTERED LABELS")
    print("="*70)
    
    # 비디오 읽기
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is None:
        max_frames = total_frames
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Processing: {max_frames} frames")
    
    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # .avi 형식
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 라벨 파일 로드
    labels_path = Path(labels_folder)
    label_files = {}
    for f in sorted(labels_path.glob('*.txt')):
        numbers = re.findall(r'\d+', f.stem)
        if numbers:
            frame_num = int(numbers[-1])
            label_files[frame_num] = f
    
    print(f"\n📊 Labels: {len(label_files)} frames with detections")
    
    # 🔥 박스 색상: 파란색으로 통일
    BLUE = (255, 100, 0)  # BGR 형식 (파란색)
    
    frame_idx = 0
    frames_with_detection = 0
    frames_without_detection = 0
    
    print(f"\n🔄 Processing frames...")
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num = frame_idx + 1
        
        # 라벨이 있는 경우 박스 그리기
        if frame_num in label_files:
            with open(label_files[frame_num], 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    wave_id = int(parts[5])
                    
                    # 좌표 변환 (normalized → pixel)
                    x_center = x_c * width
                    y_center = y_c * height
                    box_w = w * width
                    box_h = h * height
                    
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    # 파란색 통일
                    color = BLUE
                    
                    # 박스 그리기 (두꺼운 선)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # ID 라벨
                    label = f"ID: {wave_id}"
                    
                    # 라벨 배경
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # 라벨 위치 (박스 위)
                    label_x = x1
                    label_y = y1 - 10
                    if label_y < label_h + 10:
                        label_y = y1 + label_h + 10
                    
                    # 배경 사각형
                    cv2.rectangle(frame, 
                                (label_x - 2, label_y - label_h - 5),
                                (label_x + label_w + 2, label_y + 5),
                                color, -1)
                    
                    # 텍스트 (흰색)
                    cv2.putText(frame, label, (label_x, label_y), 
                               font, font_scale, (255, 255, 255), thickness)
            
            frames_with_detection += 1
        else:
            frames_without_detection += 1
        
        # 프레임 정보 표시 (좌상단)
        info_text = f"Frame: {frame_num}/{max_frames}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        # Detection 상태 표시
        if frame_num in label_files:
            status = "Detection: YES"
            status_color = (0, 255, 0)
        else:
            status = "Detection: NO"
            status_color = (128, 128, 128)
        cv2.putText(frame, status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 500 == 0:
            print(f"   Processed {frame_idx}/{max_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Video created successfully!")
    print(f"   Output: {output_video_path}")
    print(f"   Frames with detection: {frames_with_detection}")
    print(f"   Frames without detection: {frames_without_detection}")
    print("="*70)
    
    return output_video_path


if __name__ == "__main__":
    # 🔥 경로를 실제 환경에 맞게 수정하세요
    VIDEO_PATH = "/workspace/datasets/wave/mp4/V_H08.00T1.8h37.mp4"
    LABELS_FOLDER = "/ultralytics/runs/track/labels_wave"  # 필터링된 labels
    OUTPUT_VIDEO = "/ultralytics/runs/track/V_H08.00T1.8h37_wave.avi"
    
    create_video_from_labels(
        VIDEO_PATH, 
        LABELS_FOLDER, 
        OUTPUT_VIDEO,
        max_frames=None  # 전체 비디오 처리 (또는 숫자로 제한)
    )
    
    print("\n💡 TIP:")
    print("   생성된 영상을 다운로드하여 필터링 결과를 확인하세요.")
    print("   Wave ID가 올바르게 표시되는지 확인!")