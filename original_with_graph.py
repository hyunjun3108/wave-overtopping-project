import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ========================================
# 설정
# ========================================
VIDEO_PATH = '/workspace/datasets/wave/2025_nature/validation/2026-01-20T10-16-10/2026-01-20T10-16-10.mp4'
LABEL_FOLDER = '/ultralytics/runs/detect/predict3/labels'
OUTPUT_VIDEO = '/workspace/output_original_with_graph.mp4'
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
FPS = 20
GRAPH_HEIGHT = 500
GRAPH_WINDOW = 150  # 그래프에 보여줄 프레임 수

# ========================================
# 1. txt 라벨 로드 (프레임 번호 기준)
# ========================================
def load_labels(label_folder, img_h, img_w):
    """
    { 프레임번호(int): [ [x1,y1,x2,y2], ... ] }
    """
    txt_files = sorted(glob.glob(os.path.join(label_folder, '*.txt')))
    label_map = {}

    for txt_path in txt_files:
        filename = os.path.splitext(os.path.basename(txt_path))[0]
        frame_num = int(filename.split('_')[-1])
        boxes = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = map(float, parts[:5])
                # YOLO 정규화 좌표 → 픽셀 변환
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                boxes.append([x1, y1, x2, y2, h * img_h])  # 높이도 저장
        label_map[frame_num] = boxes

    return label_map

# ========================================
# 2. 그래프 생성 (현재 프레임이 오른쪽 끝)
# ========================================

# 2. 그래프 생성 함수 수정 (0값 표시 + 처음부터 시작)
def make_graph_image(height_map, current_frame, all_frame_nums, width, height, window=150):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    start_frame = max(0, current_frame - window)  # 0부터 시작
    end_frame = current_frame + 1

    # 범위 내 모든 프레임 포함 (detection 없으면 0으로 표시)
    x_all = list(range(start_frame, end_frame))
    y_all = [height_map.get(f, 0) for f in x_all]  # 없으면 0

    ax.plot(x_all, y_all, color='cyan', linewidth=1.5)
    ax.axvline(x=current_frame, color='red', linewidth=2)

    ax.set_xlim(start_frame, end_frame)
    ax.set_ylim(0, IMAGE_HEIGHT * 0.5)
    ax.set_ylabel('Box Height (px)', color='white', fontsize=9)
    ax.set_xlabel('Frame', color='white', fontsize=9)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    buf = cv2.resize(buf, (width, height))
    return buf

# ========================================
# 3. 영상 생성
# ========================================
def create_video(video_path, label_folder, output_path, fps):
    label_map = load_labels(label_folder, IMAGE_HEIGHT, IMAGE_WIDTH)
    height_map = {fn: max(b[4] for b in boxes) for fn, boxes in label_map.items()}
    all_frame_nums = sorted(label_map.keys())
    print(f"✓ {len(label_map)}개 프레임에서 라벨 로드 완료")
    print(f"  프레임 범위: {all_frame_nums[0]} ~ {all_frame_nums[-1]}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_fps = round(orig_fps)
    print(f"✓ 원본 영상: {total_frames}프레임, {orig_fps}fps")

    total_height = IMAGE_HEIGHT + GRAPH_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, orig_fps, (IMAGE_WIDTH, total_height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()  # ← 링크 형식도 수정
        if not ret:
            break

        # 박스 그리기
        if frame_idx in label_map:
            for box in label_map[frame_idx]:
                x1, y1, x2, y2, _ = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, 'wave', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 항상 그래프 생성 (조건문 제거)
        graph = make_graph_image(height_map, frame_idx, all_frame_nums,
                                  IMAGE_WIDTH, GRAPH_HEIGHT, GRAPH_WINDOW)

        combined = np.vstack([frame, graph])
        out.write(combined)

        if (frame_idx + 1) % 500 == 0:
            print(f"진행: {frame_idx+1}/{total_frames} ({(frame_idx+1)/total_frames*100:.1f}%)")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"\n✓ 완료! 저장: {output_path}")

# ========================================
# 실행
# ========================================
if __name__ == "__main__":
    create_video(VIDEO_PATH, LABEL_FOLDER, OUTPUT_VIDEO, FPS)