import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

VIDEO_PATH = '/workspace/datasets/wave/2025_nature/validation/2026-01-20T10-16-10/2026-01-20T10-16-10.mp4'
LABEL_FOLDER = '/ultralytics/runs/detect/predict3/labels'
OUTPUT_VIDEO = '/workspace/output_original_with_graph.mp4'
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
GRAPH_HEIGHT = 500
GRAPH_WINDOW = 150


def load_labels(label_folder, img_h, img_w):
    txt_files = sorted(glob.glob(os.path.join(label_folder, '*.txt')))
    label_map = {}
    for txt_path in txt_files:
        filename = os.path.splitext(os.path.basename(txt_path))[0]
        try:
            frame_num = int(filename.split('_')[-1])
        except ValueError:
            print(f"⚠️  파일명 파싱 실패: {filename}")
            continue
        boxes = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls, cx, cy, w, h = map(float, parts[:5])
                x1 = int((cx - w/2) * img_w)
                y1 = int((cy - h/2) * img_h)
                x2 = int((cx + w/2) * img_w)
                y2 = int((cy + h/2) * img_h)
                boxes.append([x1, y1, x2, y2, h * img_h])
        label_map[frame_num] = boxes
    return label_map


def make_graph_image(height_map, current_frame, all_frame_nums, width, height, window=150):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    start_frame = max(0, current_frame - window)
    end_frame = current_frame + 1
    x_all = list(range(start_frame, end_frame))
    y_all = [height_map.get(f, 0) for f in x_all]
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


def create_video():
    print("=" * 60)
    print("STEP 1: 라벨 로드")
    print("=" * 60)
    label_map = load_labels(LABEL_FOLDER, IMAGE_HEIGHT, IMAGE_WIDTH)
    if not label_map:
        print("❌ 라벨 없음, 종료")
        return
    
    height_map = {fn: max(b[4] for b in boxes) for fn, boxes in label_map.items() if boxes}
    all_frame_nums = sorted(label_map.keys())
    print(f"✓ {len(label_map)}개 프레임에서 라벨 로드 완료")
    print(f"  프레임 범위: {all_frame_nums[0]} ~ {all_frame_nums[-1]}")
    
    print("\n" + "=" * 60)
    print("STEP 2: 원본 영상 열기")
    print("=" * 60)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {VIDEO_PATH}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ 원본 영상 열림")
    print(f"  총 프레임: {total_frames}")
    print(f"  fps (원본): {orig_fps}")
    print(f"  해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    if total_frames == 0:
        print("❌ 프레임 0, 종료")
        cap.release()
        return
    
    fps_for_output = round(orig_fps) if orig_fps > 0 else 30
    print(f"  출력 fps: {fps_for_output}")
    
    print("\n" + "=" * 60)
    print("STEP 3: 출력 영상 준비")
    print("=" * 60)
    total_height = IMAGE_HEIGHT + GRAPH_HEIGHT
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_for_output, (IMAGE_WIDTH, total_height))
    
    if not out.isOpened():
        print("❌ VideoWriter 열기 실패")
        cap.release()
        return
    print(f"✓ 출력 준비: {IMAGE_WIDTH}x{total_height} @ {fps_for_output}fps")
    print(f"  경로: {OUTPUT_VIDEO}")
    
    print("\n" + "=" * 60)
    print("STEP 4: 영상 생성")
    print("=" * 60)
    frame_idx = 0
    boxes_drawn = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\n  프레임 {frame_idx}에서 종료 (더 이상 프레임 없음)")
            break
        
        if frame_idx in label_map:
            for box in label_map[frame_idx]:
                x1, y1, x2, y2, _ = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, 'wave', (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                boxes_drawn += 1
        
        graph = make_graph_image(height_map, frame_idx, all_frame_nums,
                                  IMAGE_WIDTH, GRAPH_HEIGHT, GRAPH_WINDOW)
        combined = np.vstack([frame, graph])
        out.write(combined)
        
        if (frame_idx + 1) % 500 == 0:
            print(f"  진행: {frame_idx+1}/{total_frames} "
                  f"({(frame_idx+1)/total_frames*100:.1f}%), "
                  f"박스 {boxes_drawn}개")
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print("\n" + "=" * 60)
    print("STEP 5: 완료 확인")
    print("=" * 60)
    print(f"✓ 처리 프레임: {frame_idx}")
    print(f"✓ 총 박스: {boxes_drawn}")
    
    if os.path.exists(OUTPUT_VIDEO):
        size_mb = os.path.getsize(OUTPUT_VIDEO) / 1024 / 1024
        print(f"✓ 출력 파일: {OUTPUT_VIDEO} ({size_mb:.1f} MB)")
    else:
        print(f"❌ 출력 파일 미생성")


if __name__ == "__main__":
    create_video()
