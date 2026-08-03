from ultralytics import YOLO

print("모델 로딩...")
model = YOLO("/workspace/runs/wave/yolov9e_2cls/weights/best.pt")

print("트래킹 시작 (처음 500 프레임만)...")
results = list(model.track(
    source="/workspace/datasets/wave/mp4_nature/2026-01-20T10-16-10.mp4",
    conf=0.3,
    imgsz=1280,
    classes=[1],
    stream=True,
    save=False,
    verbose=False,
    vid_stride=50,  # 50프레임마다 1장 (빠른 테스트용)
))

print(f"\n총 처리 프레임: {len(results)}")

# detection 있는 프레임 카운트
detected = 0
total_boxes = 0
sample_shown = False

for i, r in enumerate(results):
    if r.boxes is not None and len(r.boxes) > 0:
        detected += 1
        total_boxes += len(r.boxes)
        
        # 첫 detection 샘플 출력
        if not sample_shown:
            print(f"\n=== 첫 detection (frame {i}) ===")
            print(f"  클래스: {r.boxes.cls.cpu().numpy()}")
            print(f"  conf: {r.boxes.conf.cpu().numpy()}")
            print(f"  track ID: {r.boxes.id.cpu().numpy() if r.boxes.id is not None else 'None'}")
            print(f"  박스 개수: {len(r.boxes)}")
            sample_shown = True

print(f"\n=== 결과 ===")
print(f"  detection 있는 프레임: {detected} / {len(results)}")
print(f"  평균 박스 수: {total_boxes/max(1,detected):.2f}")
