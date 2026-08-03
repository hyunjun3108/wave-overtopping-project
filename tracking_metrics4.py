"""
Tracking Metrics - 실해역 신규 스타일 평가
"""

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm 
from pathlib import Path
import re


def load_boxes(file_path, img_width, img_height, target_class=None):
    """YOLO 형식 박스 로드 → xywh (pixel)"""
    boxes, ids = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                if target_class is not None and cls != target_class:
                    continue
                x_c, y_c, w, h = map(float, parts[1:5])
                x = (x_c - w/2) * img_width
                y = (y_c - h/2) * img_height
                bw = w * img_width
                bh = h * img_height
                boxes.append([x, y, bw, bh])
                ids.append(int(parts[5]) if len(parts) >= 6 else -1)
    return np.array(boxes) if boxes else np.empty((0, 4)), ids


def calculate_trajectory_metrics(gt_files, pred_files, img_width, img_height, 
                                  target_class=1, gap_threshold=20):
    frame_nums = sorted(gt_files.keys())
    wave_trajectories = {}
    current_wave = 0
    prev_frame = None

    for frame_num in frame_nums:
        gt_boxes, _ = load_boxes(gt_files[frame_num], img_width, img_height, target_class)
        if len(gt_boxes) == 0:
            continue
        if prev_frame is None or frame_num - prev_frame > gap_threshold:
            current_wave += 1
        wave_trajectories.setdefault(current_wave, set()).add(frame_num)
        prev_frame = frame_num

    pred_frames_with_detection = set()
    for frame_num in frame_nums:
        if frame_num in pred_files:
            pred_boxes, _ = load_boxes(pred_files[frame_num], img_width, img_height, target_class)
            if len(pred_boxes) > 0:
                pred_frames_with_detection.add(frame_num)

    mt, pt, ml = 0, 0, 0
    for wave_frames in wave_trajectories.values():
        ratio = len(wave_frames & pred_frames_with_detection) / len(wave_frames)
        if ratio >= 0.8: mt += 1
        elif ratio >= 0.2: pt += 1
        else: ml += 1

    total = len(wave_trajectories)
    return {
        'mt': mt, 'pt': pt, 'ml': ml, 'total': total,
        'mtr': mt/total if total else 0,
        'ptr': pt/total if total else 0,
        'mlr': ml/total if total else 0,
    }


def evaluate(gt_dir, pred_dir, img_width, img_height, target_class=1):
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)

    gt_files = {}
    for f in sorted(gt_path.glob('*.txt')):
        nums = re.findall(r'\d+', f.stem)
        if nums:
            gt_files[int(nums[-1])] = f

    pred_files = {}
    for f in sorted(pred_path.glob('*.txt')):
        nums = re.findall(r'\d+', f.stem)
        if nums:
            pred_files[int(nums[-1])] = f

    print("="*80)
    print(f"📊 TRACKING EVALUATION (class={target_class})")
    print("="*80)
    print(f"GT frames: {len(gt_files)} | Pred frames: {len(pred_files)}")

    acc = mm.MOTAccumulator(auto_id=True)
    evaluated_frames, frames_with_pred, frames_without_pred = 0, 0, 0

    for frame_num in sorted(gt_files.keys()):
        gt_boxes, _ = load_boxes(gt_files[frame_num], img_width, img_height, target_class)
        gt_ids = list(range(len(gt_boxes)))

        if frame_num in pred_files:
            pred_boxes, pred_ids = load_boxes(pred_files[frame_num], img_width, img_height, target_class)
            if len(pred_boxes) > 0:
                frames_with_pred += 1
            else:
                frames_without_pred += 1
        else:
            pred_boxes, pred_ids = np.empty((0, 4)), []
            frames_without_pred += 1

        if len(gt_boxes) and len(pred_boxes):
            dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            dist = np.empty((len(gt_ids), len(pred_ids)))

        acc.update(gt_ids, pred_ids, dist)
        evaluated_frames += 1

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'mota', 'motp', 'num_false_positives', 'num_misses', 'num_switches',
        'recall', 'precision', 'num_objects', 'num_predictions', 'num_matches'
    ], name='res')

    mota = summary['mota'].values[0]
    motp = summary['motp'].values[0]
    motp_percent = (1 - motp) * 100 if not np.isnan(motp) else 0
    fp = int(summary['num_false_positives'].values[0])
    fn = int(summary['num_misses'].values[0])
    idsw = int(summary['num_switches'].values[0])
    recall = summary['recall'].values[0]
    precision = summary['precision'].values[0]
    num_objects = int(summary['num_objects'].values[0])
    num_predictions = int(summary['num_predictions'].values[0])
    num_matches = int(summary['num_matches'].values[0])

    traj = calculate_trajectory_metrics(gt_files, pred_files, img_width, img_height, target_class)
    fp_per_frame = fp / evaluated_frames if evaluated_frames > 0 else 0

    print("\n" + "="*80)
    print("🎯 PRIMARY TRACKING METRICS")
    print("="*80)
    print(f"  1. MOTA:    {mota*100:>7.2f}%")
    print(f"  2. MOTP:    {motp_percent:>7.2f}%")
    print(f"  3. Recall:  {recall*100:>7.2f}%")
    print(f"  4. Precision: {precision*100:>7.2f}%")

    print("\n🎭 TRAJECTORY QUALITY METRICS")
    print("="*80)
    print(f"  5. MTR: {traj['mtr']*100:>7.2f}% ({traj['mt']}/{traj['total']})")
    print(f"  6. PTR: {traj['ptr']*100:>7.2f}% ({traj['pt']}/{traj['total']})")
    print(f"  7. MLR: {traj['mlr']*100:>7.2f}% ({traj['ml']}/{traj['total']})")
    print(f"  8. FP/frame: {fp_per_frame:>7.4f}")

    print("\n📈 DETAILED STATISTICS")
    print("="*80)
    print(f"  Evaluated Frames:     {evaluated_frames}")
    print(f"  Frames with Pred:     {frames_with_pred}")
    print(f"  Frames without Pred:  {frames_without_pred}")
    print(f"  GT Objects:           {num_objects}")
    print(f"  Predictions:          {num_predictions}")
    print(f"  TP / FP / FN / IDSW:  {num_matches} / {fp} / {fn} / {idsw}")

    return {
        'mota': mota, 'motp': motp_percent/100,
        'recall': recall, 'precision': precision,
        'mtr': traj['mtr'], 'ptr': traj['ptr'], 'mlr': traj['mlr'],
        'fp_per_frame': fp_per_frame, 'fp': fp, 'fn': fn, 'idsw': idsw
    }


def get_video_resolution(video_path):
    """비디오에서 실제 해상도 추출"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


if __name__ == "__main__":
    GT_DIR = "/workspace/datasets/wave/labels/val2025_nature/2026-01-20T10-16-10"
    PRED_DIR = "/workspace/runs/wave/track_final_optimized/labels_v2_final_conf0.75_t150_6col"
    VIDEO = "/workspace/datasets/wave/mp4_nature/2026-01-20T10-16-10.mp4"
    TARGET_CLASS = 1

    IMG_WIDTH, IMG_HEIGHT = get_video_resolution(VIDEO)
    print(f"📹 Resolution: {IMG_WIDTH}x{IMG_HEIGHT}\n")
    
    results = evaluate(GT_DIR, PRED_DIR, IMG_WIDTH, IMG_HEIGHT, TARGET_CLASS)