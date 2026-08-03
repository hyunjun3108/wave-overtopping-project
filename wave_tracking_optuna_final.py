"""
Wave Tracking - 최종 Optuna
전체 파이프라인 (track → filter → wave_id → tracking_metrics) 최적화
"""
import optuna
import numpy as np

# NumPy 2.0+ 호환성 패치
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

from ultralytics import YOLO
from pathlib import Path
import motmetrics as mm
import warnings
import re
import cv2
import shutil
from collections import defaultdict

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# 경로 (필요시 수정)
# ============================================================
VIDEO = "/workspace/datasets/wave/mp4_nature/2026-01-20T10-16-10.mp4"
GT_FOLDER = "/workspace/datasets/wave/labels/val2025_nature/2026-01-20T10-16-10"
MODEL_PATH = "/workspace/runs/wave/yolov9e_2cls/weights/best.pt"
TARGET_CLASS = 1
IMGSZ = 1280
VID_STRIDE = 3   # 속도 vs 정확도 트레이드오프
# ============================================================


def load_yolo_gt(gt_folder, img_w, img_h, target_class=1):
    """GT 로드"""
    gt_data = {}
    for txt in sorted(Path(gt_folder).glob("*.txt")):
        nums = re.findall(r"\d+", txt.stem)
        if not nums: continue
        frame = int(nums[-1])
        boxes = []
        with open(txt) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    cls, x, y, w, h = map(float, parts[:5])
                    if int(cls) != target_class: continue
                    xc, yc = x * img_w, y * img_h
                    bw, bh = w * img_w, h * img_h
                    boxes.append([xc - bw/2, yc - bh/2, bw, bh])
        if boxes:
            gt_data[frame] = boxes
    return gt_data


def get_video_info(video):
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, n


def reassign_wave_ids_in_memory(all_detections, wave_gap_threshold):
    """
    all_detections: {frame_num: [(cls, xc, yc, w, h, conf), ...]}
    return: {frame_num: [(cls, xc, yc, w, h, wave_id), ...]}
    """
    sorted_frames = sorted(all_detections.keys())
    if not sorted_frames:
        return {}
    
    current_wave_id = 0
    prev_frame = None
    result = {}
    
    for frame_num in sorted_frames:
        if prev_frame is None:
            current_wave_id = 1
        elif frame_num - prev_frame > wave_gap_threshold:
            current_wave_id += 1
        
        result[frame_num] = [
            (cls, xc, yc, w, h, current_wave_id)
            for (cls, xc, yc, w, h, conf) in all_detections[frame_num]
        ]
        prev_frame = frame_num
    
    return result


def evaluate_full_pipeline(results, gt_data, target_class, vid_stride,
                            y_threshold, conf_post, wave_gap,
                            img_w, img_h):
    """
    전체 파이프라인:
    1. tracking 결과 필터링 (y_top, conf)
    2. wave ID 재할당
    3. MOTA 계산 (tracking_metrics 방식)
    """
    # 1) Tracking 결과 → detection dict
    all_detections = defaultdict(list)
    
    for idx, r in enumerate(results):
        actual_frame = idx * vid_stride + 1
        
        if r.boxes is None or len(r.boxes) == 0:
            continue
        
        try:
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            boxes_all = r.boxes.xyxy.cpu().numpy()
            conf_arr = r.boxes.conf.cpu().numpy()
            
            n = min(len(cls_arr), len(boxes_all), len(conf_arr))
            if n == 0: continue
            
            for i in range(n):
                if int(cls_arr[i]) != target_class:
                    continue
                
                # xyxy → 정규화 xywh
                x1, y1, x2, y2 = boxes_all[i][:4]
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                
                # 필터
                y_top = yc - h / 2
                if y_top >= y_threshold:
                    continue
                if conf_arr[i] < conf_post:
                    continue
                
                all_detections[actual_frame].append(
                    (int(cls_arr[i]), xc, yc, w, h, float(conf_arr[i]))
                )
        except Exception:
            continue
    
    # 2) Wave ID 재할당
    detections_with_wave_id = reassign_wave_ids_in_memory(dict(all_detections), wave_gap)
    
    # 3) MOTA 계산 (프레임별 매칭)
    acc = mm.MOTAccumulator(auto_id=True)
    evaluated_frames = 0
    wave_frames = defaultdict(set)  # wave_id → 등장 프레임 (MTR/MLR 계산용)
    
    for frame_num in sorted(gt_data.keys()):
        gt_boxes = gt_data[frame_num]
        gt_ids = list(range(len(gt_boxes)))
        gt_bboxes = np.array(gt_boxes)
        
        # 예측 프레임 찾기 (vid_stride 근처)
        pred_frame = None
        for offset in range(vid_stride):
            if (frame_num + offset) in detections_with_wave_id:
                pred_frame = frame_num + offset
                break
            if offset > 0 and (frame_num - offset) in detections_with_wave_id:
                pred_frame = frame_num - offset
                break
        
        pred_ids = []
        pred_bboxes_list = []
        
        if pred_frame is not None:
            for cls, xc, yc, w, h, wave_id in detections_with_wave_id[pred_frame]:
                # 정규화 → 픽셀
                x_px = (xc - w/2) * img_w
                y_px = (yc - h/2) * img_h
                w_px = w * img_w
                h_px = h * img_h
                pred_bboxes_list.append([x_px, y_px, w_px, h_px])
                pred_ids.append(wave_id)
                wave_frames[wave_id].add(frame_num)
        
        pred_bboxes = np.array(pred_bboxes_list) if pred_bboxes_list else np.empty((0, 4))
        
        if len(gt_bboxes) and len(pred_bboxes):
            dist = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
        else:
            dist = np.empty((len(gt_ids), len(pred_ids)))
        
        acc.update(gt_ids, pred_ids, dist)
        evaluated_frames += 1
    
    if evaluated_frames == 0:
        return {"mota": 0.0, "fp": 0, "fn": 0, "idsw": 0,
                "recall": 0.0, "precision": 0.0, "mtr": 0.0, "n_waves": 0}
    
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "num_false_positives", "num_misses",
                                  "num_switches", "recall", "precision"],
                   name="res")
    
    # MTR 계산: 각 wave가 얼마나 오래 추적됐는지
    # (실제 파도 개수는 GT 프레임 그룹으로 추정)
    gt_frames_sorted = sorted(gt_data.keys())
    gt_wave_groups = []
    current_group = []
    for f in gt_frames_sorted:
        if not current_group or f - current_group[-1] <= wave_gap:
            current_group.append(f)
        else:
            gt_wave_groups.append(current_group)
            current_group = [f]
    if current_group:
        gt_wave_groups.append(current_group)
    
    n_gt_waves = len(gt_wave_groups)
    
    # 각 GT wave가 pred에서 얼마나 커버됐는지
    mt_count = 0
    for gt_wave in gt_wave_groups:
        gt_wave_set = set(gt_wave)
        # 이 GT wave와 매칭된 pred wave 찾기
        pred_frames_covered = set()
        for wid, frames in wave_frames.items():
            overlap = gt_wave_set & frames
            if overlap:
                pred_frames_covered.update(overlap)
        
        coverage = len(pred_frames_covered) / len(gt_wave_set)
        if coverage >= 0.8:
            mt_count += 1
    
    mtr = mt_count / n_gt_waves if n_gt_waves > 0 else 0
    
    return {
        "mota": float(s["mota"][0]) if not np.isnan(s["mota"][0]) else 0.0,
        "fp": int(s["num_false_positives"][0]),
        "fn": int(s["num_misses"][0]),
        "idsw": int(s["num_switches"][0]),
        "recall": float(s["recall"][0]) if not np.isnan(s["recall"][0]) else 0.0,
        "precision": float(s["precision"][0]) if not np.isnan(s["precision"][0]) else 0.0,
        "mtr": mtr,
        "n_waves": len(wave_frames),
        "n_gt_waves": n_gt_waves,
    }


def objective(trial, video, gt_data, model, img_w, img_h):
    # Detection
    conf = trial.suggest_float("conf", 0.05, 0.50)
    
    # 사후 필터
    y_threshold = trial.suggest_float("y_threshold", 0.55, 0.70)
    conf_post = trial.suggest_float("conf_post", 0.0, 0.85)  # 트래킹 후 conf 필터
    
    # Wave ID gap (25.25fps 기준: 200 = 8초, 500 = 20초)
    wave_gap = trial.suggest_int("wave_gap_threshold", 100, 500)
    
    # Tracker 선택
    tracker_type = trial.suggest_categorical("tracker_type", ["bytetrack", "botsort"])
    
    high = trial.suggest_float("track_high_thresh", 0.05, 0.60)
    low = trial.suggest_float("track_low_thresh", 0.005, 0.15)
    new = trial.suggest_float("new_track_thresh", 0.10, 0.90)
    buf = trial.suggest_int("track_buffer", 30, 400)
    match = trial.suggest_float("match_thresh", 0.30, 0.98)
    area = trial.suggest_int("min_box_area", 1, 200)
    fuse = trial.suggest_categorical("fuse_score", [True, False])
    
    if low >= high:
        return -10.0
    
    if tracker_type == "bytetrack":
        tracker_yaml = f"""tracker_type: bytetrack
track_high_thresh: {high}
track_low_thresh: {low}
new_track_thresh: {new}
track_buffer: {buf}
match_thresh: {match}
min_box_area: {area}
fuse_score: {str(fuse).lower()}
"""
    else:
        tracker_yaml = f"""tracker_type: botsort
track_high_thresh: {high}
track_low_thresh: {low}
new_track_thresh: {new}
track_buffer: {buf}
match_thresh: {match}
min_box_area: {area}
fuse_score: {str(fuse).lower()}
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""
    
    yaml_path = f"/tmp/tracker_optuna_{trial.number}.yaml"
    with open(yaml_path, "w") as f:
        f.write(tracker_yaml)
    
    try:
        results = list(
            model.track(
                source=video,
                conf=conf,
                imgsz=IMGSZ,
                tracker=yaml_path,
                vid_stride=VID_STRIDE,
                stream=True,
                save=False,
                verbose=False,
            )
        )
        
        metrics = evaluate_full_pipeline(
            results, gt_data, TARGET_CLASS, VID_STRIDE,
            y_threshold, conf_post, wave_gap,
            img_w, img_h
        )
        
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("tracker_type", tracker_type)
        
        # ========== 페널티 강화 스코어 ==========
        mota = metrics["mota"]
        recall = metrics["recall"]
        idsw = metrics["idsw"]
        fp = metrics["fp"]
        mtr = metrics["mtr"]
        n_waves = metrics["n_waves"]
        n_gt_waves = metrics["n_gt_waves"]
        
        # 1. Recall 70% 이상 유지
        recall_penalty = max(0, (0.70 - recall) * 0.6)
        
        # 2. IDSW 강력 억제
        idsw_penalty = idsw * 0.008
        
        # 3. FP 억제
        fp_penalty = max(0, (fp - 50) * 0.003)
        
        # 4. MTR 보너스 (0.9 이상이면 보너스)
        mtr_bonus = max(0, (mtr - 0.9) * 0.1)
        
        # 5. Wave 개수가 GT와 크게 차이나면 페널티
        wave_diff_penalty = 0
        if n_gt_waves > 0:
            ratio = n_waves / n_gt_waves
            if ratio < 0.7 or ratio > 1.5:
                wave_diff_penalty = 0.05
        
        score = mota - recall_penalty - idsw_penalty - fp_penalty \
                + mtr_bonus - wave_diff_penalty
        
        # 파일 정리
        try:
            import os
            os.remove(yaml_path)
        except: pass
        
        return score
    
    except Exception as e:
        print(f"Trial {trial.number} failed:", e)
        return -10.0


def run_optuna(n_trials=60):
    print("=" * 70)
    print("🎯 WAVE TRACKING OPTUNA - 최종판 (전체 파이프라인 최적화)")
    print("=" * 70)
    
    w, h, n = get_video_info(VIDEO)
    print(f"✓ Video: {w}x{h}, {n} frames")
    
    gt_data = load_yolo_gt(GT_FOLDER, w, h, TARGET_CLASS)
    print(f"✓ GT loaded: {len(gt_data)} frames")
    
    model = YOLO(MODEL_PATH)
    print(f"✓ Model loaded")
    print()
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=25)
    )
    
    best_mota = -10
    def callback(study, trial):
        nonlocal best_mota
        m = trial.user_attrs
        current_mota = m.get('mota', 0)
        if current_mota > best_mota and m.get('recall', 0) >= 0.65:
            best_mota = current_mota
            print(f"  ⭐ Trial {trial.number}: "
                  f"MOTA={current_mota*100:.1f}%, "
                  f"Re={m.get('recall', 0)*100:.0f}%, "
                  f"MTR={m.get('mtr', 0)*100:.0f}%, "
                  f"IDSW={m.get('idsw', 0)}, "
                  f"waves={m.get('n_waves', 0)}, "
                  f"tracker={m.get('tracker_type', '?')}")
    
    study.optimize(
        lambda t: objective(t, VIDEO, gt_data, model, w, h),
        n_trials=n_trials,
        callbacks=[callback]
    )
    
    # 결과 정렬
    valid_trials = [t for t in study.trials
                    if t.user_attrs.get('recall', 0) >= 0.65
                    and t.user_attrs.get('mota') is not None]
    
    if not valid_trials:
        valid_trials = [t for t in study.trials
                        if t.user_attrs.get('mota') is not None]
    
    if not valid_trials:
        print("❌ 유효한 trial 없음")
        return
    
    # MOTA 순으로 정렬
    valid_trials.sort(key=lambda t: t.user_attrs.get('mota', 0), reverse=True)
    best = valid_trials[0]
    m = best.user_attrs
    
    print("\n" + "=" * 70)
    print("🏆 BEST RESULT")
    print("=" * 70)
    print(f"  MOTA:      {m.get('mota', 0)*100:6.2f}%")
    print(f"  Recall:    {m.get('recall', 0)*100:6.2f}%")
    print(f"  Precision: {m.get('precision', 0)*100:6.2f}%")
    print(f"  MTR:       {m.get('mtr', 0)*100:6.2f}%")
    print(f"  FP:        {m.get('fp', 0):>4}")
    print(f"  FN:        {m.get('fn', 0):>4}")
    print(f"  IDSW:      {m.get('idsw', 0):>4}")
    print(f"  Waves:     {m.get('n_waves', 0)} / GT {m.get('n_gt_waves', 0)}")
    print(f"  Tracker:   {m.get('tracker_type', '?')}")
    
    print(f"\n⚙️  BEST PARAMETERS:")
    print(f"  --- Detection ---")
    print(f"  conf: {best.params['conf']:.4f}")
    print(f"  --- 사후 필터 ---")
    print(f"  y_threshold: {best.params['y_threshold']:.4f}")
    print(f"  conf_post: {best.params['conf_post']:.4f}")
    print(f"  wave_gap_threshold: {best.params['wave_gap_threshold']}")
    print(f"  --- Tracker ({best.params['tracker_type']}) ---")
    for k in ['track_high_thresh', 'track_low_thresh', 'new_track_thresh',
              'track_buffer', 'match_thresh', 'min_box_area', 'fuse_score']:
        v = best.params[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 최종 tracker yaml 저장
    tt = best.params['tracker_type']
    if tt == "bytetrack":
        final_yaml = f"""tracker_type: bytetrack
track_high_thresh: {best.params['track_high_thresh']:.4f}
track_low_thresh: {best.params['track_low_thresh']:.4f}
new_track_thresh: {best.params['new_track_thresh']:.4f}
track_buffer: {best.params['track_buffer']}
match_thresh: {best.params['match_thresh']:.4f}
min_box_area: {best.params['min_box_area']}
fuse_score: {str(best.params['fuse_score']).lower()}
"""
    else:
        final_yaml = f"""tracker_type: botsort
track_high_thresh: {best.params['track_high_thresh']:.4f}
track_low_thresh: {best.params['track_low_thresh']:.4f}
new_track_thresh: {best.params['new_track_thresh']:.4f}
track_buffer: {best.params['track_buffer']}
match_thresh: {best.params['match_thresh']:.4f}
min_box_area: {best.params['min_box_area']}
fuse_score: {str(best.params['fuse_score']).lower()}
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""
    with open("/workspace/tracker_final.yaml", "w") as f:
        f.write(final_yaml)
    
    # 필터 파라미터 저장
    with open("/workspace/filter_final.txt", "w") as f:
        f.write(f"conf={best.params['conf']:.4f}\n")
        f.write(f"y_threshold={best.params['y_threshold']:.4f}\n")
        f.write(f"conf_post={best.params['conf_post']:.4f}\n")
        f.write(f"wave_gap_threshold={best.params['wave_gap_threshold']}\n")
        f.write(f"tracker_type={tt}\n")
    
    print(f"\n💾 저장:")
    print(f"  /workspace/tracker_final.yaml")
    print(f"  /workspace/filter_final.txt")
    
    print(f"\n📊 상위 5 trials:")
    for i, t in enumerate(valid_trials[:5]):
        ma = t.user_attrs
        print(f"  {i+1}. MOTA={ma.get('mota',0)*100:.1f}%, "
              f"Re={ma.get('recall',0)*100:.0f}%, "
              f"MTR={ma.get('mtr',0)*100:.0f}%, "
              f"IDSW={ma.get('idsw',0)}, "
              f"waves={ma.get('n_waves',0)}, "
              f"tracker={ma.get('tracker_type','?')}")
    
    return study


if __name__ == "__main__":
    run_optuna(n_trials=60)
