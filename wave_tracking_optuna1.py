"""
Wave Tracking Optuna - 실해역 신규 스타일 영상용
"""

import optuna
import numpy as np

if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    
from ultralytics import YOLO
from pathlib import Path
import motmetrics as mm
import warnings
import re
import cv2

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_yolo_gt(gt_folder, img_width, img_height, target_class=1):
    """GT 라벨 로드 - target_class만 필터링"""
    gt_folder = Path(gt_folder)
    gt_data = {}

    for txt in sorted(gt_folder.glob("*.txt")):
        nums = re.findall(r"\d+", txt.stem)
        if not nums:
            continue
        frame = int(nums[-1])

        boxes = []
        with open(txt) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    cls, x, y, w, h = map(float, parts[:5])
                    if int(cls) != target_class:  # 🔥 클래스 필터
                        continue
                    xc, yc = x * img_width, y * img_height
                    bw, bh = w * img_width, h * img_height
                    x1, y1 = xc - bw / 2, yc - bh / 2
                    boxes.append([x1, y1, bw, bh])

        if boxes:
            gt_data[frame] = boxes

    print(f"✓ GT loaded: {len(gt_data)} frames (class={target_class} only)")
    return gt_data


def get_video_info(video):
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"✓ Video: {w}x{h}, {n} frames")
    return w, h, n


def evaluate_mota_with_filter(results, gt_data, target_class=1, vid_stride=3,
                                y_threshold=0.62, min_area=0.001, min_height=0.05):
    """
    필터 적용된 MOTA 평가
    
    필터 조건:
    - y_top < y_threshold: 박스 상단이 방파제 위쪽
    - area >= min_area: 너무 작은 박스 제외
    - height >= min_height: 너무 낮은 박스 제외 (물보라만은 대체로 낮음)
    """
    acc = mm.MOTAccumulator(auto_id=True)
    evaluated_frames = 0

    for idx, r in enumerate(results):
        actual_frame = idx * vid_stride + 1
        
        # GT frame 매칭
        gt_frame = None
        for offset in range(vid_stride):
            if (actual_frame + offset) in gt_data:
                gt_frame = actual_frame + offset
                break
            if offset > 0 and (actual_frame - offset) in gt_data:
                gt_frame = actual_frame - offset
                break
        
        if gt_frame is None:
            continue

        evaluated_frames += 1
        gt_boxes = gt_data[gt_frame]
        gt_ids = list(range(len(gt_boxes)))
        gt_bboxes = np.array(gt_boxes)

        if r.boxes is not None and r.boxes.id is not None:
            try:
                cls_arr = r.boxes.cls.cpu().numpy().astype(int)
                ids_all = r.boxes.id.cpu().numpy().astype(int)
                boxes_all = r.boxes.xyxy.cpu().numpy()
                
                # 길이 정합성
                n = min(len(cls_arr), len(ids_all), len(boxes_all))
                if n == 0:
                    pred_ids, pred_bboxes = [], np.empty((0, 4))
                else:
                    cls_arr = cls_arr[:n]
                    ids_all = ids_all[:n]
                    boxes_all = boxes_all[:n]
                    
                    # 이미지 크기 정보 (박스 정규화용)
                    orig_h, orig_w = r.orig_shape[:2]
                    
                    # target_class 필터
                    mask_cls = cls_arr == target_class
                    
                    # 🔥 추가 필터: y_threshold + area + height
                    # xyxy → 정규화 좌표
                    x1n = boxes_all[:, 0] / orig_w
                    y1n = boxes_all[:, 1] / orig_h
                    x2n = boxes_all[:, 2] / orig_w
                    y2n = boxes_all[:, 3] / orig_h
                    
                    w_norm = x2n - x1n
                    h_norm = y2n - y1n
                    area = w_norm * h_norm
                    y_top = y1n
                    
                    # 3가지 조건 모두 만족
                    mask_y = y_top < y_threshold
                    mask_area = area >= min_area
                    mask_height = h_norm >= min_height
                    
                    # 종합 마스크
                    mask = mask_cls & mask_y & mask_area & mask_height
                    
                    pred_ids = ids_all[mask].tolist()
                    boxes = boxes_all[mask]
                    pred_bboxes = np.array(
                        [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
                    ) if len(boxes) > 0 else np.empty((0, 4))
            except Exception:
                pred_ids, pred_bboxes = [], np.empty((0, 4))
        else:
            pred_ids, pred_bboxes = [], np.empty((0, 4))

        if len(gt_bboxes) and len(pred_bboxes):
            dist = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
        else:
            dist = np.empty((len(gt_ids), len(pred_ids)))

        acc.update(gt_ids, pred_ids, dist)

    if evaluated_frames == 0:
        return {"mota": 0.0, "fp": 0, "fn": 0, "idsw": 0,
                "recall": 0.0, "precision": 0.0, "evaluated_frames": 0}

    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "motp", "num_false_positives", 
                                  "num_misses", "num_switches", "recall", 
                                  "precision"], name="res")

    return {
        "mota": float(s["mota"][0]) if not np.isnan(s["mota"][0]) else 0.0,
        "fp": int(s["num_false_positives"][0]),
        "fn": int(s["num_misses"][0]),
        "idsw": int(s["num_switches"][0]),
        "recall": float(s["recall"][0]) if not np.isnan(s["recall"][0]) else 0.0,
        "precision": float(s["precision"][0]) if not np.isnan(s["precision"][0]) else 0.0,
        "evaluated_frames": evaluated_frames
    }

def objective(trial, video, gt_data, model, target_class=1, imgsz=1280, vid_stride=3):
    # ===== Detection =====
    conf = trial.suggest_float("conf", 0.02, 0.50)
    
    # ===== 후처리 필터 (물보라 제거) =====
    y_threshold = trial.suggest_float("y_threshold", 0.50, 0.75)     # 방파제 위 기준
    min_area = trial.suggest_float("min_area", 0.0005, 0.02)         # 최소 면적
    min_height = trial.suggest_float("min_height", 0.03, 0.20)       # 최소 높이
    
    # ===== Tracker (범위 확장) =====
    tracker_type = trial.suggest_categorical("tracker_type", ["bytetrack", "botsort"])
    high = trial.suggest_float("track_high_thresh", 0.05, 0.60)
    low = trial.suggest_float("track_low_thresh", 0.005, 0.15)
    new = trial.suggest_float("new_track_thresh", 0.10, 0.90)
    buf = trial.suggest_int("track_buffer", 30, 400)                 # 확장
    match = trial.suggest_float("match_thresh", 0.30, 0.98)
    area_min = trial.suggest_int("min_box_area", 1, 200)
    fuse = trial.suggest_categorical("fuse_score", [True, False])

    if low >= high:
        return -10.0

    # Tracker yaml 생성
    if tracker_type == "bytetrack":
        tracker_yaml = f"""tracker_type: bytetrack
track_high_thresh: {high}
track_low_thresh: {low}
new_track_thresh: {new}
track_buffer: {buf}
match_thresh: {match}
min_box_area: {area_min}
fuse_score: {str(fuse).lower()}
"""
    else:
        tracker_yaml = f"""tracker_type: botsort
track_high_thresh: {high}
track_low_thresh: {low}
new_track_thresh: {new}
track_buffer: {buf}
match_thresh: {match}
min_box_area: {area_min}
fuse_score: {str(fuse).lower()}
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""
    
    yaml_path = "/tmp/tracker_optuna.yaml"
    with open(yaml_path, "w") as f:
        f.write(tracker_yaml)

    try:
        results = list(
            model.track(
                source=video,
                conf=conf,
                imgsz=imgsz,
                tracker=yaml_path,
                vid_stride=vid_stride,
                half=True,
                stream=True,
                save=False,
                save_txt=False,
                verbose=False,
            )
        )

        # 필터 적용된 MOTA 계산
        metrics = evaluate_mota_with_filter(
            results, gt_data, target_class, vid_stride,
            y_threshold=y_threshold,
            min_area=min_area,
            min_height=min_height,
        )

        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("tracker_type", tracker_type)

        mota = metrics["mota"]
        recall = metrics["recall"]
        idsw = metrics["idsw"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        
        # ===== 종합 페널티 =====
        # 1. Recall 최소 70% 유지 (파도 놓치면 안 됨)
        recall_penalty = max(0, (0.70 - recall) * 0.6)
        
        # 2. IDSW 강력 억제 (핵심)
        idsw_penalty = idsw * 0.008
        
        # 3. FP 억제 (물보라 false positive)
        fp_penalty = max(0, (fp - 30) * 0.003)
        
        # 4. FN 억제
        fn_penalty = max(0, (fn - 40) * 0.002)
        
        score = mota - recall_penalty - idsw_penalty - fp_penalty - fn_penalty
        return score

    except Exception as e:
        print(f"Trial {trial.number} failed:", e)
        return -10.0


def run_optuna(video, gt_folder, model_path, n_trials=60, target_class=1):
    print("="*70)
    print("🎯 WAVE TRACKING OPTUNA - IDSW/FP 최소화 강화판")
    print("="*70)

    w, h, n = get_video_info(video)
    gt_data = load_yolo_gt(gt_folder, w, h, target_class)

    model = YOLO(model_path)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=25)
    )

    best_mota = -10
    def callback(study, trial):
        nonlocal best_mota
        m = trial.user_attrs
        # recall 65% 이상 + MOTA 향상 시 출력
        if (m.get('mota', 0) > best_mota 
            and m.get('recall', 0) >= 0.65):
            best_mota = m['mota']
            print(f"  ⭐ Trial {trial.number}: MOTA={m['mota']*100:.1f}%, "
                  f"Re={m['recall']*100:.0f}%, FP={m['fp']}, "
                  f"FN={m['fn']}, IDSW={m['idsw']}, "
                  f"tracker={m.get('tracker_type', '?')}")

    study.optimize(
        lambda t: objective(t, video, gt_data, model, target_class),
        n_trials=n_trials,
        callbacks=[callback]
    )

    # Recall 65% 이상 trial 중 MOTA 최고
    valid_trials = [t for t in study.trials 
                    if t.user_attrs.get('recall', 0) >= 0.65 
                    and t.user_attrs.get('mota') is not None]
    
    if valid_trials:
        best = max(valid_trials, key=lambda t: t.user_attrs.get('mota', 0))
        print(f"\n✅ Recall>=65% trial: {len(valid_trials)}개")
    else:
        all_trials = [t for t in study.trials 
                      if t.user_attrs.get('mota') is not None]
        if not all_trials:
            print("❌ 모든 trial 실패")
            return study
        best = max(all_trials, key=lambda t: t.user_attrs.get('mota', 0))
        print(f"\n⚠️ Recall 65% 미달, MOTA 기준 선택")

    m = best.user_attrs
    print("\n" + "="*70)
    print("🏆 BEST RESULT")
    print("="*70)
    print(f"  MOTA:      {m.get('mota', 0)*100:6.2f}%")
    print(f"  Recall:    {m.get('recall', 0)*100:6.2f}%")
    print(f"  Precision: {m.get('precision', 0)*100:6.2f}%")
    print(f"  FP:   {m.get('fp', 0):>4}")
    print(f"  FN:   {m.get('fn', 0):>4}")
    print(f"  IDSW: {m.get('idsw', 0):>4}")
    print(f"  Tracker: {m.get('tracker_type', '?')}")

    print(f"\n⚙️ 최적 파라미터:")
    print(f"  --- Detection ---")
    print(f"  conf: {best.params['conf']:.4f}")
    print(f"  --- 후처리 필터 (물보라 제거) ---")
    print(f"  y_threshold: {best.params['y_threshold']:.4f}")
    print(f"  min_area: {best.params['min_area']:.5f}")
    print(f"  min_height: {best.params['min_height']:.4f}")
    print(f"  --- Tracker ({best.params['tracker_type']}) ---")
    for k in ['track_high_thresh', 'track_low_thresh', 'new_track_thresh',
              'track_buffer', 'match_thresh', 'min_box_area', 'fuse_score']:
        v = best.params[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 최적 tracker yaml 저장
    tracker_type = best.params['tracker_type']
    if tracker_type == "bytetrack":
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
    with open("/workspace/tracker_best.yaml", "w") as f:
        f.write(final_yaml)
    
    # 필터 파라미터도 별도 저장 (사후 필터에 사용)
    with open("/workspace/filter_best.txt", "w") as f:
        f.write(f"y_threshold={best.params['y_threshold']:.4f}\n")
        f.write(f"min_area={best.params['min_area']:.5f}\n")
        f.write(f"min_height={best.params['min_height']:.4f}\n")
        f.write(f"conf={best.params['conf']:.4f}\n")
    
    print(f"\n💾 저장:")
    print(f"  /workspace/tracker_best.yaml")
    print(f"  /workspace/filter_best.txt")

    return study


if __name__ == "__main__":
    VIDEO = "/workspace/datasets/wave/mp4_nature/2026-01-20T10-16-10.mp4"
    GT = "/workspace/datasets/wave/labels/val2025_nature/2026-01-20T10-16-10"
    MODEL = "/workspace/runs/wave/yolov9e_2cls/weights/best.pt"
    TARGET_CLASS = 1

    run_optuna(VIDEO, GT, MODEL, n_trials=60, target_class=TARGET_CLASS)

