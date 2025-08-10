#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian(Optuna) 하이퍼파라미터 탐색 — 자동 다중 런 처리(트랙 자동탐색)
YOLO bbox 높이(h_norm) → '임펄스(impulse) + bias' 라벨을 CSV 시간축에 스냅핑하여 RMSE 최소화

요청 반영
- CSV는 fps 50이더라도 실제 기록 간격이 0.05s일 수 있음 → CSV의 실제 time 컬럼만 신뢰(별도 fps 가정 없음)
- Label은 fps 30(고정)이며 프레임 기준의 txt로 기록됨 → (frame - first_frame_index)/label_fps 로 절대시간 계산
- 시간 연산 윈도우(팽창/평균/분리) 제거. 임펄스만 사용(라벨 프레임 시점만 값 존재, 그 외는 0).
- 라벨 임펄스는 CSV의 불균일/임의 샘플링에도 대응하도록 "가장 가까운 CSV 시점"으로 스냅핑(Nearest) 처리
- Optuna 탐색 변수: meter_per_pixel(mpp), bias_m, (선택) time_shift
- 자동 정렬(auto-align): CSV와 라벨 임펄스의 상관을 이용해 ±max_lag 내에서 지연(초) 추정
  · 추정 후 사용자가 허용하면 time_shift(탐색변수)와 합산되어 최종 이동량으로 적용

자동 실행(인자 없이):
- /ultralytics/runs/detect/*/labels 내의 *.txt에서 run_name 자동 수집
- 각 run_name에 대해 /workspace/datasets/wave/2025/result/<run>/<run>.csv 자동 매칭
- 결과를 /workspace/runs/height/<run>/ 아래로 저장

예시:
python /workspace/compare_overtopping_height.py
"""

import argparse
import glob
import json
import os
import re
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- Optuna 준비 --------
try:
    import optuna
except ImportError as e:
    raise SystemExit("Optuna가 필요합니다. 설치:  pip install optuna") from e


# =========================
# 탐색 범위(코드에서 수정)
# =========================
# meter_per_pixel 탐색 범위 (로그 샘플링 권장)
MPP_MIN: float = 0.015
MPP_MAX: float = 0.03
MPP_LOG_SAMPLING: bool = True

# 라벨 오프셋(bias, m) 범위: "모두 위로" 평행이동량
BIAS_MIN_M: float = 0.0
BIAS_MAX_M: float = 0.5


# =========================
# 유틸/입출력
# =========================
def find_csv_path(csv_root: str, run_name: str) -> str:
    cand = os.path.join(csv_root, run_name, f"{run_name}.csv")
    if os.path.isfile(cand):
        return cand
    globbed = glob.glob(os.path.join(csv_root, run_name, "*.csv"))
    if len(globbed) == 1:
        return globbed[0]
    raise FileNotFoundError(f"CSV 미발견: {cand}")


def load_csv_timeseries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = [c.strip().lower() for c in df.columns]
    cmap = {}
    for i, c in enumerate(cols):
        if c.startswith("time"):
            cmap["time"] = df.columns[i]
        if c.startswith("h"):
            cmap["h"] = df.columns[i]
        if c == "time (s)":
            cmap["time"] = df.columns[i]
        if c == "h (m)":
            cmap["h"] = df.columns[i]
    if "time" not in cmap or "h" not in cmap:
        raise ValueError(f"CSV 헤더 해석 실패: {list(df.columns)}")
    out = pd.DataFrame({
        "time": df[cmap["time"]].astype(float).values,
        "h_csv": df[cmap["h"]].astype(float).values,
    }).sort_values("time").reset_index(drop=True)
    return out


def list_label_files(labels_dir: str, run_name: str) -> List[str]:
    pattern = os.path.join(labels_dir, f"{run_name}_*.txt")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_(\d+)\.txt$", p).group(1)) if re.search(r"_(\d+)\.txt$", p) else 0
    )
    if not files:
        raise FileNotFoundError(f"라벨 파일 미발견: {pattern}")
    return files


def parse_yolo_xywh_norm(line: str) -> Optional[Dict[str, float]]:
    """
    견고한 YOLO txt 파서: class x y w h [conf ...] (정규화 좌표)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            return None
    cls = int(vals[0])
    xywh = None
    for i in range(1, len(vals) - 3):
        a, b, c, d = vals[i:i+4]
        if 0.0 <= a <= 1.5 and 0.0 <= b <= 1.5 and 0.0 <= c <= 1.5 and 0.0 <= d <= 1.5:
            xywh = (a, b, c, d)
    if xywh is None:
        return None
    x, y, w, h = xywh
    if not (0 <= x <= 1.2 and 0 <= y <= 1.2 and 0 <= w <= 1.2 and 0 <= h <= 1.2):
        return None
    return {"cls": cls, "x": float(x), "y": float(y), "w": float(w), "h": float(h)}


# =========================
# 라벨: 프레임별 h_norm 집계
# =========================
def load_labels_hnorm(files: List[str],
                      allow_classes: Optional[List[int]],
                      agg: str) -> pd.DataFrame:
    """
    각 프레임에서 bbox 높이(h_norm)를 집계하여 반환.
    반환: DataFrame ['frame','h_norm'] (프레임 오름차순)
    """
    rows = []
    frame_re = re.compile(r"_(\d+)\.txt$")
    for fp in files:
        m_ = frame_re.search(fp)
        if not m_:
            continue
        frame = int(m_.group(1))
        vals_h = []
        with open(fp, "r") as f:
            for line in f:
                parsed = parse_yolo_xywh_norm(line)
                if parsed is None:
                    continue
                if allow_classes is not None and parsed["cls"] not in allow_classes:
                    continue
                vals_h.append(parsed["h"])
        if not vals_h:
            continue
        if agg == "max":
            v = float(np.max(vals_h))
        elif agg == "mean":
            v = float(np.mean(vals_h))
        else:
            v = float(np.median(vals_h))
        rows.append((frame, v))
    if not rows:
        raise ValueError("라벨에서 유효한 bbox를 찾지 못했습니다.")
    df = pd.DataFrame(rows, columns=["frame", "h_norm"]).sort_values("frame").reset_index(drop=True)
    return df


# =========================
# 보조 유틸
# =========================
def frames_to_time(frames: np.ndarray, fps: float, first_frame_index: int) -> np.ndarray:
    return (frames.astype(float) - float(first_frame_index)) / float(fps)


def _nearest_indices(target_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """
    target_times(오름차순)에서 각 query_times에 가장 가까운 인덱스 반환.
    """
    idx_right = np.searchsorted(target_times, query_times, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(target_times) - 1)
    idx_right = np.clip(idx_right, 0, len(target_times) - 1)
    left_dist = np.abs(target_times[idx_left] - query_times)
    right_dist = np.abs(target_times[idx_right] - query_times)
    pick_right = right_dist < left_dist
    out = idx_left.copy()
    out[pick_right] = idx_right[pick_right]
    return out


def estimate_time_shift_grid(csv_time: np.ndarray,
                             csv_h: np.ndarray,
                             lbl_on_csv: np.ndarray,
                             max_lag_s: float) -> float:
    """
    CSV 그리드 상에서의 정수/부분 라그 추정(간단 상관도 기반).
    - 입력 두 신호는 같은 csv_time 그리드에 정의되어 있어야 함.
    - grid step은 median(diff(csv_time))로 추정.
    """
    step = float(np.median(np.diff(csv_time)))
    if not np.isfinite(step) or step <= 0:
        return 0.0

    # 기준선 제거 및 정규화
    a = np.asarray(csv_h, dtype=float)
    b = np.asarray(lbl_on_csv, dtype=float)
    a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12)
    b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-12)

    max_lag_steps = int(max(1, round(max_lag_s / step)))
    best_lag = 0
    best_corr = -np.inf
    n = len(a)

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[:n + lag]
        elif lag > 0:
            aa = a[:n - lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        if len(aa) < max(10, int(0.2 * n)):
            continue
        corr = float(np.nanmean(aa * bb))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag * step


# =========================
# 임펄스 스냅핑(라벨 fps=30을 실제 시간으로 환산 후 CSV 그리드로 스냅)
# =========================
def make_label_impulses_snapped(lbl_hnorm_df: pd.DataFrame,
                                csv_time: np.ndarray,
                                label_fps: float,
                                first_frame_index: int,
                                H_lb: float, mpp: float, bias_m: float,
                                time_shift_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    반환:
      - label_on_csv: CSV 그리드와 동일 길이, 기본값=bias_m, 이벤트 인덱스에만 (scaled + bias_m)
      - event_times: 각 라벨 프레임이 매핑된 절대시간(shift 포함, 스냅 전)
      - event_indices: CSV 그리드에서의 스냅된 인덱스
    """
    frames = lbl_hnorm_df["frame"].values.astype(int)
    h_norm = lbl_hnorm_df["h_norm"].values.astype(float)

    # 라벨 프레임 절대시간(초)
    event_times = (frames - float(first_frame_index)) / float(label_fps) + float(time_shift_s)

    # 스냅핑할 CSV 인덱스
    idxs = _nearest_indices(csv_time, event_times)

    # 임펄스 높이
    impulses = h_norm * float(H_lb) * float(mpp) + float(bias_m)

    # CSV 그리드에 임펄스 배치(겹치면 최대값 유지)
    out = np.full_like(csv_time, float(bias_m), dtype=float)
    for i, v in zip(idxs, impulses):
        if v > out[i]:
            out[i] = v

    return out, event_times, idxs


# =========================
# 평가지표
# =========================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "PearsonR": np.nan}
    yt = y_true[m]
    yp = y_pred[m]
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    yt_z = (yt - yt.mean()) / (yt.std() + 1e-12)
    yp_z = (yp - yp.mean()) / (yp.std() + 1e-12)
    pearson = float(np.mean(yt_z * yp_z))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "PearsonR": pearson}


# =========================
# Optuna 목적함수
# =========================
def build_objective(args, csv_time, csv_h, lbl_hnorm_df):
    """
    mpp, bias_m, (선택) time_shift를 탐색.
    - label_fps는 고정(--label-fps)
    - auto-align이 켜져 있으면 지연 추정 후 time_shift와 합산
    """
    H_lb = float(args.norm_base_height)
    label_fps = float(args.label_fps)

    def objective(trial: "optuna.Trial") -> float:
        meter_per_pixel = trial.suggest_float(
            "meter_per_pixel", MPP_MIN, MPP_MAX, log=MPP_LOG_SAMPLING
        )
        bias_m = trial.suggest_float("bias_m", BIAS_MIN_M, BIAS_MAX_M)

        # 탐색용 time_shift(초)
        time_shift = 0.0
        if args.tune_time_shift:
            time_shift = trial.suggest_float("time_shift", -args.max_lag, args.max_lag)

        # 1) 임시: shift=0으로 스냅 → auto-align으로 지연 추정
        lbl_on_csv_tmp, _, _ = make_label_impulses_snapped(
            lbl_hnorm_df, csv_time, label_fps, args.first_frame_index,
            H_lb, meter_per_pixel, bias_m, time_shift_s=0.0
        )
        shift_est = estimate_time_shift_grid(csv_time, csv_h, lbl_on_csv_tmp, args.max_lag) if args.auto_align else 0.0

        # 2) 최종 스냅(추정+탐색 time_shift 합산)
        total_shift = shift_est + time_shift
        lbl_on_csv, _, _ = make_label_impulses_snapped(
            lbl_hnorm_df, csv_time, label_fps, args.first_frame_index,
            H_lb, meter_per_pixel, bias_m, time_shift_s=total_shift
        )

        # 3) 목적함수: RMSE
        rmse = float(np.sqrt(np.mean((csv_h - lbl_on_csv) ** 2)))
        return rmse

    return objective


# =========================
# 최종 평가/플롯/저장 (단일 런)
# =========================
def run_final_eval(args, best_params, csv_time, csv_h, lbl_hnorm_df,
                   out_plot: Optional[str], out_merged: Optional[str], out_debug: Optional[str],
                   run_name: str):
    H_lb = float(args.norm_base_height)
    label_fps = float(args.label_fps)

    meter_per_pixel = float(best_params.get("meter_per_pixel"))
    bias_m = float(best_params.get("bias_m", 0.0))
    time_shift = float(best_params.get("time_shift", 0.0))

    # auto-align 추정
    lbl_on_csv_tmp, _, _ = make_label_impulses_snapped(
        lbl_hnorm_df, csv_time, label_fps, args.first_frame_index,
        H_lb, meter_per_pixel, bias_m, time_shift_s=0.0
    )
    shift_est = estimate_time_shift_grid(csv_time, csv_h, lbl_on_csv_tmp, args.max_lag) if args.auto_align else 0.0

    total_shift = shift_est + time_shift
    lbl_on_csv, event_times, event_idxs = make_label_impulses_snapped(
        lbl_hnorm_df, csv_time, label_fps, args.first_frame_index,
        H_lb, meter_per_pixel, bias_m, time_shift_s=total_shift
    )

    metrics = compute_metrics(csv_h, lbl_on_csv)

    # 병합 저장
    if out_merged:
        os.makedirs(os.path.dirname(out_merged), exist_ok=True)
        merged = pd.DataFrame({
            "time_s": csv_time,
            "h_csv_m": csv_h,
            "h_label_m_on_csv": lbl_on_csv
        })
        merged.to_csv(out_merged, index=False)
        print(f"[{run_name}] [INFO] 병합 시계열 저장: {out_merged}")

    # 디버그 저장(스파스/이벤트)
    if out_debug:
        os.makedirs(os.path.dirname(out_debug), exist_ok=True)
        frames = lbl_hnorm_df["frame"].values.astype(int)
        h_norm = lbl_hnorm_df["h_norm"].values.astype(float)
        scaled_no_bias = h_norm * H_lb * meter_per_pixel
        time_s_sparse_no_shift = frames_to_time(frames, label_fps, args.first_frame_index)
        time_s_sparse = time_s_sparse_no_shift + total_shift

        dbg_sparse = pd.DataFrame({
            "frame": frames,
            "time_s_sparse_no_shift": time_s_sparse_no_shift,
            "time_s_sparse_applied_shift": time_s_sparse,
            "h_norm_sparse": h_norm,
            "scaled_no_bias_sparse_m": scaled_no_bias,
            "bias_m": np.full_like(time_s_sparse, bias_m, dtype=float),
            "label_fps": np.full_like(time_s_sparse, label_fps, dtype=float),
        })
        dbg_sparse.to_csv(out_debug, index=False)

        dbg_events_path = os.path.splitext(out_debug)[0] + "_events.csv"
        pd.DataFrame({
            "event_time_s_before_snap": event_times,
            "mapped_csv_index": event_idxs,
            "mapped_csv_time_s": csv_time[event_idxs],
        }).to_csv(dbg_events_path, index=False)

        print(f"[{run_name}] [INFO] 디버그 저장: {out_debug}")
        print(f"[{run_name}] [INFO] 디버그(이벤트) 저장: {dbg_events_path}")

    # 플롯
    if out_plot:
        os.makedirs(os.path.dirname(out_plot), exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(csv_time, csv_h, label="CSV: h (m)")
        plt.plot(csv_time, lbl_on_csv,
                 label=f"Label (impulses snapped, fps={label_fps:.0f}, shift={total_shift:+.3f}s)",
                 alpha=0.9)
        plt.xlabel("Time (s)")
        plt.ylabel("Height (m)")
        title = (
            f"{run_name} — CSV vs Label (impulses snapped to CSV)\n"
            f"RMSE={metrics['RMSE']:.4f}, mpp={meter_per_pixel:.6f}, "
            f"H_lb={int(H_lb)}, bias={bias_m:.3f} m, shift(est+opt)={total_shift:+.3f}s"
        )
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(out_plot, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"[{run_name}] [INFO] 플롯 저장: {out_plot}")

    return metrics


# =========================
# 자동 탐색: detect/*/labels 안의 run_name → labels_dir 매핑
# =========================
def discover_runs(detect_root: str) -> List[Tuple[str, str]]:
    """
    /ultralytics/runs/detect/*/labels 폴더 내의 *.txt를 스캔하여
    (run_name, labels_dir) 목록을 반환.
    동일 run_name이 여러 labels_dir에 있을 경우 파일 수가 가장 많은 경로를 선택.
    """
    label_dirs = sorted(glob.glob(os.path.join(detect_root, "*", "labels")))
    if not label_dirs:
        print(f"[WARN] labels 디렉토리 미발견: {detect_root}/*/labels")
        return []

    run_to_dirs: Dict[str, Dict[str, int]] = {}
    base_re = re.compile(r"^(.+?)_(\d+)\.txt$")
    for ld in label_dirs:
        txts = glob.glob(os.path.join(ld, "*.txt"))
        for p in txts:
            base = os.path.basename(p)
            m = base_re.match(base)
            if not m:
                continue
            run = m.group(1)
            run_to_dirs.setdefault(run, {})
            run_to_dirs[run][ld] = run_to_dirs[run].get(ld, 0) + 1

    pairs: List[Tuple[str, str]] = []
    for run, dcount in run_to_dirs.items():
        labels_dir = sorted(dcount.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        pairs.append((run, labels_dir))

    pairs.sort(key=lambda x: x[0])
    return pairs


# =========================
# 단일 런 처리 함수
# =========================
def process_one_run(run_name: str,
                    csv_root: str,
                    labels_dir: str,
                    out_root: str,
                    args) -> None:
    print(f"[{run_name}] [INFO] 시작 — labels_dir={labels_dir}")

    # CSV 로드
    try:
        csv_path = find_csv_path(csv_root, run_name)
    except FileNotFoundError as e:
        print(f"[{run_name}] [WARN] CSV 없음: {e}")
        return
    csv_df = load_csv_timeseries(csv_path)
    csv_time = csv_df["time"].values.astype(float)
    csv_h = csv_df["h_csv"].values.astype(float)

    # 라벨 파일 목록/집계
    try:
        label_files = list_label_files(labels_dir, run_name)
    except FileNotFoundError as e:
        print(f"[{run_name}] [WARN] 라벨 없음: {e}")
        return
    try:
        lbl_hnorm_df = load_labels_hnorm(label_files, allow_classes=args.class_ids, agg=args.agg)
    except ValueError as e:
        print(f"[{run_name}] [WARN] 유효 bbox 미발견: {e}")
        return

    # 목적함수/Optuna
    objective = build_objective(args, csv_time, csv_h, lbl_hnorm_df)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, args.n_trials // 10))
    study_name = args.study_name or f"{run_name}"
    study = optuna.create_study(direction="minimize",
                                sampler=sampler,
                                pruner=pruner,
                                study_name=study_name,
                                storage=args.storage,
                                load_if_exists=bool(args.storage))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print(f"\n[{run_name}] [RESULT] Best trial:")
    print(f"[{run_name}]   RMSE: {study.best_value:.6f}")
    for k, v in study.best_params.items():
        print(f"[{run_name}]   {k}: {v}")

    # 출력 경로 구성(개별 런 폴더)
    out_dir = os.path.join(out_root, run_name)
    out_best_json = args.out_best_json or os.path.join(out_dir, "best_params.json")
    out_plot = args.out_plot or os.path.join(out_dir, "compare_plot.png")
    out_merged = args.out_merged or os.path.join(out_dir, "merged_timeseries.csv")
    out_debug = args.out_debug or os.path.join(out_dir, "per_frame_debug.csv")

    # 최종 평가/저장
    os.makedirs(out_dir, exist_ok=True)
    with open(out_best_json, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params},
                  f, ensure_ascii=False, indent=2)
    print(f"[{run_name}] [INFO] 최적 파라미터 저장: {out_best_json}")

    metrics = run_final_eval(args, study.best_params, csv_time, csv_h, lbl_hnorm_df,
                             out_plot=out_plot, out_merged=out_merged, out_debug=out_debug,
                             run_name=run_name)
    print(f"\n[{run_name}] [METRICS @BEST]")
    for k, v in metrics.items():
        print(f"[{run_name}]   {k}: {v:.6f}")


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="Bayesian 월파고 하이퍼파라미터 탐색 — 라벨 fps=30 임펄스 스냅핑 + bias (자동 다중 런)"
    )

    # 자동탐색 루트
    p.add_argument("--detect-root", default="/ultralytics/runs/detect",
                   help="YOLO 결과 루트(내부의 */labels를 자동 탐색)")
    p.add_argument("--csv-root", default="/workspace/datasets/wave/2025/result",
                   help="CSV 루트(각 run 폴더 하위에 <run>.csv)")
    p.add_argument("--out-root", default="/workspace/runs/height",
                   help="출력 루트(각 run 이름별 하위 폴더 생성)")

    # 단일/선택 실행(옵션): 미지정 시 자동으로 전체 탐색
    p.add_argument("--run-name", nargs="*", default=None,
                   help="실행할 run 이름(여러 개 가능). 미지정 시 자동탐색 결과 전체 실행")
    p.add_argument("--labels-dir", default=None,
                   help="단일 런 강제 실행 시 라벨 경로를 직접 지정(자동탐색 우선)")
    p.add_argument("--csv-dir", default=None,
                   help="단일 런 강제 실행 시 CSV 경로 루트를 직접 지정(기본 csv-root)")

    # 시간/정규화 기준
    p.add_argument("--first-frame-index", type=int, default=0,
                   help="YOLO 라벨 프레임 시작 인덱스(보통 0 또는 1)")
    p.add_argument("--label-fps", type=float, default=30.0,
                   help="라벨 프레임레이트(fps). 예: 30")
    p.add_argument("--norm-base-width", type=int, default=1280)
    p.add_argument("--norm-base-height", type=int, default=720)

    # 라벨 필터/집계
    p.add_argument("--class-ids", type=int, nargs="*", default=None)
    p.add_argument("--agg", choices=["max", "mean", "median"], default="max")

    # 시간 정렬/검색 범위
    p.add_argument("--tune-time-shift", type=int, default=1, help="time_shift도 탐색(1) / 비활성(0)")
    p.add_argument("--max-lag", type=float, default=15.0, help="auto-align 및 time_shift 범위(초)")
    p.add_argument("--auto-align", action="store_true", help="CSV와 라벨 임펄스의 상관을 이용해 지연 추정")

    # Optuna
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default=None, help="optuna RDB URL (예: sqlite:///tune.db)")

    # 출력(경로를 직접 지정하고 싶을 때만 사용)
    p.add_argument("--out-best-json", default=None)
    p.add_argument("--out-plot", default=None)
    p.add_argument("--out-merged", default=None)
    p.add_argument("--out-debug", default=None)

    args = p.parse_args()
    return args


# =========================
# 메인
# =========================
def main():
    args = parse_args()
    np.random.seed(args.seed)

    # 실행 대상 run 목록/labels_dir 결정
    targets: List[Tuple[str, str]] = []

    if args.run_name:
        discovered = discover_runs(args.detect_root)
        disc_map = {r: ld for (r, ld) in discovered}
        for rn in args.run_name:
            if args.labels_dir:
                ld = args.labels_dir
            else:
                if rn not in disc_map:
                    print(f"[{rn}] [WARN] 자동탐색에서 labels_dir을 찾지 못했습니다. --labels-dir로 직접 지정하거나 경로 확인")
                    continue
                ld = disc_map[rn]
            targets.append((rn, ld))
    else:
        targets = discover_runs(args.detect_root)
        if not targets:
            print("[ERROR] 실행 대상 런(run_name)을 찾지 못했습니다.")
            return

    # 각 런 처리
    for run_name, labels_dir in targets:
        try:
            process_one_run(
                run_name=run_name,
                csv_root=(args.csv_dir or args.csv_root),
                labels_dir=labels_dir,
                out_root=args.out_root,
                args=args
            )
        except Exception as e:
            print(f"[{run_name}] [ERROR] 처리 실패: {e}")

    print("\n[INFO] 모든 작업 종료")


if __name__ == "__main__":
    main()
