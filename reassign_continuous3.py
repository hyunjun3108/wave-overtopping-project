"""
Optuna 최적 파라미터로 트래킹한 결과에 Wave ID 재할당
"""

from pathlib import Path
import re


def reassign_wave_ids(pred_folder, output_folder, wave_gap_threshold=20):
    pred_path = Path(pred_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 프레임 번호로 정렬된 파일 리스트
    file_with_frame = []
    for f in pred_path.glob('*.txt'):
        nums = re.findall(r'\d+', f.stem)
        if nums:
            file_with_frame.append((int(nums[-1]), f))
    file_with_frame.sort()

    print("=" * 70)
    print("🌊 WAVE ID ASSIGNMENT")
    print("=" * 70)

    current_wave_id = 0
    prev_frame_num = None
    frame_data = {}

    for frame_num, txt_file in file_with_frame:
        if prev_frame_num is None:
            current_wave_id = 1
        elif frame_num - prev_frame_num > wave_gap_threshold:
            current_wave_id += 1

        boxes = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x_c, y_c, w, h = map(float, parts[:5])
                    boxes.append((cls, x_c, y_c, w, h, current_wave_id))

        if boxes:
            frame_data[frame_num] = (boxes, txt_file.name)

        prev_frame_num = frame_num

    print(f"  Total frames with detection: {len(frame_data)}")

    # Wave ID 연속화 (1, 2, 3, ...)
    used_ids = sorted({box[5] for boxes, _ in frame_data.values() for box in boxes})
    id_mapping = {old: new for new, old in enumerate(used_ids, start=1)}

    print(f"  Wave IDs: 1 ~ {len(id_mapping)}")

    saved = 0
    for frame_num, (boxes, fname) in frame_data.items():
        out_file = output_path / fname
        with open(out_file, 'w') as f:
            for cls, x_c, y_c, w, h, temp_id in boxes:
                new_id = id_mapping[temp_id]
                f.write(f"{int(cls)} {x_c} {y_c} {w} {h} {new_id}\n")
        saved += 1

    print(f"  Saved: {saved} files → {output_folder}")
    print("=" * 70)

    return len(id_mapping)


if __name__ == "__main__":
    # 🔥 필터 적용된 폴더 사용
    PRED_FOLDER = "/workspace/runs/wave/track_optimized/labels_filtered"
    OUTPUT_FOLDER = "/workspace/runs/wave/track_optimized/labels_wave"
    
    # 파도 gap threshold (25fps에서 파도 간격 고려)
    num_waves = reassign_wave_ids(PRED_FOLDER, OUTPUT_FOLDER, wave_gap_threshold=20)
    
    print(f"\n✅ Complete! Wave IDs: 1 ~ {num_waves}")