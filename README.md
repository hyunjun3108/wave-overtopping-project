# 월파(Wave Overtopping) 검출 및 추적 시스템

YOLOv9(GELAN) 기반 객체 검출과 ByteTrack + 칼만 필터 기반 추적을 결합하여, 수리모형실험 영상에서 월파를 프레임별로 검출하고 동일 개체를 추적하는 파이프라인입니다.

## 1. 전체 파이프라인 구조

프레임마다 월파를 검출하고, 이전 프레임과의 겹침(IoU) 및 특징맵 유사도(코사인 유사도)를 함께 비교하여 동일한 월파인지 판단한 뒤 지속적으로 추적합니다.

```mermaid
flowchart TD
    A[입력 프레임 t] --> B[YOLOv9 GELAN 객체 검출]
    B --> C[월파 Bounding Box + 특징맵 추출]
    C --> D{이전 프레임 t-1의\n월파 Tracklet과 비교}
    D --> E[IoU 비교\n위치/크기 겹침 정도]
    D --> F[코사인 유사도 비교\n특징맵 외형 유사도]
    E --> G{IoU & 유사도\n모두 높은가?}
    F --> G
    G -- Yes --> H[동일 월파로 판단\n기존 Track ID 유지]
    G -- No --> I[다른 월파로 판단\n새 Track ID 부여]
    H --> J[다음 프레임에서 반복 추적]
    I --> J
    J --> A
```

## 2. 학습 구조

객체 검출과 추적은 서로 다른 방식으로 학습됩니다.

```mermaid
flowchart LR
    subgraph Detection["객체 검출 학습"]
        GT[정답 위치/크기\nGround Truth] -->|오차 계산| Loss1[Bounding Box\n+ Classification Loss]
        Pred[모델 예측\nBounding Box] -->|오차 계산| Loss1
        Loss1 -->|역전파| Update1[가중치 갱신]
    end
    subgraph Tracking["객체 추적 학습"]
        Orig[원본 이미지] -->|유사도 비교| Loss2[특징 유사도 Loss]
        Aug[증강된 이미지] -->|유사도 비교| Loss2
        Loss2 -->|역전파| Update2[가중치 갱신\n원본을 잘 찾아가도록]
    end
```

## 3. 객체 검출 모델 — YOLOv9 (GELAN)

```mermaid
flowchart TB
    In[입력 영상] --> BB[GELAN Backbone\n다중 스케일 특징맵 추출]
    BB -. 학습 시에만 .-> Aux[PGI Auxiliary Branch\ngradient 정보 손실 최소화]
    BB --> Neck[PAN-FPN Neck\n다중 스케일 특징 융합]
    Neck --> Head[Detection Head\nP3 / P4 / P5]
    Head --> Out1[Bounding Box]
    Head --> Out2[Confidence Score]
    Aux -. 추론 시 미사용 .-> Head
```

- **Backbone (GELAN)**: 사전학습 가중치를 활용해 입력 영상에서 특징을 추출. PGI 기법으로 깊은 네트워크에서의 정보 손실(information bottleneck) 문제를 완화.
- **학습 단계**에서는 PGI의 Auxiliary Branch를 함께 사용해 gradient 정보 손실을 최소화하고, **추론 단계**에서는 Main Branch만 사용해 효율적으로 검출.
- **Neck (PAN-FPN)**: 서로 다른 stage의 feature map을 aggregation하여 다양한 크기의 월파를 효과적으로 검출.
- **Head**: P3/P4/P5 세 가지 스케일에서 Bounding Box와 Confidence Score를 예측.

### YOLOv7 대비 개선점

| 항목 | YOLOv7 (3차년도) | YOLOv9 (현재) |
|---|---|---|
| Backbone 구조 | CSPNet 기반 | GELAN 기반 |
| 정보 손실 대응 | 없음 | PGI로 gradient 정보 보존 |
| 파라미터 대비 성능 | 상대적으로 낮음 | 더 적은 파라미터로 더 높은 정확도 |

## 4. 객체 추적 — ByteTrack + 칼만 필터

```mermaid
flowchart TD
    Start[이전 Tracklet 상태\n위치/속도] --> KF[칼만 필터\n현재 프레임 위치 예측]
    KF --> S1{Stage 1\n고신뢰도 검출과 IoU 매칭\nconfidence >= θ}
    S1 -- 매칭 성공 --> Update[칼만 필터 상태 업데이트\n측정값 반영]
    S1 -- 매칭 실패 --> S2{Stage 2\n저신뢰도 검출과 추가 매칭\nconfidence < θ}
    S2 -- 매칭 성공 --> Update
    S2 -- 매칭 실패 --> Lost{일정 프레임 이상\n미매칭 지속?}
    Lost -- Yes --> End[추적 종료 Lost]
    Lost -- No --> Start
    Update --> Continue[다음 프레임에서 반복]
    Continue --> Start
    New[신규 검출\n매칭되지 않음] --> Init[새로운 Track ID로 초기화]
```

- **Stage 1 (고신뢰도 매칭)**: 칼만 필터로 예측한 Tracklet 위치와 고신뢰도 검출 결과 간 IoU를 계산해 매칭.
- **Stage 2 (저신뢰도 매칭)**: Stage 1에서 매칭되지 않은 Tracklet을 저신뢰도 검출과 추가로 매칭 — 가려짐(occlusion)이나 모션 블러 상황에서도 추적 연속성 유지.
- **칼만 필터 업데이트**: 매칭 성공 시 실측 위치로 상태를 갱신해 다음 프레임 예측 정확도를 향상.
- 측면 영상에는 IOU tracker를 사용했으나 크기 변화·겹침/분리 구분에 취약해, 정면 영상에는 크기 변화에 강건하고 일시적 가려짐에도 추적이 유지되는 **ByteTrack**을 적용.

## 5. 학습 기법

- **전이학습**: MS COCO 사전학습 가중치 기반 Transfer Learning.
- **데이터 증강**: Mosaic, MixUp, 색상 변환, 기하학적 변환 등.
- **다중 손실 함수**: IoU Loss, Bounding Box Regression Loss, Classification Loss를 결합한 복합 손실.

## 6. 주요 스크립트

| 파일 | 설명 |
|---|---|
| [wave_tracking_optuna_final.py](wave_tracking_optuna_final.py) | Optuna를 이용한 ByteTrack/BoT-SORT 추적 하이퍼파라미터 튜닝 및 MOT metric 평가 |
| [wave_tracking_optuna1.py](wave_tracking_optuna1.py) | 추적 하이퍼파라미터 튜닝 초기 버전 |
| [tracking_metrics4.py](tracking_metrics4.py) | 추적 성능 지표(MOTA 등) 계산 |
| [reassign_continuous3.py](reassign_continuous3.py) | Track ID 재할당 후처리 |
| [threshold_filtering](threshold_filtering) | 방파제 위로 솟구치는 정도를 기준으로 한 월파 필터링 |
| [compare_overtopping_height.py](compare_overtopping_height.py) | 검출된 Bounding Box로부터 월파고 추정 및 비교 |
| [original_with_graph.py](original_with_graph.py) / [run_with_graph.py](run_with_graph.py) | 검출·추적 결과를 그래프와 함께 시각화 |
| [create_video_from_labels.py](create_video_from_labels.py) | 라벨 결과를 영상으로 변환 |
| [bytetrack_best.yaml](bytetrack_best.yaml) / [tracker_final.yaml](tracker_final.yaml) | 튜닝된 ByteTrack 설정 |

> 대용량 결과 영상(mp4)은 저장소 용량 문제로 `.gitignore` 처리되어 있으며, 로컬에서만 관리됩니다.
