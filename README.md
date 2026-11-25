# A Minimal Subset Approach for Informed Keyframe Sampling in Large-Scale SLAM — Project Page

공간지능특론 프로젝트 페이지 (정석환 · 윤승희)

원 논문: N. Stathoulopoulos, C. Kanellakis, G. Nikolakopoulos, "A Minimal Subset Approach for Informed Keyframe Sampling in Large-Scale SLAM," IEEE RA-L, vol. 11, no. 1, pp. 738–745, 2026.

- `index.html` — 프로젝트 페이지 (정적 HTML, 빌드 불필요)
- `assets/` — 그림 및 데이터셋 예시 이미지 (KITTI, DiTer, SeePerSea)
- `msa_slam/`, `run.py`, `configs/` — Code Demo 파이프라인

## Code Demo: Front-end / MSA / Back-end

```
LiDAR scan ─▶ FRONT-END (교체 가능) ─▶ KEYFRAME SAMPLING (고정) ─▶ BACK-END (교체 가능)
              KISS-ICP 오도메트리        MSA (공식 코드)              6. OverlapTransformer 장소 인식
                                         비교: all / fixed / entropy   7. small_gicp 루프 검증
                                               / spaciousness          8. GTSAM iSAM2 포즈 그래프 최적화
```

| 모듈 | 파일 | 구현 |
|---|---|---|
| Front-end | [msa_slam/frontend.py](msa_slam/frontend.py) | `Frontend` 인터페이스, `KissICPFrontend` |
| Descriptor | [msa_slam/descriptors.py](msa_slam/descriptors.py) | OverlapTransformer (저자 공개 KITTI 가중치) |
| Keyframe Sampling | [msa_slam/sampling.py](msa_slam/sampling.py) | 공식 [opt-key](https://github.com/LTU-RAI/opt-key) 코드 호출 (`RMIP_Online`, `FixedDistanceOnline`, `EntropyOnline`, `SpaciousnessOnline`), `all` = w/o MSA |
| Back-end | [msa_slam/backend.py](msa_slam/backend.py) | `Backend` 인터페이스, `LoopClosureBackend` |
| 평가 | [msa_slam/metrics.py](msa_slam/metrics.py) | ATE, PR/F1-max (공식 평가식), 루프 정확도 |

공식 opt-key 코드는 키프레임 샘플링과 PR 평가만 제공하고, GT 포즈를 입력으로 씁니다. 여기서는 그 앞에 KISS-ICP 오도메트리를, 뒤에 루프 검증과 iSAM2를 붙여서 SLAM 전체에서 비교합니다.

### 설치

```bash
git submodule update --init          # third_party/opt-key (MIT), third_party/OverlapTransformer (GPL-3)
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
# OverlapTransformer KITTI 가중치 (README of haomo-ai/OverlapTransformer)
mkdir -p weights && curl -L -o weights/pretrained_overlap_transformer.pth.tar \
  "https://drive.usercontent.google.com/download?id=1FNrx9pcDa9NF7z8CFtuTWyauNkeSEFW4&export=download&confirm=t"
```

데이터: KITTI odometry (`dataset/sequences/00/velodyne`, `dataset/poses/00.txt`). 경로는 [configs/kitti00.yaml](configs/kitti00.yaml)의 `dataset.root`.

### 실행

```bash
.venv/bin/python run.py --config configs/kitti00.yaml                        # 설정 파일의 모든 방법
.venv/bin/python run.py --config configs/kitti00.yaml --methods all msa      # w/o MSA vs w/ MSA
.venv/bin/python run.py --config configs/kitti00.yaml --max-frames 1000      # 앞부분만 빠르게
```

1단계(KISS-ICP + 디스크립터)는 모든 방법이 공유하므로 `results/<seq>_<n>/stage1_cache.npz`에 한 번만 계산해 둡니다. 2단계에서 각 방법이 프레임을 순서대로 받아 온라인으로 키프레임을 고르고, 고른 프레임만 Back-end로 갑니다.

### 프로젝트 페이지에서 시연 (Try It Yourself)

```bash
.venv/bin/python server.py          # http://localhost:8000 (이 PC에서만 접속 가능)
```

페이지의 Try It Yourself에서 서버에 있는 KITTI 00 샘플을 고르거나, 스캔 폴더(`.bin` KITTI 형식 또는 `.pcd`, 파일 이름 순)를 끌어다 놓고 실행합니다.
`poses.txt`(KITTI 형식, 스캔당 한 줄)를 함께 넣으면 ATE · F1-max까지 계산하고, `calib.txt`에 `Tr:`가 있으면 KITTI처럼 카메라 → LiDAR 좌표로 바꿉니다.
한 번에 1.7 GB(KITTI 약 900개)까지만 올립니다. 이보다 많이 고르면 앞부분만 올리는데, 이 PC의 Chrome에서 업로드한 데이터가 탭 메모리에 쌓여 약 2 GB에서 페이지가 멈췄기 때문입니다. 더 긴 시퀀스는 `run.py`로 돌리세요.
업로드는 `results/uploads/`, 결과는 `results/demo/`에 저장됩니다(`--clean`으로 지움). 파일로 연 페이지(`file://`)에서는 실행 버튼이 꺼져 있습니다.

결과: `results/<seq>_<n>/summary.md`(비교표), `trajectories.png`, 방법별 `metrics.json`, `poses_opt.txt`(KITTI 형식), `keyframes.txt`, `loops.json`.

지표:
- **Memory raw**: 키프레임 원본 스캔 크기 합 (지도에 스캔을 저장할 때). **Memory back-end**: 실제로 보관한 디스크립터 + 0.5 m 다운샘플 스캔.
- **Time**: 키프레임 샘플링 + Back-end(장소 인식, 검증, PGO). 1단계 시간은 공통이라 제외.
- **F1-max**: 모든 프레임이 키프레임 지도에 질의 (공식 평가식, δ = 3 m). 공식 코드와 달리 후보를 GT 위치로 좁히지 않고 디스크립터만으로 전체에서 찾습니다.
- **ATE**: 모든 프레임 궤적 (키프레임 아닌 프레임은 직전 키프레임 + 오도메트리), SE(3) 정렬 후 RMSE. **ATE impr.** = KISS-ICP 대비 개선율.
- **Loop prec.**: 채택한 루프 중 상대 포즈가 GT와 δ(3 m) 이내로 맞는 비율. KITTI GT가 일부 재방문 지점에서 ~2.5 m 어긋나 있어서(GT 상대 포즈에서 시작한 GICP가 우리 루프 결과로 수렴) 1 m 기준은 맞는 루프도 틀렸다고 셉니다.

### 결과: KITTI 00 (4,541 프레임, CPU)

| Method | Keyframes | Memory raw (MB) | Sampling (s) | Back-end (s) | Loops | F1-max | ATE (m) | ATE impr. (%) |
|---|---|---|---|---|---|---|---|---|
| all (w/o MSA) | 4541 | 8827 | 0.0 | 34.2 | 824 | 0.952 | 0.89 | 74.6 |
| **msa** | **962** | **1869** | 62.8 | **3.6** | 148 | 0.917 | 0.89 | 74.7 |
| fixed_1 | 2717 | 5280 | 0.2 | 12.5 | 514 | 0.948 | 0.90 | 74.4 |
| fixed_2 | 1541 | 2994 | 0.2 | 5.9 | 282 | 0.934 | 0.88 | 74.8 |
| entropy | 1722 | 3342 | 77.5 | 8.0 | 309 | 0.920 | 0.88 | 74.8 |
| spaciousness | 1183 | 2300 | 52.8 | 5.0 | 202 | 0.914 | 0.90 | 74.4 |

1단계: KISS-ICP 55.9 s, OverlapTransformer 272.7 s(CPU). KISS-ICP만 쓴 ATE는 3.50 m. 채택된 루프는 모든 방법에서 GT와 3 m 이내.
