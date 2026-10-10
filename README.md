# Physical AI & Mobility Simulation

Physical AI와 모빌리티 시뮬레이션 수업의 과제 및 프로젝트를 정리하는 저장소입니다.
각 폴더에 구현 코드, 수행 내역, 실험 결과를 함께 보관합니다.

## 과제 수행 내역

| 과제 | 주제 | 주요 수행 내용 | 상세 |
| --- | --- | --- | --- |
| Task 1 | NVIDIA Isaac Sim 기반 센서 융합으로 주변 환경 인식하기 | RGB-D로 목표 객체 식별 및 로봇 작업반경 판단, 3가지 배치 실험, 모니터·반경 시각화 구현 | [과제 요약](task1_sensor_fusion/README.md) |

## 파일 구조

```text
physicalAI-mobility-simulation/
├── README.md                          # 수업 과제·프로젝트 목록
├── .gitignore                         # 캐시·가상환경·자동 생성 결과 제외
└── task1_sensor_fusion/
    ├── README.md                      # 수행 내역 및 결과 요약
    ├── sensor_fusion_reachability.py   # Isaac Sim 실행 코드
    ├── results/
    │   ├── results.csv                # 객체별 측정 데이터
    │   ├── *_annotated_rgb.png        # 객체 검출·판정 이미지 5장
    │   ├── *_depth_vis.png            # Depth 시각화 5장
    │   └── *_summary.txt              # 프레임별 결과 요약 5개
    └── screenshots/                   # 시뮬레이션 실행 화면 10장
```

추가 과제는 `task2_주제명/`, 프로젝트는 `projects/프로젝트명/`으로 확장하고 위 목록에 연결합니다.
