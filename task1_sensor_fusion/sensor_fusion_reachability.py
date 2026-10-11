# -*- coding: utf-8 -*-
"""
sensor_fusion_reachability.py
=============================
RGB-D 센서 융합으로 "Target 물체가 로봇팔 작업 반경 안에 있는가?"를 판단하는
Isaac Sim 5.0 / 5.1 Standalone 예제.

■ 이 코드가 답하는 질문
  1) 여러 물체 중 어느 것이 Target 인가?          → RGB 로 판단 (색)
  2) 그 Target 은 로봇에서 얼마나 떨어져 있는가?   → Depth 로 판단 (거리)
  3) 그 거리가 MAX_REACH_M 이하인가?              → 최종 판정 (REACHABLE / OUT OF RANGE)

■ 처리 흐름 (한 카메라의 같은 frame 에서 RGB 와 Depth 를 함께 사용)
  1. RGB 영상   : 색(HSV)으로 물체를 찾아 mask 와 중심 pixel (u, v) 를 구한다
  2. Depth 영상 : 1 에서 구한 mask 안쪽 pixel 의 depth 만 모아 중앙값 Z 를 구한다
  3. 3D 복원    : (u, v, Z) → 카메라 좌표 → world 좌표 → 로봇 base 좌표
  4. 판정       : 로봇 base 원점까지의 거리 ≤ MAX_REACH_M 이면 REACHABLE

■ 알아둘 점
  - REACHABLE 은 "로봇 base 를 중심으로 한 반지름 MAX_REACH_M 구(sphere) 안에 있다"는 뜻일 뿐이다.
    실제로 IK 를 풀 수 있는지, 잡을 수 있는지, 충돌 없이 접근할 수 있는지는 보장하지 않는다.
  - 시뮬레이터가 알고 있는 물체의 실제 위치(Ground Truth)는 오차 확인용 출력에만 쓰고, 판정에는 쓰지 않는다.

■ 실행 방법 (Windows, Isaac Sim 설치 폴더에서)
  python.bat C:\\path\\to\\sensor_fusion_reachability.py --scenario A

■ 핵심 코드 위치 (에디터에서 "[CORE-" 로 검색)
  [CORE-1] acquire_rgbd_frame()     같은 frame 의 RGB + Depth 가져오기
  [CORE-2] detect_objects_rgb()     RGB 로 물체 찾기                 (무엇인가?)
  [CORE-3] extract_object_depth()   물체 mask 안쪽의 depth 만 사용    ← 두 센서가 만나는 지점
  [CORE-4] pixel_depth_to_3d()      pixel + depth → 3D 위치          (어디에 있나?)
  [CORE-5] world_to_robot_base()    world 좌표 → 로봇 base 좌표
  [CORE-6] evaluate_reachability()  거리 ≤ MAX_REACH_M ? → 최종 판정
  [CORE-7] process_rgbd()           CORE-2 ~ 6 을 한 frame 에 순서대로 실행
  [CORE-8] depth_only_baseline()    비교용: Depth 만 쓰면 어떻게 되는가
  [VIZ]    ReachVisualizer          Viewport 에 작업 반경과 Target 연결선 그리기 (시각화 전용)
"""

# =============================================================================
# 0. 명령행 인자   ※ SimulationApp 을 만들기 전에 읽어야 한다
# =============================================================================
import argparse
import csv
import math
import os
import sys
import time
import traceback


def _parse_args():
    parser = argparse.ArgumentParser(description="RGB-D sensor fusion reachability demo (Isaac Sim 5.x)")
    parser.add_argument("--scenario", choices=["A", "B", "C", "a", "b", "c"], default=None,
                        help="initial object layout preset (A: in range, B: out of range, C: distractor closer)")
    parser.add_argument("--target", default=None, help="override TARGET_NAME (e.g. BlueBox)")
    parser.add_argument("--max-reach", type=float, default=None, help="override MAX_REACH_M [m]")
    parser.add_argument("--no-monitor", action="store_true", help="disable the in-app omni.ui Fusion Monitor panel")
    parser.add_argument("--cv-windows", action="store_true",
                        help="also open OpenCV windows (annotated RGB + depth); off by default")
    parser.add_argument("--no-depth-window", action="store_true", help="with --cv-windows: skip the depth window")
    parser.add_argument("--save", action="store_true", help="periodically save results to output/")
    # Isaac Sim(Kit)이 자체 인자를 덧붙일 수 있으므로, 모르는 인자는 무시한다
    args, _unknown = parser.parse_known_args()
    return args


ARGS = _parse_args()

# =============================================================================
# 1. 설정값   ※ 실험 조건을 바꿀 때는 이 블록만 수정하면 된다
# =============================================================================

# ---- 실험 조건 ----------------------------------------------------------------
SCENARIO = (ARGS.scenario or "A").upper()          # 물체 배치 프리셋 "A" / "B" / "C" (실행 중 패널 버튼으로도 변경)
TARGET_NAME = ARGS.target or "RedContainer"        # 판정 대상 물체 이름 (OBJECT_SPECS 의 name 중 하나)
# 로봇 base 를 중심으로 한 구(sphere)의 반지름 [m].
# 이 실험을 위해 단순화한 "작업 가능 반경"이며, 제조사 공식 사양값이 아니다.
MAX_REACH_M = ARGS.max_reach if ARGS.max_reach is not None else 0.70

# ---- 작업대 (world 좌표: Z 축이 위쪽, 단위 m) -------------------------------------
TABLE_TOP_Z = 0.40                                  # 작업대 윗면 높이 [m]
# 작업대 범위: x -1.40 ~ 2.20, y -1.50 ~ 1.50
#   - 로봇 base(0, 0)에서 가장 가까운 가장자리까지 1.40 m → 반경 약 1.3 m 까지는 작업대 위에 그려진다.
#   - 앞쪽 가장자리(y = -1.5)는 카메라 위치(y = -1.75)보다 안쪽에 있다.
TABLE_CENTER_XY = (0.40, 0.0)
TABLE_SIZE_XY = (3.60, 3.00)
# 작업대 색: 채도가 낮은 짙은 회색.
#   색 검출은 채도(S)가 100 이상인 pixel 만 보므로, R·G·B 를 비슷하게 맞춰 채도를 낮게 유지한다.
#   (파란 기가 강하면 BLUE 로 잘못 검출될 수 있음)
TABLE_COLOR = (0.18, 0.19, 0.21)

# ---- 배경 (보기 위한 장식일 뿐, 판정 로직과는 무관) ----------------------------------------
# 창고 작업장 구성: 콘크리트 바닥 + 노란 안전선, 블록/금속 벽, 철골 기둥, 지붕 트러스, 파렛트 랙, 카메라 스탠드
#
# 색을 고를 때 지킨 규칙 (배경이 물체로 잘못 검출되지 않도록):
#   - 대부분 회색/금속색                   → 채도가 낮아 검출 대상에서 자연히 빠짐
#   - 안전선 노랑 (hue ≈ 23)               → RED(0~10, 170~179), GREEN(40~85) 범위 밖
#   - 골판지/파렛트 갈색 (hue ≈ 17, S ≈ 95) → RED 범위 밖이고, RED 채도 하한(120)보다도 낮음
#   - 빨강·파랑·초록 소품은 두지 않음
#     (Isaac 기본 창고 asset 을 쓰지 않은 이유: 색 있는 소품이 오검출되고, 지붕이 조명을 가림)
ENVIRONMENT_STYLE = "warehouse"             # "warehouse" = 창고 배경 / "plain" = 회색 바닥만
FLOOR_SIZE_M = 60.0                         # 바닥 한 변 길이 [m] (충돌체 있음)
FLOOR_JOINT_COLOR = (0.20, 0.20, 0.20)      # 바닥 기본색 = 콘크리트 판 사이 줄눈 색
FLOOR_PANEL_SIZE_M = 2.0                    # 콘크리트 판 한 변 [m]
FLOOR_CONCRETE_COLORS = ((0.47, 0.47, 0.45), (0.50, 0.49, 0.47), (0.44, 0.44, 0.43))   # 판마다 번갈아 쓰는 3가지 톤
ROOM_CENTER_XY = TABLE_CENTER_XY
ROOM_SIZE_XY = (16.0, 14.0)                 # 벽 안쪽 크기 [m]
ROOM_WALL_HEIGHT_M = 6.0
WALL_BLOCK_HEIGHT_M = 1.2                   # 벽 아래쪽 콘크리트 블록 높이 [m]
WALL_BLOCK_COLOR = (0.52, 0.52, 0.50)
WALL_CLADDING_COLOR = (0.68, 0.69, 0.70)    # 벽 위쪽 금속 골강판
STEEL_DARK_COLOR = (0.28, 0.29, 0.31)       # 기둥 · 지붕 트러스 · 카메라 스탠드
STEEL_GALV_COLOR = (0.60, 0.61, 0.63)       # 아연도금 강재 (랙 기둥 · 작업대 다리)
SAFETY_YELLOW = (0.95, 0.75, 0.05)          # 바닥 안전선 · 기둥 보호대 · 랙 빔 · 작업대 테두리
CARDBOARD_COLOR = (0.55, 0.46, 0.34)
PALLET_WOOD_COLOR = (0.50, 0.42, 0.32)
WORKBENCH_TOP_THICKNESS_M = 0.06
# (선택) 실제 텍스처 재질(MDL) 사용하기
#   Content Browser 에서 .mdl 파일 우클릭 → Copy URL 후 아래에 붙여넣는다. 빈 문자열이면 사용 안 함.
#   빨강·파랑·초록 계열 재질은 색 검출을 방해하므로 회색 콘크리트/금속 계열만 쓸 것.
MDL_OVERRIDES = {"floor": "", "walls": "", "workbench_top": ""}

# ---- 로봇 ----------------------------------------------------------------------
ROBOT_PRIM_PATH = "/World/Franka"
# 아래 두 값은 처음 scene 을 만들 때 로봇을 놓는 위치로만 쓴다.
# 좌표 변환에는 이 숫자 대신, 매 frame stage 에서 읽은 실제 로봇 base 위치를 사용한다.
ROBOT_INITIAL_POSITION = (0.0, 0.0, TABLE_TOP_Z)
ROBOT_INITIAL_YAW_DEG = 0.0
# Franka 대기 자세 (팔 관절 7개 + 그리퍼 2개). 화면 구성용이며, 로봇 동작/IK 는 다루지 않는다.
ROBOT_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
# Franka USD 파일 경로 — 아래 순서대로 시도한다.
#   1) 환경변수 FRANKA_USD_PATH 에 지정한 경로 (로컬 또는 Nucleus 전체 경로)
#   2) Isaac Sim 5.0/5.1 기본 경로: /Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd
#   3) Isaac Sim 4.x 기본 경로:     /Isaac/Robots/Franka/franka.usd
FRANKA_USD_OVERRIDE = os.environ.get("FRANKA_USD_PATH", "")
FRANKA_USD_RELATIVE_CANDIDATES = [
    "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "/Isaac/Robots/Franka/franka.usd",
    "/Isaac/Robots/Franka/franka_instanceable.usd",
]

# ---- RGB-D 카메라 (로봇 밖에 고정 설치, eye-to-hand) -----------------------------
CAMERA_PRIM_PATH = "/World/RGBD_Camera"
# Franka 전체(대기 자세 윗부분 포함)와 A/B/C 시나리오의 모든 물체가 한 화면에 들어오도록 맞춘 위치
CAMERA_POSITION = (1.00, -1.75, 1.45)               # 카메라 위치 (world) [m]
CAMERA_TARGET = (0.45, 0.05, 0.55)                  # 카메라가 바라보는 점 (world) [m]
CAMERA_RESOLUTION = (640, 480)                      # 해상도 (가로, 세로) [px]
CAMERA_FREQUENCY = 20                               # 카메라 영상 갱신 주기 [Hz]
CAMERA_FOCAL_LENGTH_MM = 18.0                       # 초점거리 [mm] → 수평 화각 약 60°
CAMERA_HORIZONTAL_APERTURE_MM = 20.955              # 센서 가로 크기 [mm]
CAMERA_CLIPPING_RANGE = (0.05, 50.0)                # 렌더링 거리 범위 (최소, 최대) [m]

# ---- 물체 ----------------------------------------------------------------------
# 각 물체가 TARGET 인지 아닌지는 TARGET_NAME 으로 자동 결정된다.
# color_key: 아래 COLOR_RANGES 중 어떤 색 범위로 이 물체를 찾을지
OBJECT_SPECS = [
    {"name": "RedContainer", "visual": "red opaque plastic container (cylinder)",
     "shape": "cylinder", "radius": 0.045, "height": 0.12,
     "display_color": (0.85, 0.05, 0.05), "color_key": "RED", "roughness": 0.6, "metallic": 0.0},
    {"name": "BlueBox", "visual": "blue plastic box (cuboid)",
     "shape": "cuboid", "size": (0.10, 0.10, 0.10),
     "display_color": (0.05, 0.15, 0.85), "color_key": "BLUE", "roughness": 0.6, "metallic": 0.0},
    {"name": "GreenMetalSphere", "visual": "green metallic-look sphere",
     "shape": "sphere", "radius": 0.05,
     "display_color": (0.05, 0.70, 0.10), "color_key": "GREEN", "roughness": 0.35, "metallic": 0.5},
]

# 시나리오별 물체 초기 위치 (x, y) [m]. 높이 z 는 작업대 위에 놓이도록 자동 계산한다.
# 프리셋은 시작할 때, 그리고 사용자가 시나리오를 바꿀 때 한 번만 적용된다.
# 그 뒤에는 UI 에서 물체를 자유롭게 옮길 수 있다 (매 frame 위치를 되돌리지 않음).
SCENARIO_PRESETS = {
    # A: Target 이 반경 안 → REACHABLE RANGE
    "A": {"RedContainer": (0.45, 0.10), "BlueBox": (0.70, -0.35), "GreenMetalSphere": (0.20, 0.45)},
    # B: Target 이 반경 밖 → OUT OF RANGE
    "B": {"RedContainer": (1.00, 0.25), "BlueBox": (0.60, -0.20), "GreenMetalSphere": (0.45, 0.35)},
    # C: 방해 물체(BlueBox)가 로봇·카메라에 가장 가깝고, Target 은 반경 밖
    #    → Depth 만 보고 "가장 가까운 물체 = Target" 이라고 가정하면 틀린다. 융합 결과는 OUT OF RANGE.
    "C": {"RedContainer": (0.95, 0.30), "BlueBox": (0.30, -0.35), "GreenMetalSphere": (0.75, 0.45)},
}
if SCENARIO not in SCENARIO_PRESETS:
    SCENARIO = "A"
OBJECT_INITIAL_POSITIONS = SCENARIO_PRESETS[SCENARIO]

# ---- RGB 색 검출 (OpenCV HSV 범위: H 0~179, S 0~255, V 0~255) ----------------------
# 색마다 (하한, 상한) 쌍의 리스트. 형식: ((H, S, V), (H, S, V))
COLOR_RANGES = {
    "RED":   [((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (179, 255, 255))],  # 빨강은 hue 0 과 179 양 끝에 걸쳐 있어 둘로 나눔
    "BLUE":  [((100, 120, 50), (130, 255, 255))],
    "GREEN": [((40, 100, 50), (85, 255, 255))],
}
MIN_MASK_PIXELS = 60            # mask 가 이보다 작으면 미검출로 처리 (노이즈이거나 거의 가려진 경우)

# ---- Depth ---------------------------------------------------------------------
MIN_VALID_DEPTH_PIXELS = 20     # mask 안의 유효 depth pixel 이 이보다 적으면 INVALID DEPTH
DEPTH_MAX_VALID_M = 20.0        # 이보다 먼 depth 는 배경으로 보고 버린다 [m]
DEPTH_ERODE_MASK = True         # mask 가장자리를 1px 깎아서 사용 (물체 경계에 섞인 배경 depth 제거)

# ---- pixel → 3D 변환 -----------------------------------------------------------
USE_OFFICIAL_UNPROJECTION = True    # Isaac Sim 공식 함수 get_world_points_from_image_coords() 를 우선 사용
UNPROJECTION_CHECK_TOL_M = 0.01     # 공식 함수 결과와 직접 계산 결과의 허용 오차 [m] (넘으면 경고)

# ---- 비교용: Depth 만 쓰는 방식 ----------------------------------------------------
# RGB 없이 "작업대 위로 솟아 있는 표면 중 가장 가까운 점"을 찾는다. → 그게 어떤 물체인지는 모른다.
DEPTH_ONLY_STRIDE = 2                       # 계산량을 줄이기 위해 2 pixel 마다 1개만 사용
DEPTH_ONLY_HEIGHT_BAND = (0.015, 0.25)      # 작업대 윗면 기준, 이 높이 범위의 점만 물체 후보로 본다 [m]
DEPTH_ONLY_ROBOT_EXCLUSION_RADIUS = 0.20    # 로봇 base 주변 이 반경(XY) 안은 로봇 몸체로 보고 제외 [m]

# ---- 결과 패널 (Isaac Sim 안의 "RGB-D Fusion Monitor") ------------------------------
SHOW_FUSION_MONITOR = not ARGS.no_monitor  # 패널 표시 여부 (--no-monitor 로 끔)
MONITOR_WIDTH = 380                         # 패널 크기 [px]
MONITOR_HEIGHT = 840
MONITOR_DOCK_RIGHT_OF_VIEWPORT = True       # Viewport 오른쪽에 붙이기 (실패하면 떠 있는 창으로 유지)
MONITOR_DOCK_RATIO = 0.27                   # 붙였을 때 패널이 차지하는 폭 비율 (어색하면 위 값을 False 로)

# ---- 작업 반경 시각화 (Viewport 에 선만 그림. 물리 / USD / 판정에 영향 없음) ----------------
REACH_VIZ_ENABLED = True
REACH_VIZ_TABLE_OFFSET_M = 0.005            # 작업대 위 원을 표면보다 5 mm 띄워 그림 (표면과 겹쳐 깜빡이는 현상 방지)
# 색: (r, g, b, a), 각 0~1
#   경계선은 채도 낮은 하늘색으로 그려, 혹시 카메라 영상에 찍혀도 RED/GREEN/BLUE 로 검출되지 않게 했다.
REACH_VIZ_BOUNDARY_COLOR = (0.65, 0.85, 0.95, 0.55)     # 반구 wireframe
REACH_VIZ_TABLE_RING_COLOR = (0.70, 0.88, 0.97, 0.85)   # 작업대 위 원
REACH_VIZ_BASE_COLOR = (1.0, 1.0, 1.0, 1.0)             # 로봇 base 표시
REACH_VIZ_IN_COLOR = (0.15, 0.90, 0.25, 1.0)            # Target 연결선 (반경 안)
REACH_VIZ_OUT_COLOR = (1.00, 0.20, 0.20, 1.0)           # Target 연결선 (반경 밖)
REACH_VIZ_BOUNDARY_WIDTH = 1.5             # 선 두께 [px]
REACH_VIZ_TABLE_RING_WIDTH = 2.5
REACH_VIZ_TARGET_LINE_WIDTH = 2.0          # 얇게 유지: 카메라에 찍혀도 색 검출의 노이즈 제거(3x3) 단계에서 지워짐
REACH_VIZ_TARGET_MARKER_SIZE = 8.0         # Target 점 크기 [px]
REACH_VIZ_LATITUDES_DEG = (30.0, 60.0)     # 반구에 그릴 위도선 [deg]

# ---- 기타 출력 / 저장 --------------------------------------------------------------
SHOW_OVERLAY = bool(ARGS.cv_windows)        # OpenCV 결과 이미지 창 (--cv-windows 로 켬, 기본 끔)
SHOW_DEPTH_WINDOW = bool(ARGS.cv_windows) and not ARGS.no_depth_window  # OpenCV depth 이미지 창 (보조)
SHOW_GT_VALIDATION = True                   # 실제 위치(GT)와의 오차 출력 (판정에는 사용 안 함)
PRINT_INTERVAL_FRAMES = 15                  # 콘솔 결과 출력 간격 [step]
WARMUP_FRAMES = 20                          # 시작 직후 카메라 버퍼가 비어 있는 구간은 건너뜀 [step]
SAVE_RESULTS = bool(ARGS.save)              # True 면 SAVE_INTERVAL_FRAMES 마다 output/ 에 자동 저장 (--save)
SAVE_INTERVAL_FRAMES = 300
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPTH_VIS_RANGE_M = (1.0, 3.5)              # depth 이미지 색 표시 범위 [m]

# ---- 시뮬레이션 ------------------------------------------------------------------
PHYSICS_DT = 1.0 / 60.0                     # 물리 계산 간격 [s]
RENDERING_DT = 1.0 / 60.0                   # 렌더링 간격 [s]
DOME_LIGHT_INTENSITY = 700.0                # 전체 환경광 세기
KEY_LIGHT_INTENSITY = 1500.0                # 주 조명(햇빛) 세기

# =============================================================================
# 2. SimulationApp 시작   ※ 다른 omni / isaacsim 모듈을 import 하기 전에 반드시 먼저
# =============================================================================
from isaacsim import SimulationApp  # noqa: E402

# window_width/height: 앱 창 크기,  width/height: Viewport 렌더링 해상도
simulation_app = SimulationApp({"headless": False, "window_width": 1600, "window_height": 900,
                                "width": 1280, "height": 720})

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # noqa: E402


def log(msg, level="INFO"):
    print(f"[SensorFusion][{level}] {msg}", flush=True)


class RateLimitedLog:
    """같은 경고가 매 frame 콘솔에 반복 출력되지 않도록, 경고 종류(key)별로 최소 간격을 둔다."""

    def __init__(self, interval_s=3.0):
        self.interval_s = interval_s
        self.last = {}

    def warn(self, key, msg, level="WARN"):
        now = time.time()
        if now - self.last.get(key, -1e9) >= self.interval_s:
            self.last[key] = now
            log(msg, level)


RL = RateLimitedLog()

# ---- Isaac Sim 5.x 필수 모듈 ------------------------------------------------------
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.sensors.camera")      # 기본으로 켜져 있지만 확실히 하기 위해 다시 켬
except Exception as _e:  # noqa: BLE001
    log(f"enable_extension('isaacsim.sensors.camera') failed (may already be enabled): {_e}", "WARN")

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, FixedCylinder, FixedSphere, VisualCuboid, VisualCylinder
    from isaacsim.core.prims import SingleXFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.sensors.camera import Camera
except ImportError as _e:
    log("Failed to import Isaac Sim 5.x core modules. Run this file with Isaac Sim 5.0/5.1 python.bat "
        f"(or the Isaac Sim pip environment). Detail: {_e}", "ERROR")
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

# ---- 선택 모듈 (없으면 해당 기능만 꺼지고 나머지는 정상 동작) ------------------------------
try:
    from isaacsim.core.api.objects import GroundPlane
except ImportError:
    GroundPlane = None
try:
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    SingleArticulation = None
    ArticulationAction = None
try:
    from isaacsim.core.api.materials import PreviewSurface
except ImportError:
    PreviewSurface = None

# asset 저장소 경로 함수: 5.x 는 isaacsim.storage.native, 4.x 는 isaacsim.core.utils.nucleus 에 있다
_get_assets_root_path = None
try:
    from isaacsim.storage.native import get_assets_root_path as _get_assets_root_path
except ImportError:
    try:  # 4.x 위치 (5.x 에서는 없을 수 있음)
        from isaacsim.core.utils.nucleus import get_assets_root_path as _get_assets_root_path
    except ImportError:
        _get_assets_root_path = None

# ---- debug_draw: Viewport 에 선/점을 그리는 도구 (작업 반경 시각화용, 선택) -------------------
# 사용법:
#   draw = _debug_draw.acquire_debug_draw_interface()
#   draw.draw_lines(시작점들, 끝점들, 색들, 두께들)  /  draw.draw_points(점들, 색들, 크기들)
#   draw.clear_lines()  /  draw.clear_points()  /  _debug_draw.release_debug_draw_interface(draw)
_debug_draw = None
try:
    from isaacsim.core.utils.extensions import enable_extension as _enable_ext
    _enable_ext("isaacsim.util.debug_draw")
    from isaacsim.util.debug_draw import _debug_draw
except Exception as _e:  # noqa: BLE001
    _debug_draw = None
    log(f"isaacsim.util.debug_draw not available ({_e}); reach visualization disabled.", "WARN")

# ---- omni.ui: Isaac Sim 안에 결과 패널을 띄우는 UI 라이브러리 (선택) ----------------------
try:
    import omni.ui as ui
except Exception as _e:  # noqa: BLE001
    ui = None
    log(f"omni.ui not available ({_e}); Fusion Monitor panel disabled (console only).", "WARN")

# ---- OpenCV (선택. 없으면 numpy 로 색 검출하고, 이미지 창 / PNG 저장은 꺼짐) --------------
try:
    import cv2
except Exception as _e:  # noqa: BLE001
    cv2 = None
    log(f"OpenCV not available ({_e}). Detection falls back to numpy HSV; overlay windows/PNG saving disabled. "
        "Install: python.bat -m pip install opencv-python==4.10.0.84", "WARN")


# =============================================================================
# 3. 수학 helper   ※ quaternion 은 Isaac Sim 과 같은 (w, x, y, z) 순서
# =============================================================================
def quat_wxyz_to_rotmat(q):
    """quaternion (w, x, y, z) → 3x3 회전행렬 R.   (자식 좌표의 점 p 를 부모 좌표로: R @ p)"""
    w, x, y, z = [float(c) for c in np.asarray(q, dtype=float).reshape(4)]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat_wxyz(R):
    """3x3 회전행렬 → quaternion (w, x, y, z).  (수치적으로 안정한 Shepperd 방법)"""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# Z 축 회전각 yaw [deg] → quaternion (w, x, y, z)
def yaw_to_quat_wxyz(yaw_deg):
    h = math.radians(yaw_deg) / 2.0
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def look_at_quat_usd(eye, target, world_up=(0.0, 0.0, 1.0)):
    """eye 위치에서 target 을 바라보는 카메라 방향을 quaternion (w, x, y, z) 으로 계산한다.

    USD 카메라 축 규약: 로컬 -Z = 바라보는 방향, +Y = 화면 위쪽, +X = 화면 오른쪽"""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - eye
    if np.linalg.norm(f) < 1e-9:
        raise ValueError("CAMERA_POSITION and CAMERA_TARGET must be different points")
    f /= np.linalg.norm(f)
    up = np.asarray(world_up, dtype=float)
    if abs(float(np.dot(f, up))) > 0.999:          # 바로 위/아래를 보면 외적이 0 이 되므로 up 벡터를 바꿈
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.column_stack([r, u, -f])                # 각 열 = 카메라의 x, y, z 축이 world 에서 가리키는 방향
    return rotmat_to_quat_wxyz(R)


def fmt(v, nd=2, width=0):
    """숫자를 소수점 nd 자리 문자열로 바꾼다. 값이 None / NaN 이면 'N/A'."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        s = "N/A"
    else:
        s = f"{v:.{nd}f}"
    return s.rjust(width) if width else s


# 3D 벡터 → "(+x, +y, +z)" 문자열. None 이면 'N/A'
def fmt_vec(v, nd=2):
    if v is None:
        return "N/A"
    return "(" + ", ".join(f"{float(c):+.{nd}f}" for c in v) + ")"


# 물체 이름이 TARGET_NAME 과 같으면 "TARGET", 아니면 "NON_TARGET"
def object_role(name):
    return "TARGET" if name == TARGET_NAME else "NON_TARGET"


def object_center_z(spec):
    """물체를 작업대 위에 올려놓았을 때 물체 중심의 높이 z [m]."""
    if spec["shape"] == "cuboid":
        half = spec["size"][2] / 2.0
    elif spec["shape"] == "cylinder":
        half = spec["height"] / 2.0
    else:
        half = spec["radius"]
    return TABLE_TOP_Z + half


# =============================================================================
# 4. Scene 구성 (바닥 · 배경 · 작업대 · 조명 · 로봇 · 물체)
# =============================================================================
def get_assets_root_safe():
    """Isaac Sim asset 저장소(Nucleus / S3) 경로를 가져온다. 실패하면 None 을 반환하고 프로그램은 계속 진행."""
    if _get_assets_root_path is None:
        log("get_assets_root_path() not importable (isaacsim.storage.native missing?).", "ERROR")
        return None
    try:
        log("Resolving Isaac Sim assets root path (can take a while on first run)...")
        root = _get_assets_root_path()
    except Exception as e:  # noqa: BLE001
        log(f"get_assets_root_path() raised: {e}", "ERROR")
        return None
    if not root:
        log("get_assets_root_path() returned None. Check network / Nucleus / local asset pack settings.", "ERROR")
        return None
    log(f"Assets root: {root}")
    return root


def _prim_has_composed_children(prim):
    """USD reference 가 실제로 불러와졌는지(하위 prim 이 생겼는지) 확인한다."""
    try:
        return len(list(prim.GetFilteredChildren(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))) > 0
    except Exception:  # noqa: BLE001
        return len(prim.GetChildren()) > 0


def load_franka_reference(stage, assets_root):
    """Franka USD 를 후보 경로 순서대로 불러와 보고, 성공한 경로를 반환한다. 모두 실패하면 None."""
    candidates = []
    if FRANKA_USD_OVERRIDE:
        candidates.append(FRANKA_USD_OVERRIDE)
    if assets_root:
        candidates += [assets_root.rstrip("/") + rel for rel in FRANKA_USD_RELATIVE_CANDIDATES]
    if not candidates:
        log("No Franka USD candidates (no assets root and no FRANKA_USD_PATH).", "ERROR")
        return None
    for usd_path in candidates:
        log(f"Trying Franka asset: {usd_path}")
        try:
            add_reference_to_stage(usd_path=usd_path, prim_path=ROBOT_PRIM_PATH)
        except Exception as e:  # noqa: BLE001
            log(f"  add_reference_to_stage failed: {e}", "WARN")
            if stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
                stage.RemovePrim(ROBOT_PRIM_PATH)
            continue
        prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if prim.IsValid() and _prim_has_composed_children(prim):
            log(f"  Franka loaded from: {usd_path}")
            return usd_path
        log("  Reference did not resolve (empty prim). Trying next candidate.", "WARN")
        stage.RemovePrim(ROBOT_PRIM_PATH)
    log("Could not load any Franka asset. Set env var FRANKA_USD_PATH to a valid franka.usd path "
        "(Isaac Sim Content Browser > Isaac > Robots > ... > franka.usd, right-click > Copy URL).", "ERROR")
    return None


def setup_lights(stage):
    """조명 = Dome light(전체를 고르게 밝힘) + Distant light(햇빛 같은 주 조명).
    너무 밝으면 색이 날아가 검출이 불안정해지므로 적당한 세기로 둔다."""
    UsdGeom.Xform.Define(stage, "/World/Lights")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(DOME_LIGHT_INTENSITY)
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key.CreateIntensityAttr(KEY_LIGHT_INTENSITY)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))


def create_object(world, spec, xy):
    """OBJECT_SPECS 의 한 항목으로 고정(Fixed) 물체를 만든다.
    Fixed 를 쓰는 이유: 중력에 굴러가지 않고, UI 에서 옮기면 그 자리에 그대로 있다."""
    name = spec["name"]
    position = np.array([xy[0], xy[1], object_center_z(spec)], dtype=float)
    kwargs = {"prim_path": f"/World/Objects/{name}", "name": name.lower(), "position": position}
    material = None
    if PreviewSurface is not None:
        try:
            material = PreviewSurface(prim_path=f"/World/Looks/{name}_Mat",
                                      color=np.array(spec["display_color"], dtype=float),
                                      roughness=float(spec["roughness"]), metallic=float(spec["metallic"]))
        except Exception as e:  # noqa: BLE001
            log(f"PreviewSurface for {name} failed ({e}); using plain color.", "WARN")
            material = None
    if material is not None:
        kwargs["visual_material"] = material
    else:
        kwargs["color"] = np.array(spec["display_color"], dtype=float)

    if spec["shape"] == "cuboid":
        obj = FixedCuboid(size=1.0, scale=np.array(spec["size"], dtype=float), **kwargs)
    elif spec["shape"] == "cylinder":
        obj = FixedCylinder(radius=float(spec["radius"]), height=float(spec["height"]), **kwargs)
    elif spec["shape"] == "sphere":
        obj = FixedSphere(radius=float(spec["radius"]), **kwargs)
    else:
        raise ValueError(f"Unknown shape '{spec['shape']}' for {name}")
    world.scene.add(obj)
    return obj


def _shared_surface(prim_path, color, roughness, metallic=0.0):
    """여러 prim 이 함께 쓰는 재질을 하나 만든다. (바닥 판 수십 개가 재질을 각각 만들지 않도록)"""
    if PreviewSurface is None:
        return None
    try:
        return PreviewSurface(prim_path=prim_path, color=np.array(color, dtype=float),
                              roughness=float(roughness), metallic=float(metallic))
    except Exception as e:  # noqa: BLE001
        log(f"PreviewSurface {prim_path} failed ({e}); using plain color.", "WARN")
        return None


def _visual_box(prim_path, center, size, material, color, cast_shadows=True):
    """충돌·물리가 없는 장식용 박스를 만든다. cast_shadows=False 면 그림자를 만들지 않는다."""
    kwargs = {"prim_path": prim_path, "name": prim_path.split("/")[-1].lower(),
              "position": np.array(center, dtype=float), "size": 1.0, "scale": np.array(size, dtype=float)}
    if material is not None:
        kwargs["visual_material"] = material
    else:
        kwargs["color"] = np.array(color, dtype=float)
    VisualCuboid(**kwargs)
    if not cast_shadows:
        try:
            prim = get_current_stage().GetPrimAtPath(prim_path)
            prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        except Exception:  # noqa: BLE001
            pass
    return prim_path


def _build_warehouse(stage):
    """창고 배경을 만든다 (전부 장식용 박스, 물리 없음).
    반환: MDL 재질을 덮어쓸 prim 경로 목록 {"floor": [...], "walls": [...]}"""
    E = "/World/Environment"
    for sub in ("", "/Floor", "/Markings", "/Walls", "/Columns", "/Roof", "/Racks", "/CameraStand"):
        UsdGeom.Xform.Define(stage, E + sub)
    groups = {"floor": [], "walls": []}
    cx, cy = ROOM_CENTER_XY
    rx, ry = ROOM_SIZE_XY
    H = ROOM_WALL_HEIGHT_M
    x0, x1, y0, y1 = cx - rx / 2, cx + rx / 2, cy - ry / 2, cy + ry / 2

    m_conc = [_shared_surface(f"/World/Looks/Concrete_{k}", c, 0.5) for k, c in enumerate(FLOOR_CONCRETE_COLORS)]
    m_yellow = _shared_surface("/World/Looks/SafetyYellow", SAFETY_YELLOW, 0.6)
    m_block = _shared_surface("/World/Looks/ConcreteBlock", WALL_BLOCK_COLOR, 0.9)
    m_clad = _shared_surface("/World/Looks/MetalCladding", WALL_CLADDING_COLOR, 0.45, 0.5)
    m_dark = _shared_surface("/World/Looks/SteelDark", STEEL_DARK_COLOR, 0.5, 0.6)
    m_galv = _shared_surface("/World/Looks/SteelGalvanized", STEEL_GALV_COLOR, 0.4, 0.7)
    m_card = _shared_surface("/World/Looks/Cardboard", CARDBOARD_COLOR, 0.85)
    m_wood = _shared_surface("/World/Looks/PalletWood", PALLET_WOOD_COLOR, 0.8)

    # (1) 바닥: 2 m 콘크리트 판을 3가지 톤으로 번갈아 깔고, 판 사이에 8 mm 줄눈을 둔다
    p = FLOOR_PANEL_SIZE_M
    for i in range(int(round(rx / p))):
        for j in range(int(round(ry / p))):
            k = (i * 7 + j * 3) % len(FLOOR_CONCRETE_COLORS)
            groups["floor"].append(_visual_box(
                f"{E}/Floor/Slab_{i:02d}_{j:02d}", (x0 + (i + 0.5) * p, y0 + (j + 0.5) * p, 0.002),
                (p - 0.008, p - 0.008, 0.004), m_conc[k], FLOOR_CONCRETE_COLORS[k]))

    # (2) 노란 안전선: 로봇 작업 구역 경계(작업대 + 0.6 m 여유) + 남쪽 보행 통로 2줄
    w, zc, th = 0.10, 0.005, 0.002
    X0 = TABLE_CENTER_XY[0] - TABLE_SIZE_XY[0] / 2 - 0.6
    X1 = TABLE_CENTER_XY[0] + TABLE_SIZE_XY[0] / 2 + 0.6
    Y0 = TABLE_CENTER_XY[1] - TABLE_SIZE_XY[1] / 2 - 0.6
    Y1 = TABLE_CENTER_XY[1] + TABLE_SIZE_XY[1] / 2 + 0.6
    lines = {"Cell_S": ((X0 + X1) / 2, Y0, X1 - X0 + w, w), "Cell_N": ((X0 + X1) / 2, Y1, X1 - X0 + w, w),
             "Cell_W": (X0, (Y0 + Y1) / 2, w, Y1 - Y0), "Cell_E": (X1, (Y0 + Y1) / 2, w, Y1 - Y0),
             "Aisle_1": (cx, y0 + 1.2, rx - 1.0, w), "Aisle_2": (cx, y0 + 2.4, rx - 1.0, w)}
    for name, (lx, ly, sx, sy) in lines.items():
        _visual_box(f"{E}/Markings/{name}", (lx, ly, zc), (sx, sy, th), m_yellow, SAFETY_YELLOW)

    # (3) 벽: 아래 콘크리트 블록(1.2 m) + 위 금속 골강판 + 0.6 m 간격 세로 리브. 벽 안쪽 면이 방 경계
    hb = WALL_BLOCK_HEIGHT_M
    hc = H - hb
    walls = (("North", "x", y1, 1.0), ("South", "x", y0, -1.0), ("East", "y", x1, 1.0), ("West", "y", x0, -1.0))

    def wall_box(path, axis, bnd, out, depth_off, along_c, length, thick, z, h, mat, color, group=True):
        if axis == "x":
            center, size = (along_c, bnd + out * depth_off, z), (length, thick, h)
        else:
            center, size = (bnd + out * depth_off, along_c, z), (thick, length, h)
        path = _visual_box(path, center, size, mat, color)
        if group:
            groups["walls"].append(path)

    for name, axis, bnd, out in walls:
        length = rx + 0.5 if axis == "x" else ry
        along_c = cx if axis == "x" else cy
        wall_box(f"{E}/Walls/{name}_Block", axis, bnd, out, 0.125, along_c, length, 0.25, hb / 2, hb,
                 m_block, WALL_BLOCK_COLOR)
        wall_box(f"{E}/Walls/{name}_Cladding", axis, bnd, out, 0.05, along_c, length, 0.10, hb + hc / 2, hc,
                 m_clad, WALL_CLADDING_COLOR)
        start = (x0 if axis == "x" else y0) + 0.3
        stop = (x1 if axis == "x" else y1) - 0.1
        n = 0
        a = start
        while a < stop:
            wall_box(f"{E}/Walls/{name}_Rib_{n:02d}", axis, bnd, out, -0.02, a, 0.06, 0.04, hb + hc / 2, hc,
                     m_clad, WALL_CLADDING_COLOR, group=False)
            a += 0.6
            n += 1

    # (4) 철골 기둥 + 아래쪽 노란 보호대 (벽 안쪽)
    cols = [(min(max(x0 + 4.0 * k, x0 + 0.25), x1 - 0.25), y1 - 0.25) for k in range(5)]
    cols += [(min(max(x0 + 4.0 * k, x0 + 0.25), x1 - 0.25), y0 + 0.25) for k in range(5)]
    cols += [(x1 - 0.25, y0 + 3.5 * k) for k in range(1, 4)] + [(x0 + 0.25, y0 + 3.5 * k) for k in range(1, 4)]
    for n, (px, py) in enumerate(cols):
        _visual_box(f"{E}/Columns/Column_{n:02d}", (px, py, H / 2), (0.30, 0.30, H), m_dark, STEEL_DARK_COLOR)
        _visual_box(f"{E}/Columns/Guard_{n:02d}", (px, py, 0.5), (0.38, 0.38, 1.0), m_yellow, SAFETY_YELLOW)

    # (5) 지붕 트러스: 지붕 판은 두지 않고(조명을 가리지 않도록), 그림자도 만들지 않음
    for n, (px, _) in enumerate(cols[:5]):
        _visual_box(f"{E}/Roof/Truss_{n}", (px, cy, H - 0.2), (0.25, ry, 0.40), m_dark, STEEL_DARK_COLOR,
                    cast_shadows=False)
    for n, py in enumerate((cy - ry / 4, cy + ry / 4)):
        _visual_box(f"{E}/Roof/Purlin_{n}", (cx, py, H - 0.05), (rx, 0.15, 0.20), m_dark, STEEL_DARK_COLOR,
                    cast_shadows=False)

    # (6) 파렛트 랙 (북쪽 벽): 4칸 x 3단, 아연도금 기둥 + 노란 빔 + 나무 파렛트 + 골판지 상자
    yf, yb = y1 - 1.6, y1 - 0.5
    bay_w, levels, rack_h = 2.7, (0.12, 1.65, 3.20), 4.6
    rack_x0 = cx - 6.4
    for k in range(5):
        for side, yy in (("F", yf), ("B", yb)):
            _visual_box(f"{E}/Racks/Upright_{k}{side}", (rack_x0 + k * bay_w, yy, rack_h / 2), (0.09, 0.09, rack_h),
                        m_galv, STEEL_GALV_COLOR)
    heights = (0.9, 0.7, 1.1, 0.8, 1.0, 0.6)
    n = 0
    for b in range(4):
        bx = rack_x0 + b * bay_w
        for li, lz in enumerate(levels):
            for side, yy in (("F", yf), ("B", yb)):
                _visual_box(f"{E}/Racks/Beam_{b}_{li}{side}", (bx + bay_w / 2, yy, lz), (bay_w - 0.09, 0.06, 0.12),
                            m_yellow, SAFETY_YELLOW)
            for s, sx in enumerate((bx + 0.72, bx + 1.98)):
                if (b + li + s) % 5 == 3:          # 일부 칸은 비워 둠
                    continue
                pz = lz + 0.06 + 0.07
                _visual_box(f"{E}/Racks/Pallet_{n:02d}", (sx, (yf + yb) / 2, pz), (1.1, 0.95, 0.14), m_wood,
                            PALLET_WOOD_COLOR)
                bh = heights[n % len(heights)]
                _visual_box(f"{E}/Racks/Box_{n:02d}", (sx, (yf + yb) / 2, pz + 0.07 + bh / 2), (1.0, 0.85, bh),
                            m_card, CARDBOARD_COLOR)
                n += 1

    # (7) 바닥에 놓인 파렛트 + 상자 (동쪽 벽 근처)
    for n, py in enumerate((cy - 3.0, cy + 3.0)):
        px = x1 - 1.4
        _visual_box(f"{E}/Racks/FloorPallet_{n}", (px, py, 0.07), (1.2, 1.0, 0.14), m_wood, PALLET_WOOD_COLOR)
        _visual_box(f"{E}/Racks/FloorBox_{n}", (px, py, 0.14 + 0.45), (1.1, 0.9, 0.9), m_card, CARDBOARD_COLOR)

    # (8) 카메라 스탠드: 카메라 바로 아래의 기둥 (카메라는 아래로 최대 약 50° 까지만 보므로 영상에 찍히지 않음)
    cam_x, cam_y, cam_z = CAMERA_POSITION
    _visual_box(f"{E}/CameraStand/BasePlate", (cam_x, cam_y, 0.01), (0.45, 0.45, 0.02), m_dark, STEEL_DARK_COLOR)
    pole_h = cam_z - 0.12 - 0.02
    _visual_box(f"{E}/CameraStand/Pole", (cam_x, cam_y, 0.02 + pole_h / 2), (0.06, 0.06, pole_h), m_dark,
                STEEL_DARK_COLOR)
    return groups


def setup_environment(world, stage):
    """바닥(충돌체 있음)과 창고 배경을 만든다.
    배경은 장식일 뿐 판정과 무관하며, 만들다 실패해도 바닥만으로 계속 진행한다.
    반환: MDL 재질을 덮어쓸 prim 경로 목록 {"floor": [...], "walls": [...]}"""
    # (1) 넓은 바닥: 별도 다운로드 없이 만드는 GroundPlane (물리 바닥 역할)
    try:
        if GroundPlane is None:
            raise ImportError("GroundPlane class not available")
        world.scene.add(GroundPlane(prim_path="/World/GroundPlane", name="ground_plane", size=FLOOR_SIZE_M,
                                    z_position=0.0, color=np.array(FLOOR_JOINT_COLOR)))
    except Exception as e:  # noqa: BLE001
        log(f"GroundPlane failed ({e}); trying world.scene.add_default_ground_plane()", "WARN")
        try:
            world.scene.add_default_ground_plane()
        except Exception as e2:  # noqa: BLE001
            log(f"Default ground plane also failed: {e2}", "ERROR")
    if ENVIRONMENT_STYLE != "warehouse":
        return {}
    try:
        groups = _build_warehouse(stage)
        log(f"Environment: warehouse cell {ROOM_SIZE_XY[0]:.0f} x {ROOM_SIZE_XY[1]:.0f} m (concrete floor, "
            f"block + metal walls, columns, racks), floor {FLOOR_SIZE_M:.0f} x {FLOOR_SIZE_M:.0f} m.")
        return groups
    except Exception as e:  # noqa: BLE001
        log(f"Environment decoration failed ({e}); continuing with the plain floor.", "WARN")
        traceback.print_exc()
        return {}


def setup_workbench(world, stage):
    """철제 작업대를 만든다: 상판(충돌체, 윗면 높이 = TABLE_TOP_Z) + 다리·프레임 + 노란 테두리.
    판정에 쓰이는 것은 TABLE_TOP_Z / TABLE_CENTER_XY / TABLE_SIZE_XY 상수뿐이고, 외형은 결과에 영향이 없다.
    반환: 상판 prim 경로 (MDL 재질 덮어쓰기용)"""
    W = "/World/Workbench"
    UsdGeom.Xform.Define(stage, W)
    tx, ty = TABLE_CENTER_XY
    sx, sy = TABLE_SIZE_XY
    zt, pt = TABLE_TOP_Z, WORKBENCH_TOP_THICKNESS_M
    top_mat = _shared_surface("/World/Looks/WorkbenchTop", TABLE_COLOR, 0.55, 0.3)   # 도장된 강판
    kwargs = {"prim_path": f"{W}/TopPlate", "name": "table", "size": 1.0,
              "position": np.array([tx, ty, zt - pt / 2]), "scale": np.array([sx, sy, pt])}
    if top_mat is not None:
        kwargs["visual_material"] = top_mat
    else:
        kwargs["color"] = np.array(TABLE_COLOR)
    world.scene.add(FixedCuboid(**kwargs))
    try:
        m_galv = _shared_surface("/World/Looks/WorkbenchFrame", STEEL_GALV_COLOR, 0.4, 0.7)
        m_yellow = _shared_surface("/World/Looks/WorkbenchEdge", SAFETY_YELLOW, 0.6)
        leg_h = zt - pt
        xs = (tx - sx / 2 + 0.12, tx, tx + sx / 2 - 0.12)
        ys = (ty - sy / 2 + 0.12, ty, ty + sy / 2 - 0.12)
        for i, lx in enumerate(xs):
            for j, ly in enumerate(ys):
                _visual_box(f"{W}/Leg_{i}{j}", (lx, ly, leg_h / 2), (0.10, 0.10, leg_h), m_galv, STEEL_GALV_COLOR)
        for j, ly in enumerate(ys):
            _visual_box(f"{W}/RailX_{j}", (tx, ly, 0.12), (sx - 0.24, 0.06, 0.06), m_galv, STEEL_GALV_COLOR)
        for i, lx in enumerate(xs):
            _visual_box(f"{W}/RailY_{i}", (lx, ty, 0.12), (0.06, sy - 0.24, 0.06), m_galv, STEEL_GALV_COLOR)
        # 상판 옆면의 노란 테두리 (윗면보다 1 cm 낮게 끝나므로 작업 표면 높이에는 영향 없음)
        ez = zt - pt / 2
        edges = {"S": ((tx, ty - sy / 2 - 0.003, ez), (sx, 0.006, 0.04)),
                 "N": ((tx, ty + sy / 2 + 0.003, ez), (sx, 0.006, 0.04)),
                 "W": ((tx - sx / 2 - 0.003, ty, ez), (0.006, sy, 0.04)),
                 "E": ((tx + sx / 2 + 0.003, ty, ez), (0.006, sy, 0.04))}
        for name, (c, s) in edges.items():
            _visual_box(f"{W}/EdgeBand_{name}", c, s, m_yellow, SAFETY_YELLOW)
    except Exception as e:  # noqa: BLE001
        log(f"Workbench frame decoration failed ({e}); top plate only.", "WARN")
    return f"{W}/TopPlate"


def apply_mdl_overrides(groups):
    """MDL_OVERRIDES 에 URL 을 넣은 그룹에만 실제 텍스처 재질(MDL)을 입힌다 (기본: 사용 안 함).
    실패하면 기본 재질을 그대로 유지한다."""
    for key, url in MDL_OVERRIDES.items():
        if not url or not groups.get(key):
            continue
        try:
            import omni.kit.commands
            mtl_path = f"/World/Looks/MDL_{key}"
            omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=url,
                                      mtl_name=os.path.splitext(os.path.basename(url))[0], mtl_path=mtl_path)
            for prim_path in groups[key]:
                omni.kit.commands.execute("BindMaterial", prim_path=prim_path, material_path=mtl_path)
            log(f"MDL material applied to '{key}' ({len(groups[key])} prims): {url}")
        except Exception as e:  # noqa: BLE001
            log(f"MDL override for '{key}' failed ({e}); keeping default look.", "WARN")


def setup_scene(world):
    """바닥 · 배경 · 작업대 · 조명 · Franka · 물체 3개를 만들고, 이후에 쓸 핸들을 dict 로 반환한다."""
    stage = get_current_stage()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    UsdGeom.Xform.Define(stage, "/World/Looks")
    scene = {"objects": {}, "articulation": None, "robot_xform": None,
             "robot_usd": None, "robot_is_placeholder": False}

    # (1) 바닥 + 창고 배경
    groups = setup_environment(world, stage)

    # (2) 작업대 (윗면 높이 = TABLE_TOP_Z)
    groups["workbench_top"] = [setup_workbench(world, stage)]
    apply_mdl_overrides(groups)

    # (3) 조명
    setup_lights(stage)

    # (4) 로봇 (Franka). 불러오기에 실패하면 회색 원기둥으로 대신하고 계속 진행
    #     (좌표 변환과 거리 계산은 그대로 확인할 수 있음)
    assets_root = get_assets_root_safe()
    usd = load_franka_reference(stage, assets_root)
    if usd is None:
        scene["robot_is_placeholder"] = True
        UsdGeom.Xform.Define(stage, ROBOT_PRIM_PATH)
        VisualCylinder(prim_path=ROBOT_PRIM_PATH + "/PlaceholderBody", name="robot_placeholder",
                       translation=np.array([0.0, 0.0, 0.15]), radius=0.08, height=0.30,
                       color=np.array([0.3, 0.3, 0.3]))
        log("Using a PLACEHOLDER robot base (gray cylinder). Distances are still computed in its base frame.",
            "ERROR")
    scene["robot_usd"] = usd
    # 로봇 base 좌표계 = ROBOT_PRIM_PATH prim 의 좌표계 (Franka 의 panda_link0 가 이 원점에 있음)
    scene["robot_xform"] = SingleXFormPrim(prim_path=ROBOT_PRIM_PATH, name="robot_base_frame",
                                           position=np.array(ROBOT_INITIAL_POSITION, dtype=float),
                                           orientation=yaw_to_quat_wxyz(ROBOT_INITIAL_YAW_DEG))
    if usd is not None and SingleArticulation is not None:
        try:
            art = SingleArticulation(prim_path=ROBOT_PRIM_PATH, name="franka")
            world.scene.add(art)
            scene["articulation"] = art
        except Exception as e:  # noqa: BLE001
            log(f"SingleArticulation wrap failed ({e}); robot will stay in its default pose.", "WARN")

    # (5) 물체
    for spec in OBJECT_SPECS:
        scene["objects"][spec["name"]] = create_object(world, spec, OBJECT_INITIAL_POSITIONS[spec["name"]])
    return scene


def apply_robot_home(scene):
    """world.reset() 후 Franka 를 대기 자세로 둔다 (화면 구성용, 판정과 무관)."""
    art = scene.get("articulation")
    if art is None or not ROBOT_HOME_JOINTS or ArticulationAction is None:
        return
    try:
        q = np.array(ROBOT_HOME_JOINTS, dtype=float)
        if art.num_dof != len(q):
            log(f"Robot DOF={art.num_dof} != len(ROBOT_HOME_JOINTS)={len(q)}; skip home pose.", "WARN")
            return
        art.set_joint_positions(q)
        art.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
    except Exception as e:  # noqa: BLE001
        log(f"Could not set robot home pose: {e}", "WARN")


def apply_scenario(scene, name):
    """물체들을 시나리오 프리셋 위치로 한 번 옮긴다. (사용자가 시나리오를 바꿀 때만 호출, 매 frame 호출 X)"""
    preset = SCENARIO_PRESETS[name]
    for spec in OBJECT_SPECS:
        obj = scene["objects"][spec["name"]]
        xy = preset[spec["name"]]
        pos = np.array([xy[0], xy[1], object_center_z(spec)], dtype=float)
        obj.set_world_pose(position=pos)
        try:
            obj.set_default_state(position=pos)     # Stop → Play 로 reset 해도 이 위치가 유지되도록
        except Exception:  # noqa: BLE001
            pass
    log(f"Scenario {name} applied once. Objects can now be moved freely in the UI.")


def validate_scene_layout(camera, scene):
    """시작할 때 한 번 배치를 점검하고, 문제가 있으면 경고만 출력한다 (판정과 무관).
    - 모든 시나리오(A/B/C)의 물체가 작업대 위에 완전히 올라가 있는가?
    - 각 물체 중심과 Franka(base, 몸체 윗부분 근사점)가 카메라 화면 안에 보이는가?"""
    K, _ = get_intrinsics(camera)
    t_wc, R_wc, _ = get_camera_pose_ros(camera)
    t_wr, _ = get_robot_base_pose(scene)
    w, h = CAMERA_RESOLUTION
    tx0, tx1 = TABLE_CENTER_XY[0] - TABLE_SIZE_XY[0] / 2, TABLE_CENTER_XY[0] + TABLE_SIZE_XY[0] / 2
    ty0, ty1 = TABLE_CENTER_XY[1] - TABLE_SIZE_XY[1] / 2, TABLE_CENTER_XY[1] + TABLE_SIZE_XY[1] / 2

    # world 점 p 를 카메라 영상에 투영했을 때 화면 안에 들어오는지
    def in_view(p):
        pc = R_wc.T @ (np.asarray(p, dtype=float) - t_wc)
        if pc[2] <= 0:
            return False
        uvw = K @ (pc / pc[2])
        return 0 <= uvw[0] < w and 0 <= uvw[1] < h

    problems = 0
    for sname, preset in SCENARIO_PRESETS.items():
        for spec in OBJECT_SPECS:
            x, y = preset[spec["name"]]
            half = spec["size"][0] / 2 if spec["shape"] == "cuboid" else spec["radius"]
            if not (tx0 + half <= x <= tx1 - half and ty0 + half <= y <= ty1 - half):
                log(f"Layout: scenario {sname} {spec['name']} at ({x:.2f},{y:.2f}) is NOT fully on the table", "WARN")
                problems += 1
            if not in_view((x, y, object_center_z(spec))):
                log(f"Layout: scenario {sname} {spec['name']} is outside the camera image", "WARN")
                problems += 1
    for label, offset in (("robot base", (0.0, 0.0, 0.0)), ("robot upper body (approx.)", (-0.10, 0.0, 0.75))):
        if not in_view(t_wr + np.array(offset)):
            log(f"Layout: {label} is outside the camera image", "WARN")
            problems += 1
    log("Layout check: all scenario objects on table and in camera view; Franka in view." if problems == 0
        else f"Layout check found {problems} issue(s) (see warnings above).")


# =============================================================================
# 5. 카메라
# =============================================================================
def setup_camera():
    """로봇 밖에 고정된 RGB-D 카메라를 만든다 (eye-to-hand).
    - 위치/방향은 여기서 한 번만 정하고, 이후에는 바꾸지 않는다.
    - 세로 센서 크기 = 가로 센서 크기 × (세로 해상도 / 가로 해상도) 로 맞춰서 fx == fy 가 되게 한다."""
    w, h = CAMERA_RESOLUTION
    kwargs = {"prim_path": CAMERA_PRIM_PATH, "name": "rgbd_camera",
              "position": np.array(CAMERA_POSITION, dtype=float), "resolution": (w, h)}
    try:
        camera = Camera(frequency=CAMERA_FREQUENCY, **kwargs)
    except TypeError as e:
        log(f"Camera(frequency=...) not accepted ({e}); creating camera without frequency.", "WARN")
        camera = Camera(**kwargs)

    prim = get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
    usd_cam = UsdGeom.Camera(prim)
    usd_cam.GetFocalLengthAttr().Set(float(CAMERA_FOCAL_LENGTH_MM))
    usd_cam.GetHorizontalApertureAttr().Set(float(CAMERA_HORIZONTAL_APERTURE_MM))
    usd_cam.GetVerticalApertureAttr().Set(float(CAMERA_HORIZONTAL_APERTURE_MM * h / w))
    usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(*CAMERA_CLIPPING_RANGE))

    # 바라보는 방향을 USD 카메라 축 기준으로 계산했으므로 camera_axes="usd" 로 그대로 넣는다
    q_usd = look_at_quat_usd(CAMERA_POSITION, CAMERA_TARGET)
    camera.set_world_pose(position=np.array(CAMERA_POSITION, dtype=float), orientation=q_usd, camera_axes="usd")
    return camera


def get_intrinsics(camera):
    """카메라 내부 파라미터 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (단위 px) 를 구한다.
    공식 API 를 먼저 쓰고, 실패하면 USD 속성값(초점거리, 센서 크기)으로 직접 계산한다."""
    try:
        K = np.asarray(camera.get_intrinsics_matrix(), dtype=float)
        if K.shape == (3, 3) and K[0, 0] > 0 and K[1, 1] > 0:
            return K, "Camera.get_intrinsics_matrix()"
    except Exception as e:  # noqa: BLE001
        RL.warn("intr", f"get_intrinsics_matrix() failed: {e}; computing from USD attributes")
    w, h = CAMERA_RESOLUTION
    usd_cam = UsdGeom.Camera(get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH))
    f = usd_cam.GetFocalLengthAttr().Get()
    ha = usd_cam.GetHorizontalApertureAttr().Get()
    va = usd_cam.GetVerticalApertureAttr().Get()
    K = np.array([[w * f / ha, 0.0, w / 2.0], [0.0, h * f / va, h / 2.0], [0.0, 0.0, 1.0]])
    return K, "USD focalLength/aperture (fallback)"


def get_camera_pose_ros(camera):
    """카메라 위치/방향을 ROS optical 축 기준으로 반환한다.   p_world = R_wc @ p_cam + t_wc

    ROS optical 축: x = 화면 오른쪽, y = 화면 아래, z = 카메라가 보는 방향 (= depth 값의 방향)
    반환: (t_wc, R_wc, 사용한 방법 설명)"""
    try:
        pos, quat = camera.get_world_pose(camera_axes="ros")
        return np.asarray(pos, dtype=float).reshape(3), quat_wxyz_to_rotmat(quat), \
            "Camera.get_world_pose(camera_axes='ros')"
    except Exception as e:  # noqa: BLE001
        RL.warn("campose", f"Camera.get_world_pose(camera_axes='ros') failed: {e}; using USD transform")
    prim = get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=float)
    # USD(Gf) 행렬은 행벡터 규약 (p_world = p_local @ M) → 회전 R = M[:3,:3] 의 전치, 위치 t = M[3,:3]
    R_usd = M[:3, :3].T
    R_usd = R_usd / np.linalg.norm(R_usd, axis=0, keepdims=True)
    R_ros = R_usd @ np.diag([1.0, -1.0, -1.0])     # USD 축(y 위, z 뒤) → ROS 축(y 아래, z 앞): y, z 부호만 반전
    return M[3, :3].copy(), R_ros, "USD ComputeLocalToWorldTransform (fallback)"


def get_robot_base_pose(scene):
    """현재 로봇 base 의 위치/방향을 stage 에서 읽는다.
    반환 (t_wr, R_wr): '로봇 좌표 → world 좌표' 변환.   p_world = R_wr @ p_robot + t_wr"""
    pos, quat = scene["robot_xform"].get_world_pose()
    return np.asarray(pos, dtype=float).reshape(3), quat_wxyz_to_rotmat(quat)


# ─────────────────────────────────────────────────────────────────────────────
# ★ [CORE-1] 같은 frame 의 RGB 와 Depth 를 함께 가져오기 (센서 동기화)
# ─────────────────────────────────────────────────────────────────────────────
def acquire_rgbd_frame(camera):
    """같은 카메라에서 같은 순간에 찍힌 RGB 와 Depth 를 함께 가져온다.

    가져오는 방법 (위에서부터 시도):
      1) camera.get_current_frame() : RGB 와 Depth 가 같은 렌더링 시점으로 한 dict 에 담겨 온다.
      2) get_rgba() / get_depth()   : 따로 가져오지만, 같은 world.step() 직후라 같은 렌더링 결과다.

    Depth 의 의미 (중요):
      distance_to_image_plane = 카메라가 보는 방향(광축)으로 잰 거리 Z [m]
      ※ 렌즈 중심까지의 직선거리(distance_to_camera)가 아니다.

    반환: {"rgb", "depth", "rendering_frame", "rendering_time", "source"}
          카메라 버퍼가 아직 비어 있으면 None"""
    rgba = depth = None
    rframe = rtime = None
    source = "camera.get_current_frame()"
    # 1) 한 번에 가져오기
    try:
        frame = camera.get_current_frame()
    except Exception as e:  # noqa: BLE001
        frame = None
        RL.warn("getframe", f"get_current_frame() failed: {e}")
    if isinstance(frame, dict):
        rgba = frame.get("rgba")
        depth = frame.get("distance_to_image_plane")
        rframe = frame.get("rendering_frame")
        rtime = frame.get("rendering_time")
    # 2) 빠진 것이 있으면 개별로 가져오기
    if rgba is None or np.asarray(rgba).size == 0:
        try:
            rgba = camera.get_rgba()
            source = "separate getters (same step)"
        except Exception:  # noqa: BLE001
            rgba = None
    if depth is None or np.asarray(depth).size == 0:
        try:
            depth = camera.get_depth()
            source = "separate getters (same step)"
        except Exception:  # noqa: BLE001
            depth = None
    if rgba is None or depth is None:
        return None
    rgba = np.asarray(rgba)
    depth = np.asarray(depth, dtype=np.float32)
    if rgba.ndim != 3 or rgba.shape[2] < 3 or rgba.size == 0 or depth.size == 0:
        return None
    if depth.ndim == 3:                         # (H, W, 1) → (H, W)
        depth = depth[..., 0]
    if depth.shape[:2] != rgba.shape[:2]:
        RL.warn("shape", f"RGB {rgba.shape} and depth {depth.shape} sizes differ; skipping frame")
        return None
    rgb = rgba[..., :3]                         # RGBA → RGB (알파 채널 제거)
    if rgb.dtype != np.uint8:                   # 0~1 float 이면 0~255 정수로 변환
        rgb = np.clip(rgb * (255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return {"rgb": np.array(rgb, copy=True), "depth": np.array(depth, copy=True),
            "rendering_frame": rframe, "rendering_time": rtime, "source": source}


# =============================================================================
# 6. RGB 처리: 색으로 물체 찾기 (WHAT — 무엇인가?)
# =============================================================================
def rgb_to_hsv_cv_scale(rgb):
    """RGB 이미지 → HSV 이미지 (OpenCV 범위: H 0~179, S·V 0~255).
    OpenCV 가 없으면 numpy 로 같은 계산을 한다."""
    if cv2 is not None:
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    maxc, minc = x.max(axis=-1), x.min(axis=-1)
    delta = maxc - minc
    safe = np.where(delta > 1e-6, delta, 1.0)
    h = np.where(maxc == r, ((g - b) / safe) % 6.0,
                 np.where(maxc == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    h = np.where(delta > 1e-6, h, 0.0) * 30.0
    s = np.where(maxc > 1e-6, delta / np.where(maxc > 1e-6, maxc, 1.0), 0.0) * 255.0
    v = maxc * 255.0
    return np.stack([np.clip(h, 0, 179), s, v], axis=-1).astype(np.uint8)


def _largest_component(mask):
    """작은 노이즈를 지운 뒤 가장 큰 덩어리 하나만 남긴다. (OpenCV 가 없으면 mask 를 그대로 반환)"""
    if cv2 is None or not mask.any():
        return mask
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


# ─────────────────────────────────────────────────────────────────────────────
# ★ [CORE-2] RGB 의 역할 (WHAT): 색으로 물체를 구분하고, Target 여부와 mask 를 정한다
# ─────────────────────────────────────────────────────────────────────────────
def detect_objects_rgb(rgb, object_specs):
    """RGB 영상에서 각 물체를 색(HSV 범위)으로 찾는다.

    반환: {물체 이름: 검출 결과}.  검출 결과의 주요 항목:
      detected    찾았는가 (True / False)
      role        "TARGET" / "NON_TARGET"
      pixel_count 물체로 판정된 pixel 수
      mask        물체 pixel 위치 (True/False 2D 배열)
      bbox        물체를 감싸는 사각형 (x0, y0, x1, y1)
      centroid    물체 중심 pixel (u, v)
      det_reason  못 찾았을 때 그 이유

    pixel 좌표: 원점은 왼쪽 위, u 는 오른쪽(+), v 는 아래쪽(+).
    ※ 나중에 YOLO 같은 검출기로 바꾸려면, 반환 형식만 맞춰서 이 함수만 교체하면 된다."""
    hsv = rgb_to_hsv_cv_scale(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = {}
    for spec in object_specs:
        name = spec["name"]
        # 이 물체 색의 HSV 범위에 들어가는 pixel 을 모두 True 로 표시
        mask = np.zeros(H.shape, dtype=bool)
        for lo, hi in COLOR_RANGES[spec["color_key"]]:
            mask |= ((H >= lo[0]) & (H <= hi[0]) & (S >= lo[1]) & (S <= hi[1]) & (V >= lo[2]) & (V <= hi[2]))
        mask = _largest_component(mask)
        count = int(mask.sum())
        det = {"name": name, "role": object_role(name), "visual": spec["visual"], "detected": False,
               "pixel_count": count, "mask": None, "bbox": None, "centroid": None, "det_reason": ""}
        if count == 0:
            det["det_reason"] = "no matching pixels (out of FOV or fully occluded?)"
        elif count < MIN_MASK_PIXELS:
            det["det_reason"] = f"mask too small ({count} < {MIN_MASK_PIXELS} px)"
        else:
            ys, xs = np.nonzero(mask)
            # centroid 에 +0.5: pixel 의 왼쪽 위 모서리가 아니라 pixel 중심을 기준으로 함
            det.update(detected=True, mask=mask, det_reason="ok",
                       bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                       centroid=(float(xs.mean()) + 0.5, float(ys.mean()) + 0.5))
        out[name] = det
    return out


# =============================================================================
# 7. Depth 처리: RGB 가 고른 pixel 의 depth 만 사용 (WHERE — 어디에 있나?)
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# ★ [CORE-3] 센서 융합 지점: RGB mask 로 Depth 영상에서 쓸 pixel 을 고른다
#            (화면 전체에서 가장 가까운 depth 를 쓰는 것이 아님)
# ─────────────────────────────────────────────────────────────────────────────
def extract_object_depth(depth, mask):
    """물체 mask 안쪽 pixel 들의 depth 중앙값(median)을 구한다.

    - 사각형(bbox) 전체가 아니라 mask 안쪽만 쓰므로 배경 depth 가 섞이지 않는다.
    - 무효값(NaN, Inf, 0, DEPTH_MAX_VALID_M 보다 먼 값)은 뺀다.
    - 평균 대신 중앙값을 써서 튀는 값의 영향을 줄인다.

    반환: {depth_valid, depth_median [m], depth_valid_count, depth_reason}"""
    res = {"depth_valid": False, "depth_median": None, "depth_valid_count": 0, "depth_reason": ""}
    if mask is None:
        res["depth_reason"] = "no RGB mask"
        return res
    # mask 가장자리를 1px 깎아 경계의 배경 depth 를 뺀다 (깎은 뒤 너무 작아지면 원래 mask 사용)
    use_mask = mask
    if DEPTH_ERODE_MASK and cv2 is not None:
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        if int(eroded.sum()) >= MIN_VALID_DEPTH_PIXELS:
            use_mask = eroded
    vals = depth[use_mask]
    valid = np.isfinite(vals) & (vals > 0.0) & (vals < DEPTH_MAX_VALID_M)   # NaN / Inf / 0 / 너무 먼 값 제외
    vv = vals[valid]
    res["depth_valid_count"] = int(vv.size)
    if vv.size < MIN_VALID_DEPTH_PIXELS:
        res["depth_reason"] = f"INVALID DEPTH ({vv.size} valid px < {MIN_VALID_DEPTH_PIXELS})"
        return res
    res.update(depth_valid=True, depth_median=float(np.median(vv)), depth_reason="ok")
    return res


# =============================================================================
# 8. pixel (u, v) + depth Z → 3D 위치 (카메라 좌표 / world 좌표)
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# ★ [CORE-4] Depth 의 역할 (WHERE): pixel (u, v) + depth Z → 3D 위치
# ─────────────────────────────────────────────────────────────────────────────
def pixel_depth_to_3d(camera, u, v, z, K, cam_pose):
    """pixel (u, v) 와 depth Z 로 3D 위치를 계산한다.

    1) 카메라 좌표 (ROS optical: x 오른쪽, y 아래, z 앞)
           X = (u - cx) * Z / fx
           Y = (v - cy) * Z / fy
           Z = depth
       ※ depth 가 광축 방향 거리(distance_to_image_plane)라서 이 식이 그대로 성립한다.
         (렌즈 중심까지의 직선거리였다면 이 식은 틀림)

    2) world 좌표
       Isaac Sim 공식 함수 get_world_points_from_image_coords() 의 결과를 쓰고,
       직접 계산한 값(R_wc @ p_cam + t_wc)과 비교해 좌표 규약이 맞는지 매번 확인한다.

    반환: {p_cam, p_world, unproj_method ("official" / "manual"), unproj_check_m (두 계산의 차이 [m])}"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)
    t_wc, R_wc, _ = cam_pose
    p_world_manual = R_wc @ p_cam + t_wc
    p_world_official = None
    if USE_OFFICIAL_UNPROJECTION:
        try:
            pts = camera.get_world_points_from_image_coords(np.array([[u, v]], dtype=float),
                                                            np.array([z], dtype=float))
            p_world_official = np.asarray(pts, dtype=float).reshape(-1, 3)[0]
        except Exception as e:  # noqa: BLE001
            RL.warn("unproj", f"get_world_points_from_image_coords failed: {e}; using manual pinhole")
    if p_world_official is not None:
        check = float(np.linalg.norm(p_world_official - p_world_manual))
        if check > UNPROJECTION_CHECK_TOL_M:
            RL.warn("unproj_mismatch", f"Official vs manual unprojection differ by {check:.3f} m. "
                                       f"official={fmt_vec(p_world_official)} manual={fmt_vec(p_world_manual)}. "
                                       "Check camera axes / depth type.")
        return {"p_cam": p_cam, "p_world": p_world_official, "unproj_method": "official",
                "unproj_check_m": check}
    return {"p_cam": p_cam, "p_world": p_world_manual, "unproj_method": "manual", "unproj_check_m": None}


# =============================================================================
# 9. world 좌표 → 로봇 base 좌표
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# ★ [CORE-5] 좌표 변환: world → 로봇 base
#            (카메라 depth 는 '카메라~물체' 거리이므로, 로봇까지의 거리로 그대로 쓰면 안 된다)
# ─────────────────────────────────────────────────────────────────────────────
def world_to_robot_base(p_world, t_wr, R_wr):
    """world 좌표의 점을 로봇 base 좌표로 바꾼다.

    (t_wr, R_wr) 은 '로봇 → world' 변환이다:    p_world = R_wr @ p_robot + t_wr
    여기서는 반대 방향이 필요하므로 역변환을 쓴다:  p_robot = R_wr^T @ (p_world - t_wr)"""
    return R_wr.T @ (np.asarray(p_world, dtype=float) - t_wr)


# =============================================================================
# 10. 작업 반경 판정 (최종 결과는 Target 하나로 결정)
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# ★ [CORE-6] 최종 판정: 로봇 base 에서 Target 까지의 3D 거리 ≤ MAX_REACH_M ?
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_reachability(results, max_reach):
    """모든 물체에 IN / OUT / N/A 를 붙이고, 최종 결과(FINAL)는 Target 만 보고 정한다.

    FINAL status 종류:
      REACHABLE RANGE       Target 거리 ≤ max_reach
      OUT OF RANGE          Target 거리 > max_reach
      TARGET NOT DETECTED   RGB 에서 Target 을 찾지 못함
      TARGET DEPTH INVALID  Target 은 찾았지만 유효한 depth 가 부족함
      TARGET NOT DEFINED    TARGET_NAME 이 OBJECT_SPECS 에 없음"""
    for r in results:
        d = r.get("robot_dist")
        r["range_status"] = "N/A" if d is None else ("IN" if d <= max_reach else "OUT")
    target = next((r for r in results if r["role"] == "TARGET"), None)
    final = {"target_name": TARGET_NAME, "target_dist": None, "max_reach": max_reach}
    if target is None:
        final.update(status="TARGET NOT DEFINED", reason=f"'{TARGET_NAME}' not in OBJECT_SPECS")
    elif not target["detected"]:
        final.update(status="TARGET NOT DETECTED", reason=target["det_reason"])
    elif not target["depth_valid"]:
        final.update(status="TARGET DEPTH INVALID", reason=target["depth_reason"])
    else:
        final["target_dist"] = target["robot_dist"]
        final.update(status="REACHABLE RANGE" if target["robot_dist"] <= max_reach else "OUT OF RANGE",
                     reason="fusion of RGB identity + mask-selected depth")
    return final


# =============================================================================
# 11. 비교용: 센서를 하나만 쓰면? (RGB only / Depth only)
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# ★ [CORE-8] 비교용 Depth only: 거리는 알지만, 그게 어떤 물체인지는 모른다
# ─────────────────────────────────────────────────────────────────────────────
def depth_only_baseline(depth, K, cam_pose, robot_pose, max_reach):
    """Depth 만 사용해, 작업대 위로 솟은 표면 중 카메라에 가장 가까운 점을 찾는다.

    - RGB 는 전혀 쓰지 않는다. (작업대 높이·범위, 로봇 위치 같은 scene 정보만 사용)
    - 거리는 알 수 있지만, 그 점이 어떤 물체인지(Target 인지)는 알 수 없다.
    - naive_status: "가장 가까운 물체 = Target" 이라고 가정했을 때의 판정 결과"""
    t_wc, R_wc, _ = cam_pose
    t_wr, R_wr = robot_pose
    h, w = depth.shape
    vs, us = np.mgrid[0:h:DEPTH_ONLY_STRIDE, 0:w:DEPTH_ONLY_STRIDE]
    z = depth[vs, us]
    valid = np.isfinite(z) & (z > 0) & (z < DEPTH_MAX_VALID_M)
    out = {"available": False, "nearest_depth": None, "pixel": None, "robot_dist": None, "naive_status": "N/A"}
    if not valid.any():
        return out
    u = us[valid].astype(float) + 0.5
    v = vs[valid].astype(float) + 0.5
    z = z[valid].astype(float)
    # 유효한 모든 pixel 을 3D world 점으로 변환
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    pw = pc @ R_wc.T + t_wc
    tx0, tx1 = TABLE_CENTER_XY[0] - TABLE_SIZE_XY[0] / 2, TABLE_CENTER_XY[0] + TABLE_SIZE_XY[0] / 2
    ty0, ty1 = TABLE_CENTER_XY[1] - TABLE_SIZE_XY[1] / 2, TABLE_CENTER_XY[1] + TABLE_SIZE_XY[1] / 2
    # 물체 후보: 작업대 위 일정 높이 범위 + 작업대 영역 안 + 로봇 몸체 근처는 제외
    m = ((pw[:, 2] > TABLE_TOP_Z + DEPTH_ONLY_HEIGHT_BAND[0]) & (pw[:, 2] < TABLE_TOP_Z + DEPTH_ONLY_HEIGHT_BAND[1])
         & (pw[:, 0] > tx0) & (pw[:, 0] < tx1) & (pw[:, 1] > ty0) & (pw[:, 1] < ty1)
         & (np.hypot(pw[:, 0] - t_wr[0], pw[:, 1] - t_wr[1]) > DEPTH_ONLY_ROBOT_EXCLUSION_RADIUS))
    if not m.any():
        return out
    idx = np.flatnonzero(m)[int(np.argmin(z[m]))]          # 후보 중 카메라에 가장 가까운 점
    p_r = world_to_robot_base(pw[idx], t_wr, R_wr)
    dist = float(np.linalg.norm(p_r))
    out.update(available=True, nearest_depth=float(z[idx]), pixel=(float(u[idx]), float(v[idx])),
               robot_dist=dist, naive_status="REACHABLE RANGE" if dist <= max_reach else "OUT OF RANGE")
    return out


# 센서 하나만 썼을 때의 결과를 만들어 융합 결과와 비교한다.
#   - RGB only  : Target 이 화면 어디에 있는지는 알지만 거리를 모름 → 판정 불가
#   - Depth only: 가장 가까운 표면까지 거리는 알지만 그게 무엇인지 모름
def compute_single_sensor_baselines(results, depth, K, cam_pose, robot_pose, max_reach, final):
    target = next((r for r in results if r["role"] == "TARGET"), None)
    rgb_only = {"target_detected": bool(target and target["detected"]),
                "centroid": target["centroid"] if target else None,
                "distance": None, "reachability": "UNKNOWN"}
    d_only = depth_only_baseline(depth, K, cam_pose, robot_pose, max_reach)
    # 설명용: Depth only 가 고른 점이 실제로 어떤 물체인지 RGB mask 로 확인 (Depth 만으로는 알 수 없음)
    d_only["actually_is"] = "unknown surface"
    if d_only["available"]:
        pu, pv = int(d_only["pixel"][0]), int(d_only["pixel"][1])
        for r in results:
            if r["mask"] is not None and r["mask"][pv, pu]:
                d_only["actually_is"] = r["name"]
                break
    # Depth only 판정과 융합 판정이 같은지 (Target 판정이 가능한 경우에만 비교)
    agree = None
    if d_only["available"] and final["status"] in ("REACHABLE RANGE", "OUT OF RANGE"):
        agree = d_only["naive_status"] == final["status"]
    return {"rgb_only": rgb_only, "depth_only": d_only, "naive_agrees_with_fusion": agree}


def attach_ground_truth_validation(results, scene, robot_pose):
    """[검증용] 시뮬레이터가 아는 물체의 실제 중심 위치(GT)로 거리를 계산해, 센서 추정값과 비교한다.
    이 값은 판정(range_status / FINAL)에 절대 쓰지 않는다.
    ※ 센서는 물체의 '보이는 표면'을 재므로, 물체 중심 기준인 GT 와 몇 cm 차이 나는 것이 정상이다."""
    t_wr, R_wr = robot_pose
    for r in results:
        r["gt_robot_dist"] = None
        try:
            pos, _ = scene["objects"][r["name"]].get_world_pose()
            r["gt_robot_dist"] = float(np.linalg.norm(world_to_robot_base(pos, t_wr, R_wr)))
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# 12. 한 frame 처리 (센서 융합 전체 흐름)
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# ★ [CORE-7] 융합 파이프라인: CORE-2 → 3 → 4 → 5 → 6 을 순서대로 실행
# ─────────────────────────────────────────────────────────────────────────────
def process_rgbd(camera, frame, scene, max_reach):
    """같은 frame 의 RGB 와 Depth 를 융합해 물체별 결과와 최종 판정을 만든다.

    물체마다:  RGB 검출 → mask 안쪽 depth → 3D 위치(카메라 / world) → 로봇 기준 위치 → 거리
    그 다음 :  Target 기준 최종 판정 + 단일 센서 비교 + (선택) GT 검증

    반환 bundle: {results(물체별 결과), final(최종 판정), baselines(단일 센서 비교),
                 K, K_source, cam_pose, robot_pose}"""
    rgb, depth = frame["rgb"], frame["depth"]
    K, k_src = get_intrinsics(camera)
    cam_pose = get_camera_pose_ros(camera)
    robot_pose = get_robot_base_pose(scene)          # 로봇 위치는 매번 stage 에서 새로 읽음

    detections = detect_objects_rgb(rgb, OBJECT_SPECS)                   # [CORE-2] RGB → 무엇인가
    results = []
    for spec in OBJECT_SPECS:
        r = dict(detections[spec["name"]])
        r.update({"p_cam": None, "p_world": None, "p_robot": None, "robot_dist": None,
                  "unproj_method": None, "unproj_check_m": None})
        r.update(extract_object_depth(depth, r["mask"]))                 # [CORE-3] RGB mask 로 depth pixel 선택
        if r["detected"] and r["depth_valid"]:
            u, v = r["centroid"]
            r.update(pixel_depth_to_3d(camera, u, v, r["depth_median"], K, cam_pose))   # [CORE-4] → 카메라 / world
            r["p_robot"] = world_to_robot_base(r["p_world"], *robot_pose)               # [CORE-5] → 로봇 base
            r["robot_dist"] = float(np.linalg.norm(r["p_robot"]))                       # 로봇 base 원점까지 거리
        results.append(r)

    final = evaluate_reachability(results, max_reach)                    # [CORE-6] 최종 판정
    baselines = compute_single_sensor_baselines(results, depth, K, cam_pose, robot_pose, max_reach, final)
    if SHOW_GT_VALIDATION:
        attach_ground_truth_validation(results, scene, robot_pose)
    return {"results": results, "final": final, "baselines": baselines,
            "K": K, "K_source": k_src, "cam_pose": cam_pose, "robot_pose": robot_pose}


# =============================================================================
# 13. 콘솔 출력
# =============================================================================
# 한 frame 의 결과를 콘솔 / 텍스트 파일용 표 문자열로 만든다
def format_results_text(bundle, meta):
    results, final, base = bundle["results"], bundle["final"], bundle["baselines"]
    line = "-" * 96
    out = []
    out.append("=" * 96)
    out.append(" RGB-D SENSOR FUSION | Simplified Reachability (no IK / grasp / collision planning)")
    out.append(f" Scenario {meta['scenario']} | FRAME(step) {meta['step']} | CAM FRAME {meta['cam_frame']} | "
               f"SIM TIME {meta['sim_time']:.2f} s | CAM {CAMERA_FREQUENCY} Hz | RGB+Depth source: {meta['source']}")
    out.append(line)
    out.append(f"{'Object':<18}{'Role':<12}{'Detect':<8}{'Pixels':>7}{'ValidZ':>8}{'Depth(m)':>10}"
               f"{'Robot Dist(m)':>15}{'Range':>7}   Note")
    out.append(line)
    for r in results:
        note = "" if (r["detected"] and r["depth_valid"]) else (r["det_reason"] if not r["detected"]
                                                                 else r["depth_reason"])
        out.append(f"{r['name']:<18}{r['role']:<12}{('YES' if r['detected'] else 'NO'):<8}{r['pixel_count']:>7}"
                   f"{r['depth_valid_count']:>8}{fmt(r['depth_median'], 3, 10)}{fmt(r['robot_dist'], 3, 15)}"
                   f"{r['range_status']:>7}   {note}")
    out.append(line)
    out.append(" Coordinates [m]: Cam = ROS optical (x right, y down, z = optical axis = depth) | "
               "World = Z-up | Robot = base frame")
    for r in results:
        uv = "N/A" if r["centroid"] is None else f"({r['centroid'][0]:.0f},{r['centroid'][1]:.0f})"
        out.append(f"  {r['name']:<17} uv={uv:<11} Cam={fmt_vec(r['p_cam'])}  World={fmt_vec(r['p_world'])}  "
                   f"Robot={fmt_vec(r['p_robot'])}")
    out.append(line)
    tspec = next((s for s in OBJECT_SPECS if s["name"] == TARGET_NAME), None)
    out.append(f"TARGET        : {TARGET_NAME}" + (f" ({tspec['visual']})" if tspec else ""))
    out.append(f"MAX_REACH     : {final['max_reach']:.2f} m (simplified workspace radius, NOT a manufacturer spec)")
    out.append(f"TARGET DIST   : {fmt(final['target_dist'], 3)} m (robot base -> RGB-D estimated target surface)")
    out.append(f"FINAL RESULT  : {final['status']}" + ("" if final["status"] in ("REACHABLE RANGE", "OUT OF RANGE")
                                                      else f"  [{final['reason']}]"))
    out.append(f"FRAME         : {meta['step']}")
    out.append(f"SIM TIME      : {meta['sim_time']:.2f} sec")
    out.append(line)
    ro, do = base["rgb_only"], base["depth_only"]
    cen = "N/A" if ro["centroid"] is None else f"({ro['centroid'][0]:.0f},{ro['centroid'][1]:.0f}) px"
    out.append(f"[RGB ONLY]    Target detected: {'YES' if ro['target_detected'] else 'NO'} at {cen} | "
               f"Distance: N/A | Reachability: UNKNOWN")
    if do["available"]:
        out.append(f"[DEPTH ONLY]  Nearest depth: {do['nearest_depth']:.3f} m at "
                   f"({do['pixel'][0]:.0f},{do['pixel'][1]:.0f}) px | Identity: UNKNOWN | Target reachability: UNKNOWN")
        out.append(f"              naive 'nearest surface = target' -> robot dist {do['robot_dist']:.3f} m -> "
                   f"{do['naive_status']}   (that surface actually belongs to: {do['actually_is']}, "
                   f"known only via RGB)")
    else:
        out.append("[DEPTH ONLY]  No object-like surface above table | Identity: UNKNOWN | Target reachability: UNKNOWN")
    out.append(f"[FUSION]      Target: {TARGET_NAME} | Target robot distance: {fmt(final['target_dist'], 3)} m | "
               f"Reachability: {final['status']}")
    agree = base["naive_agrees_with_fusion"]
    if agree is not None:
        out.append("              naive depth-only vs fusion: " +
                   ("AGREE (by coincidence)" if agree else "DISAGREE -> single-sensor decision would be WRONG"))
    if SHOW_GT_VALIDATION:
        out.append(line)
        out.append(" [VALIDATION ONLY - ground-truth object centers from stage, NOT used in any decision]")
        for r in results:
            err = None
            if r.get("robot_dist") is not None and r.get("gt_robot_dist") is not None:
                err = r["robot_dist"] - r["gt_robot_dist"]
            out.append(f"  {r['name']:<17} GT center dist {fmt(r.get('gt_robot_dist'), 3)} m | "
                       f"sensor est {fmt(r.get('robot_dist'), 3)} m | diff {fmt(err, 3)} m (surface vs center) | "
                       f"unproj check {fmt(r.get('unproj_check_m'), 4)} m")
    out.append("=" * 96)
    return "\n".join(out)


def print_results(bundle, meta):
    print(format_results_text(bundle, meta), flush=True)


def print_startup_info(camera, scene):
    """시작할 때 한 번, 센서 설정 · 단위 · 좌표계 · 카메라 내부 파라미터 · 카메라/로봇 위치를 출력한다."""
    K, k_src = get_intrinsics(camera)
    t_wc, R_wc, pose_src = get_camera_pose_ros(camera)
    t_wr, R_wr = get_robot_base_pose(scene)
    f_expected = np.asarray(CAMERA_TARGET, float) - np.asarray(CAMERA_POSITION, float)
    f_expected /= np.linalg.norm(f_expected)
    # 카메라 광축(z)이 의도한 바라보는 방향과 일치하는지 확인 (1 에 가까우면 정상)
    axis_dot = float(np.dot(R_wc[:, 2], f_expected))
    log("=" * 90)
    log("SENSORS (same camera prim, same render product, same frame):")
    log(f"  RGB  : 'rgb' annotator, uint8 HxWx3, {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]} px, "
        "pixel (u right, v down), origin top-left")
    log("  Depth: 'distance_to_image_plane' annotator, float32 [m], Z along optical axis (NOT Euclidean range); "
        "inf/0 = no hit")
    log(f"  Rate : CAMERA_FREQUENCY = {CAMERA_FREQUENCY} Hz, physics/render dt = {PHYSICS_DT:.4f}/{RENDERING_DT:.4f} s")
    log(f"INTRINSICS ({k_src}): fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f} [px]")
    if abs(K[0, 0] - K[1, 1]) > 0.01 * K[0, 0]:
        log("  fx and fy differ by >1% -> check aperture/aspect settings.", "WARN")
    log(f"CAMERA POSE ({pose_src}): t_wc={fmt_vec(t_wc, 3)}  optical axis(world)={fmt_vec(R_wc[:, 2], 3)}")
    log(f"  optical-axis vs look-at direction dot = {axis_dot:.4f} " +
        ("(OK)" if axis_dot > 0.99 else "(MISMATCH! camera axis convention problem)"))
    log(f"ROBOT BASE POSE (read from stage '{ROBOT_PRIM_PATH}'): t_wr={fmt_vec(t_wr, 3)}  "
        f"yaw={math.degrees(math.atan2(R_wr[1, 0], R_wr[0, 0])):.1f} deg  "
        f"{'[PLACEHOLDER]' if scene['robot_is_placeholder'] else '[Franka: ' + str(scene['robot_usd']) + ']'}")
    log("OBJECTS:")
    for spec in OBJECT_SPECS:
        xy = OBJECT_INITIAL_POSITIONS[spec["name"]]
        log(f"  {spec['name']:<17} role={object_role(spec['name']):<10} visual='{spec['visual']}' "
            f"color={spec['display_color']} init=({xy[0]:.2f}, {xy[1]:.2f}, {object_center_z(spec):.2f})")
    log(f"TARGET_NAME={TARGET_NAME}  MAX_REACH_M={MAX_REACH_M:.2f} m (simplified)  SCENARIO={SCENARIO}")
    log("Controls: 'RGB-D Fusion Monitor' panel buttons (Scenario A/B/C, Max Range -/+, Save Snapshot).")
    if SHOW_OVERLAY or SHOW_DEPTH_WINDOW:
        log("OpenCV window keys: a/b/c = scenario, s = save, [ / ] = MAX_REACH -/+ 0.05 m, q/ESC = quit")
    log("=" * 90)


# =============================================================================
# 14. OpenCV 결과 이미지 (선택 기능, --cv-windows 또는 저장 시에만 사용)
# =============================================================================
STATUS_COLORS_BGR = {"REACHABLE RANGE": (0, 200, 0), "OUT OF RANGE": (0, 0, 255)}   # 판정별 글자색 (OpenCV 는 BGR 순서)


def _put(img, text, org, color=(255, 255, 255), scale=0.45, thick=1, bg=(0, 0, 0)):
    """글자 뒤에 배경 상자를 깔아 읽기 쉽게 텍스트를 그린다. (bg=None 이면 배경 없음)"""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(rgb, bundle, meta):
    """결과를 표시한 RGB 이미지를 만든다.
    - 물체마다  : 윤곽선 · 사각형 · 이름 · 역할 · depth · 로봇 거리 · IN/OUT
    - 위쪽 여백 : TARGET / MAX REACH / TARGET DIST / FINAL
    - 아래쪽 여백: RGB only / Depth only / 융합 결과 비교"""
    if cv2 is None:
        return None
    results, final, base = bundle["results"], bundle["final"], bundle["baselines"]
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    for r in results:
        if not r["detected"]:
            continue
        is_t = r["role"] == "TARGET"
        col = (0, 255, 255) if is_t else (255, 255, 255)
        x0, y0, x1, y1 = r["bbox"]
        contours = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.drawContours(img, contours, -1, col, 1)
        cv2.rectangle(img, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), col, 3 if is_t else 1)
        cv2.drawMarker(img, (int(r["centroid"][0]), int(r["centroid"][1])), col, cv2.MARKER_CROSS, 10, 1)
        rs = r["range_status"]
        rs_col = (0, 220, 0) if rs == "IN" else ((0, 0, 255) if rs == "OUT" else (0, 165, 255))
        lines = [(f"{r['name']} [{'TARGET' if is_t else 'NON-TARGET'}]", col),
                 (f"Z {fmt(r['depth_median'])}m  R {fmt(r['robot_dist'])}m", (255, 255, 255)),
                 ({"IN": "IN RANGE", "OUT": "OUT OF RANGE"}.get(rs, "N/A") if r["depth_valid"]
                  else "INVALID DEPTH", rs_col)]
        y = y0 - 8 - 16 * (len(lines) - 1)          # 라벨은 사각형 위에 표시
        if y < 14:                                  # 화면 위로 넘어가면 사각형 아래에 표시
            y = y1 + 18
        for text, c in lines:
            _put(img, text, (max(0, x0), y), c)
            y += 16
    # 영상 위아래에 글자용 여백을 붙인 캔버스
    top, bottom = 58, 100
    canvas = np.zeros((H + top + bottom, W, 3), dtype=np.uint8)
    canvas[top:top + H] = img
    st_col = STATUS_COLORS_BGR.get(final["status"], (0, 165, 255))
    _put(canvas, f"TARGET: {TARGET_NAME}   MAX REACH: {final['max_reach']:.2f} m (simplified)", (8, 20), bg=None)
    _put(canvas, f"TARGET DIST: {fmt(final['target_dist'])} m", (8, 46), scale=0.55, bg=None)
    _put(canvas, f"FINAL: {final['status']}", (230, 46), st_col, scale=0.65, thick=2, bg=None)
    ro, do = base["rgb_only"], base["depth_only"]
    y = top + H + 18
    _put(canvas, f"[RGB ONLY]   target detected: {'YES' if ro['target_detected'] else 'NO'} | dist: N/A | "
                 f"reach: UNKNOWN", (8, y), (200, 200, 200), bg=None)
    d_txt = (f"nearest Z {do['nearest_depth']:.2f} m | id: UNKNOWN | naive: {do['naive_status']}"
             if do["available"] else "no surface | id: UNKNOWN")
    _put(canvas, f"[DEPTH ONLY] {d_txt}", (8, y + 20), (200, 200, 200), bg=None)
    _put(canvas, f"[FUSION]     {TARGET_NAME}: R {fmt(final['target_dist'])} m -> {final['status']}",
         (8, y + 40), st_col, bg=None)
    _put(canvas, f"step {meta['step']} | cam frame {meta['cam_frame']} | t {meta['sim_time']:.2f}s | "
                 f"scn {meta['scenario']} | no IK/grasp check", (8, y + 62), (150, 150, 150), scale=0.4, bg=None)
    return canvas


def make_depth_vis(depth, bundle):
    """Depth 보조 이미지 (가까울수록 밝게, 무효값은 검정). 물체 윤곽선과 Depth only 가 고른 점도 표시한다."""
    if cv2 is None:
        return None
    valid = np.isfinite(depth) & (depth > 0)
    dmin, dmax = DEPTH_VIS_RANGE_M
    norm = np.clip((np.where(valid, depth, dmax) - dmin) / (dmax - dmin), 0, 1)
    vis = cv2.applyColorMap((255 * (1.0 - norm)).astype(np.uint8), getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    vis[~valid] = 0
    for r in bundle["results"]:
        if r["mask"] is not None:
            contours = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
    do = bundle["baselines"]["depth_only"]
    if do["available"]:
        p = (int(do["pixel"][0]), int(do["pixel"][1]))
        cv2.circle(vis, p, 8, (255, 0, 255), 2)
        _put(vis, f"DEPTH-ONLY nearest: {do['nearest_depth']:.2f} m (identity unknown)", (p[0] + 10, p[1]),
             (255, 0, 255))
    _put(vis, "Depth = distance_to_image_plane [m] (aux view)", (6, 16))
    return vis


class Visualizer:
    """OpenCV 창 관리. 창에서 오류가 나면 창만 끄고 시뮬레이션과 콘솔 출력은 계속한다."""

    def __init__(self):
        self.enabled = cv2 is not None and (SHOW_OVERLAY or SHOW_DEPTH_WINDOW)
        if (SHOW_OVERLAY or SHOW_DEPTH_WINDOW) and cv2 is None:
            log("Overlay requested but OpenCV is missing -> console only.", "WARN")

    # 이미지를 창에 띄우고, 눌린 키 코드를 반환 (없으면 -1)
    def show(self, annotated, depth_vis):
        if not self.enabled:
            return -1
        try:
            if SHOW_OVERLAY and annotated is not None:
                cv2.imshow("RGB-D Fusion Result (annotated RGB)", annotated)
            if SHOW_DEPTH_WINDOW and depth_vis is not None:
                cv2.imshow("Depth (aux)", depth_vis)
            return cv2.waitKey(1) & 0xFF
        except Exception as e:  # noqa: BLE001
            log(f"OpenCV GUI failed ({e}); disabling windows. Simulation continues.", "WARN")
            self.enabled = False
            self.close()
            return -1

    # 새 이미지 없이 키 입력만 확인
    def poll(self):
        if not self.enabled:
            return -1
        try:
            return cv2.waitKey(1) & 0xFF
        except Exception:  # noqa: BLE001
            return -1

    def close(self):
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass


# =============================================================================
# 14-B. [VIZ] Viewport 에 작업 반경 그리기 (debug_draw) — 시각화 전용
# =============================================================================
def _ring_segments(center, u_axis, v_axis, radius, n_seg, a0=0.0, a1=2.0 * math.pi):
    """center 를 중심으로 u_axis·v_axis 평면 위의 원(또는 각도 a0~a1 구간의 호)을 n_seg 개 선분으로 나눈다.
    반환: (시작점 배열, 끝점 배열)"""
    ang = np.linspace(a0, a1, n_seg + 1)
    pts = (np.asarray(center, dtype=float)[None, :]
           + radius * (np.cos(ang)[:, None] * np.asarray(u_axis, dtype=float)[None, :]
                       + np.sin(ang)[:, None] * np.asarray(v_axis, dtype=float)[None, :]))
    return pts[:-1], pts[1:]


class ReachVisualizer:
    """작업 반경(MAX_REACH_M)과 로봇–Target 연결선을 Viewport 에 그린다.

    - debug_draw 는 화면에 선만 그릴 뿐 USD prim·충돌체를 만들지 않는다 → 물리와 센서 판정에 영향 없음.
    - 판정은 3D 구(sphere) 전체 기준이지만, 보기 쉽게 위쪽 반구만 그린다.
      (아래쪽 반구를 그리지 않을 뿐, 판정에서 빠지는 것은 아니다)
    - clear_lines() 는 화면의 '모든' 선을 지우므로, 반경·로봇 위치·Target 이 바뀌었을 때만
      전부 지우고 한 번에 다시 그린다. (매 frame 다시 그리거나 선이 쌓이지 않음)"""

    def __init__(self):
        if _debug_draw is None:
            raise RuntimeError("isaacsim.util.debug_draw is not available")
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.enabled = True
        self.radius = None
        self._boundary_key = None
        self._boundary = ([], [], [], [])       # 경계선 캐시: (시작점들, 끝점들, 색들, 두께들)
        self._base_world = None
        self._target_key = None
        self._target = None                     # Target 연결선: (시작점, 끝점, 색) 또는 None
        self.target_text = "N/A"

    # numpy 점 → debug_draw 가 받는 (x, y, z) float 튜플
    @staticmethod
    def _f3(p):
        return (float(p[0]), float(p[1]), float(p[2]))

    def _build_boundary(self, t_wr, R_wr, r):
        """로봇 base 축(ex, ey, ez) 기준으로 위쪽 반구 wireframe 과 작업대 위 원을 선분 목록으로 만든다."""
        ex, ey, ez = R_wr[:, 0], R_wr[:, 1], R_wr[:, 2]
        segs = []                                                   # (시작점들, 끝점들, 색, 두께)
        bc, bw = REACH_VIZ_BOUNDARY_COLOR, REACH_VIZ_BOUNDARY_WIDTH
        # (1) 세로 반원 2개 (로봇 기준 XZ, YZ 평면의 위쪽 절반)
        segs.append((*_ring_segments(t_wr, ex, ez, r, 24, 0.0, math.pi), bc, bw))
        segs.append((*_ring_segments(t_wr, ey, ez, r, 24, 0.0, math.pi), bc, bw))
        # (2) 위도선 (30°, 60° 높이의 수평 원)
        for lat in REACH_VIZ_LATITUDES_DEG:
            el = math.radians(lat)
            segs.append((*_ring_segments(t_wr + r * math.sin(el) * ez, ex, ey, r * math.cos(el), 40), bc, bw))
        # (3) 작업대 위 원: 판정 구(sphere)와 작업대 표면이 만나는 원
        #     로봇 base 가 작업대 높이에 있으면 dz ≈ 0 이므로 이 원의 반지름 ≈ MAX_REACH_M
        z_plane = TABLE_TOP_Z + REACH_VIZ_TABLE_OFFSET_M
        dz = z_plane - t_wr[2]
        if abs(dz) < r:
            r_table = math.sqrt(r * r - dz * dz)
            segs.append((*_ring_segments((t_wr[0], t_wr[1], z_plane), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                         r_table, 72), REACH_VIZ_TABLE_RING_COLOR, REACH_VIZ_TABLE_RING_WIDTH))
        # (4) 로봇 base 높이의 적도 원: 작업대 원과 1 cm 이내로 겹치면 생략
        if abs(dz) > 0.01:
            segs.append((*_ring_segments(t_wr, ex, ey, r, 64), bc, bw))
        # (5) 로봇 base 중심의 십자 표시
        for axis in (ex, ey, ez):
            segs.append((np.array([t_wr - 0.04 * axis]), np.array([t_wr + 0.04 * axis]), REACH_VIZ_BASE_COLOR, 2.0))
        # 선분 묶음들을 draw_lines() 가 받는 평평한 리스트로 펼치기
        starts, ends, colors, widths = [], [], [], []
        for s, e, c, w in segs:
            starts += [self._f3(p) for p in s]
            ends += [self._f3(p) for p in e]
            colors += [tuple(c)] * len(s)
            widths += [float(w)] * len(s)
        return starts, ends, colors, widths

    def update_boundary(self, robot_pose, max_reach):
        """반경이나 로봇 위치가 바뀐 경우에만 경계선을 다시 계산해 다시 그린다. 다시 그렸으면 True."""
        t_wr, R_wr = robot_pose
        key = (round(float(max_reach), 4), tuple(np.round(t_wr, 3)), tuple(np.round(R_wr, 3).ravel()))
        if key == self._boundary_key:
            return False
        self._boundary_key = key
        self.radius = float(max_reach)
        self._base_world = np.asarray(t_wr, dtype=float).copy()
        self._boundary = self._build_boundary(np.asarray(t_wr, dtype=float), np.asarray(R_wr, dtype=float),
                                              float(max_reach))
        self._redraw()
        return True

    def update_target_line(self, robot_world, target_world, status):
        """로봇 base 와 Target 추정 위치(RGB-D 로 계산한 p_world)를 잇는 선을 갱신한다. GT 는 쓰지 않는다.
        선 색: REACHABLE = 초록, OUT OF RANGE = 빨강, 그 외(미검출 / depth 무효) = 선 숨김"""
        if target_world is not None and status in ("REACHABLE RANGE", "OUT OF RANGE"):
            color = REACH_VIZ_IN_COLOR if status == "REACHABLE RANGE" else REACH_VIZ_OUT_COLOR
            key = (status, tuple(np.round(target_world, 3)), tuple(np.round(robot_world, 3)))
            new = (self._f3(robot_world), self._f3(target_world), color)
            self.target_text = "IN (green)" if status == "REACHABLE RANGE" else "OUT (red)"
        else:
            key, new = None, None
            self.target_text = "hidden (" + status + ")"
        if key == self._target_key:
            return False
        self._target_key, self._target = key, new
        self._redraw()
        return True

    # 시각화 켜기 / 끄기
    def set_enabled(self, on):
        self.enabled = bool(on)
        self._redraw()

    def clear(self):
        """debug_draw 로 그린 모든 선과 점을 지운다."""
        self._draw.clear_lines()
        self._draw.clear_points()

    # 화면을 지우고, 캐시된 경계선 + Target 연결선 + 점들을 한 번에 다시 그린다
    def _redraw(self):
        self.clear()
        if not self.enabled:
            return
        starts, ends, colors, widths = (list(x) for x in self._boundary)
        points, pcolors, psizes = [], [], []
        if self._base_world is not None:
            points.append(self._f3(self._base_world))
            pcolors.append(REACH_VIZ_BASE_COLOR)
            psizes.append(6.0)
        if self._target is not None:
            s, e, c = self._target
            starts.append(s)
            ends.append(e)
            colors.append(c)
            widths.append(REACH_VIZ_TARGET_LINE_WIDTH)
            points.append(e)
            pcolors.append(c)
            psizes.append(REACH_VIZ_TARGET_MARKER_SIZE)
        if starts:
            self._draw.draw_lines(starts, ends, colors, widths)
        if points:
            self._draw.draw_points(points, pcolors, psizes)

    # 그린 것을 모두 지우고 debug_draw 인터페이스를 반납한다
    def shutdown(self):
        try:
            self.clear()
            _debug_draw.release_debug_draw_interface(self._draw)
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# 15. Isaac Sim 안의 결과 패널 (omni.ui) — 표시만 하고 계산은 하지 않음
# =============================================================================
def _abgr(r, g, b, a=255):
    """omni.ui 색상 값을 만든다. (omni.ui 는 0xAABBGGRR 정수 형식을 쓴다)"""
    return (a << 24) | (b << 16) | (g << 8) | r


# 패널에서 쓰는 색
UI_WHITE = _abgr(235, 235, 235)
UI_GRAY = _abgr(150, 150, 155)
UI_HEADER = _abgr(120, 180, 255)
UI_YELLOW = _abgr(255, 210, 0)
UI_GREEN = _abgr(60, 210, 90)
UI_RED = _abgr(240, 70, 70)
UI_ORANGE = _abgr(255, 165, 0)
UI_CARD_BG = _abgr(38, 40, 45)
UI_TARGET_BG = _abgr(62, 54, 22)
UI_FINAL_GREEN_BG = _abgr(25, 110, 45)
UI_FINAL_RED_BG = _abgr(140, 30, 30)
UI_FINAL_NEUTRAL_BG = _abgr(90, 70, 20)


def _m(v):
    """거리 값을 'x.xxx m' 로 표시한다. 값이 없으면 N/A."""
    return "N/A" if v is None else f"{v:.3f} m"


class FusionMonitorUI:
    """Isaac Sim 화면 안에 뜨는 'RGB-D Fusion Monitor' 패널.

    - 위젯은 생성자에서 한 번만 만들고, update() 에서는 글자와 색만 바꾼다.
    - process_rgbd() 결과(bundle)를 그대로 보여준다. GT 값은 표시하지 않는다.
    - 버튼을 눌러도 바로 실행하지 않고 요청 목록에 넣어 둔다.
      main loop 가 simulation step 사이에 꺼내서 한 번씩 처리한다 → 센서 처리와 UI 가 서로 간섭하지 않음."""

    def __init__(self, title="RGB-D Fusion Monitor"):
        self._requests = []
        self._ticks = 0
        self._dock_attempted = False
        self.obj_widgets = {}
        self.window = ui.Window(title, width=MONITOR_WIDTH, height=MONITOR_HEIGHT)
        try:   # 떠 있는 창의 기본 위치: 화면 오른쪽 (Stage / Property 패널의 왼쪽)
            self.window.position_x = max(0, 1600 - MONITOR_WIDTH - 430)
            self.window.position_y = 40
        except Exception:  # noqa: BLE001
            pass
        self._build()

    # ---- 위젯 생성 helper ----
    # 구역 제목 + 구분선
    @staticmethod
    def _section(text):
        ui.Spacer(height=8)
        ui.Label(text, height=20, style={"font_size": 15, "color": UI_HEADER})
        ui.Line(height=2, style={"color": UI_GRAY})
        ui.Spacer(height=2)

    # "항목 이름 | 값" 한 줄. 나중에 값을 바꿀 수 있도록 값 Label 을 반환
    @staticmethod
    def _kv(key, key_width=120):
        with ui.HStack(height=20):
            ui.Label(key, width=key_width, style={"color": UI_GRAY})
            value = ui.Label("N/A", style={"color": UI_WHITE})
        return value

    # 버튼 요청을 목록에 넣는다 (실제 처리는 main loop 에서)
    def _request(self, action):
        self._requests.append(action)
        self.lbl_action.text = f"Requested: {action}"

    # 패널 전체 레이아웃 (위에서부터 TARGET → FINAL → OBJECTS → 센서 비교 → 시각화 → 시나리오 버튼)
    def _build(self):
        with self.window.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=2, height=0):
                    ui.Label("RGB-D FUSION MONITOR", height=30, style={"font_size": 21, "color": UI_WHITE})
                    ui.Label("Simplified reachability (no IK / grasp / collision check)", height=0,
                             word_wrap=True, style={"font_size": 12, "color": UI_GRAY})
                    self.lbl_meta = ui.Label("Waiting for sensor data...", height=0, word_wrap=True,
                                             style={"font_size": 12, "color": UI_GRAY})

                    # ---- TARGET / FINAL ----
                    self._section("TARGET")
                    ui.Label(TARGET_NAME, height=26, style={"font_size": 19, "color": UI_YELLOW})
                    self.lbl_target_depth = self._kv("Target Depth")
                    self.lbl_target_dist = self._kv("Robot Distance")
                    self.lbl_max = self._kv("Maximum Range")
                    self._section("FINAL RESULT")
                    with ui.ZStack(height=46):
                        self.final_bg = ui.Rectangle(style={"background_color": UI_FINAL_NEUTRAL_BG,
                                                            "border_radius": 6})
                        self.lbl_final = ui.Label("WAITING", alignment=ui.Alignment.CENTER,
                                                  style={"font_size": 22, "color": UI_WHITE})
                    self.lbl_final_note = ui.Label("", height=0, word_wrap=True,
                                                   style={"font_size": 12, "color": UI_GRAY})

                    # ---- OBJECTS ----
                    self._section("OBJECTS")
                    for spec in OBJECT_SPECS:
                        self._build_object_card(spec)

                    # ---- SENSOR COMPARISON ----
                    self._section("SENSOR COMPARISON")
                    ui.Label("RGB ONLY", height=20, style={"color": UI_WHITE, "font_size": 14})
                    self.lbl_rgb_det = self._kv("Target detected")
                    self.lbl_rgb_dist = self._kv("Distance")
                    self.lbl_rgb_dec = self._kv("Decision")
                    ui.Spacer(height=4)
                    ui.Label("DEPTH ONLY", height=20, style={"color": UI_WHITE, "font_size": 14})
                    self.lbl_d_near = self._kv("Nearest depth")
                    self.lbl_d_id = self._kv("Identity")
                    self.lbl_d_dec = self._kv("Target decision")
                    self.lbl_d_naive = self._kv("Naive 'nearest'")
                    ui.Spacer(height=4)
                    ui.Label("RGB + DEPTH (FUSION)", height=20, style={"color": UI_WHITE, "font_size": 14})
                    self.lbl_f_target = self._kv("Target")
                    self.lbl_f_dist = self._kv("Robot Distance")
                    self.lbl_f_dec = self._kv("Decision")
                    self.lbl_agree = ui.Label("", height=0, word_wrap=True, style={"font_size": 12, "color": UI_GRAY})

                    # ---- REACH VISUALIZATION ----
                    self._section("REACH VISUALIZATION")
                    self.lbl_reach_radius = self._kv("Radius")
                    self.lbl_reach_boundary = self._kv("Boundary")
                    self.lbl_reach_line = self._kv("Target line")
                    ui.Label("Decision uses the full 3D sphere; viewport shows the upper hemisphere only.",
                             height=0, word_wrap=True, style={"font_size": 12, "color": UI_GRAY})
                    ui.Button("Toggle Reach Visualization", height=26,
                              clicked_fn=lambda: self._request("reachviz"))

                    # ---- SCENARIO CONTROL ----
                    self._section("SCENARIO CONTROL")
                    ui.Button("Scenario A - Target In Range", height=28,
                              clicked_fn=lambda: self._request("scenario:A"))
                    ui.Button("Scenario B - Target Out of Range", height=28,
                              clicked_fn=lambda: self._request("scenario:B"))
                    ui.Button("Scenario C - Distractor Closer", height=28,
                              clicked_fn=lambda: self._request("scenario:C"))
                    with ui.HStack(height=26, spacing=4):
                        ui.Button("Max Range -0.05", clicked_fn=lambda: self._request("reach-"))
                        ui.Button("Max Range +0.05", clicked_fn=lambda: self._request("reach+"))
                    ui.Button("Save Snapshot", height=28, clicked_fn=lambda: self._request("save"))
                    self.lbl_action = ui.Label("", height=0, word_wrap=True, style={"font_size": 12, "color": UI_GRAY})
                    ui.Spacer(height=10)

    # 물체 하나의 카드: 왼쪽 색 띠 + 이름 · 역할 · 검출 · depth · 거리 · 판정 (Target 은 노란 테두리)
    def _build_object_card(self, spec):
        name = spec["name"]
        is_t = object_role(name) == "TARGET"
        r, g, b = [int(255 * c) for c in spec["display_color"]]
        with ui.ZStack(height=0):
            ui.Rectangle(style={"background_color": UI_TARGET_BG if is_t else UI_CARD_BG,
                                "border_color": UI_YELLOW if is_t else UI_GRAY,
                                "border_width": 2 if is_t else 1, "border_radius": 4})
            with ui.HStack(height=0):
                ui.Rectangle(width=6, style={"background_color": _abgr(r, g, b)})
                ui.Spacer(width=8)
                with ui.VStack(height=0, spacing=1):
                    ui.Spacer(height=4)
                    ui.Label(name, height=20, style={"font_size": 16, "color": UI_WHITE})
                    ui.Label("TARGET" if is_t else "NON-TARGET", height=18,
                             style={"font_size": 13, "color": UI_YELLOW if is_t else UI_GRAY})
                    self.obj_widgets[name] = {"det": self._kv("Detected"), "depth": self._kv("Depth"),
                                              "dist": self._kv("Robot Dist"), "range": self._kv("Range")}
                    ui.Spacer(height=4)
        ui.Spacer(height=4)

    # ---- main loop 에서 호출 ----
    def tick(self):
        """매 loop 호출된다. 창이 실제로 만들어질 때까지 몇 번 기다린 뒤, Viewport 오른쪽에 붙이기를 한 번만 시도한다."""
        self._ticks += 1
        if not self._dock_attempted and self._ticks >= 10:
            self._dock_attempted = True
            if not MONITOR_DOCK_RIGHT_OF_VIEWPORT:
                return
            try:
                vp = ui.Workspace.get_window("Viewport")
                if vp is None:
                    log("Viewport window not found; Fusion Monitor stays floating.", "WARN")
                    return
                self.window.dock_in(vp, ui.DockPosition.RIGHT, MONITOR_DOCK_RATIO)
                log("Fusion Monitor docked to the right of the Viewport.")
            except Exception as e:  # noqa: BLE001
                log(f"Docking failed ({e}); Fusion Monitor stays floating.", "WARN")

    # 쌓인 버튼 요청을 꺼내고 목록을 비운다
    def pop_requests(self):
        reqs, self._requests = self._requests, []
        return reqs

    def set_status(self, text):
        self.lbl_meta.text = text

    def set_action_text(self, text):
        self.lbl_action.text = text

    def update_reach(self, radius, boundary_on, line_text):
        """REACH VISUALIZATION 구역의 글자를 갱신한다."""
        self.lbl_reach_radius.text = "N/A" if radius is None else f"{radius:.2f} m"
        self.lbl_reach_boundary.text = boundary_on
        self.lbl_reach_boundary.style = {"color": UI_GREEN if boundary_on == "ON" else UI_GRAY}
        self.lbl_reach_line.text = line_text

    def update(self, bundle, meta):
        """센서 융합 결과(bundle)로 패널의 글자와 색을 갱신한다. (위젯은 새로 만들지 않음)"""
        results, final, base = bundle["results"], bundle["final"], bundle["baselines"]
        target = next((r for r in results if r["role"] == "TARGET"), None)
        status = final["status"]
        # 최종 판정에 따라 글자색 / 배경색 선택
        if status == "REACHABLE RANGE":
            fg, bg = UI_GREEN, UI_FINAL_GREEN_BG
        elif status == "OUT OF RANGE":
            fg, bg = UI_RED, UI_FINAL_RED_BG
        else:
            fg, bg = UI_ORANGE, UI_FINAL_NEUTRAL_BG

        self.lbl_meta.text = (f"Frame {meta['step']} | Cam frame {meta['cam_frame']} | "
                              f"Sim {meta['sim_time']:.2f} s | {CAMERA_FREQUENCY} Hz | Scenario {meta['scenario']}")
        self.lbl_target_depth.text = _m(target["depth_median"] if target else None)
        self.lbl_target_dist.text = _m(final["target_dist"])
        self.lbl_max.text = f"{final['max_reach']:.3f} m"
        self.lbl_final.text = status
        self.final_bg.style = {"background_color": bg, "border_radius": 6}
        self.lbl_final_note.text = ("" if status in ("REACHABLE RANGE", "OUT OF RANGE")
                                    else f"Reason: {final.get('reason', '')}")

        # 물체별 카드
        for r in results:
            w = self.obj_widgets.get(r["name"])
            if w is None:
                continue
            w["det"].text = "YES" if r["detected"] else "NO"
            w["det"].style = {"color": UI_WHITE if r["detected"] else UI_ORANGE}
            w["depth"].text = _m(r["depth_median"]) if r["depth_valid"] else ("INVALID" if r["detected"] else "N/A")
            w["dist"].text = _m(r["robot_dist"])
            rs = r["range_status"]
            w["range"].text = rs
            w["range"].style = {"color": UI_GREEN if rs == "IN" else (UI_RED if rs == "OUT" else UI_GRAY)}

        # 단일 센서 비교
        ro, do = base["rgb_only"], base["depth_only"]
        self.lbl_rgb_det.text = "YES" if ro["target_detected"] else "NO"
        self.lbl_rgb_dist.text = "N/A"
        self.lbl_rgb_dec.text = "UNKNOWN"
        self.lbl_d_near.text = _m(do["nearest_depth"])
        self.lbl_d_id.text = "UNKNOWN"
        self.lbl_d_dec.text = "UNKNOWN"
        self.lbl_d_naive.text = (f"{do['naive_status']} ({_m(do['robot_dist'])})" if do["available"] else "N/A")
        self.lbl_f_target.text = TARGET_NAME
        self.lbl_f_dist.text = _m(final["target_dist"])
        self.lbl_f_dec.text = status
        self.lbl_f_dec.style = {"color": fg}
        agree = base["naive_agrees_with_fusion"]
        if agree is None:
            self.lbl_agree.text = ""
        elif agree:
            self.lbl_agree.text = "Naive depth-only agrees with fusion (by coincidence)."
        else:
            self.lbl_agree.text = (f"Naive depth-only DISAGREES with fusion: the nearest surface is "
                                   f"{do['actually_is']}, not the target -> single-sensor decision would be wrong.")

    def destroy(self):
        try:
            self.window.visible = False
            self.window.destroy()
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# 16. 결과 저장 (output/ 폴더)
# =============================================================================
# results.csv 의 열 목록 (물체 하나당 한 줄)
CSV_FIELDS = ["wall_time", "scenario", "frame", "camera_frame", "sim_time", "object", "role", "detected",
              "pixel_count", "valid_depth_px", "depth_m", "u", "v", "cam_x", "cam_y", "cam_z",
              "world_x", "world_y", "world_z", "robot_x", "robot_y", "robot_z", "robot_distance_m",
              "range_status", "max_reach_m", "final_result", "depth_only_nearest_m", "depth_only_naive_result"]


def save_results(bundle, meta, rgb, depth, annotated, depth_vis):
    """output/ 폴더에 결과를 저장한다.
      {tag}_annotated_rgb.png : 결과를 표시한 RGB 이미지
      {tag}_depth_vis.png     : depth 이미지
      {tag}_summary.txt       : 콘솔에 출력되는 것과 같은 결과 표
      results.csv             : 물체별 수치 (파일 끝에 계속 추가됨)"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tag = f"scn{meta['scenario']}_f{meta['step']:06d}"
        saved = []
        if cv2 is not None:
            if annotated is not None:
                p = os.path.join(OUTPUT_DIR, f"{tag}_annotated_rgb.png")
                cv2.imwrite(p, annotated)
                saved.append(p)
            if depth_vis is not None:
                p = os.path.join(OUTPUT_DIR, f"{tag}_depth_vis.png")
                cv2.imwrite(p, depth_vis)
                saved.append(p)
        else:                                   # OpenCV 가 없으면 원본 RGB 만 PIL 로 저장
            try:
                from PIL import Image
                p = os.path.join(OUTPUT_DIR, f"{tag}_raw_rgb.png")
                Image.fromarray(rgb).save(p)
                saved.append(p)
            except Exception as e:  # noqa: BLE001
                log(f"No OpenCV/PIL for image saving: {e}", "WARN")
        p = os.path.join(OUTPUT_DIR, f"{tag}_summary.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(format_results_text(bundle, meta) + "\n")
        saved.append(p)

        csv_path = os.path.join(OUTPUT_DIR, "results.csv")
        new_file = not os.path.exists(csv_path)
        final, do = bundle["final"], bundle["baselines"]["depth_only"]
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new_file:
                wr.writeheader()
            for r in bundle["results"]:
                # 벡터의 i 번째 성분을 문자열로 (값이 없으면 빈칸)
                def comp(vec, i):
                    return "" if vec is None else f"{float(vec[i]):.4f}"
                wr.writerow({
                    "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"), "scenario": meta["scenario"],
                    "frame": meta["step"], "camera_frame": meta["cam_frame"], "sim_time": f"{meta['sim_time']:.3f}",
                    "object": r["name"], "role": r["role"], "detected": "YES" if r["detected"] else "NO",
                    "pixel_count": r["pixel_count"], "valid_depth_px": r["depth_valid_count"],
                    "depth_m": "" if r["depth_median"] is None else f"{r['depth_median']:.4f}",
                    "u": "" if r["centroid"] is None else f"{r['centroid'][0]:.1f}",
                    "v": "" if r["centroid"] is None else f"{r['centroid'][1]:.1f}",
                    "cam_x": comp(r["p_cam"], 0), "cam_y": comp(r["p_cam"], 1), "cam_z": comp(r["p_cam"], 2),
                    "world_x": comp(r["p_world"], 0), "world_y": comp(r["p_world"], 1),
                    "world_z": comp(r["p_world"], 2),
                    "robot_x": comp(r["p_robot"], 0), "robot_y": comp(r["p_robot"], 1),
                    "robot_z": comp(r["p_robot"], 2),
                    "robot_distance_m": "" if r["robot_dist"] is None else f"{r['robot_dist']:.4f}",
                    "range_status": r["range_status"], "max_reach_m": f"{final['max_reach']:.3f}",
                    "final_result": final["status"],
                    "depth_only_nearest_m": "" if do["nearest_depth"] is None else f"{do['nearest_depth']:.4f}",
                    "depth_only_naive_result": do["naive_status"],
                })
        saved.append(csv_path)
        log("Saved: " + " | ".join(saved))
    except Exception as e:  # noqa: BLE001
        log(f"save_results failed: {e}", "ERROR")
        traceback.print_exc()


# =============================================================================
# 17. Main loop
# =============================================================================
def _key_to_action(key):
    """OpenCV 창의 키 입력을 패널 버튼과 같은 요청 문자열로 바꾼다."""
    if key in (ord("q"), 27):
        return "quit"
    if key in (ord("a"), ord("b"), ord("c")):
        return "scenario:" + chr(key).upper()
    return {ord("s"): "save", ord("]"): "reach+", ord("["): "reach-"}.get(key)


# scene 을 만든 뒤, 매 simulation step 마다 "센서 처리 → 결과 표시 → 사용자 요청 처리"를 반복한다
def main():
    global SCENARIO, MAX_REACH_M
    if TARGET_NAME not in [s["name"] for s in OBJECT_SPECS]:
        log(f"TARGET_NAME '{TARGET_NAME}' is not one of {[s['name'] for s in OBJECT_SPECS]}", "ERROR")
        return

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    scene = setup_scene(world)
    camera = setup_camera()

    log("world.reset() ... (asset loading / shader compile can take minutes on first run)")
    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()   # RGB 와 같은 카메라 렌더 결과에 Depth(Z) 출력 추가
    apply_robot_home(scene)
    print_startup_info(camera, scene)
    validate_scene_layout(camera, scene)

    # ---- 결과 패널 (센서 처리와 분리. 만들기에 실패해도 콘솔 출력만으로 계속) ----
    ui_panel = None
    if SHOW_FUSION_MONITOR and ui is not None:
        try:
            ui_panel = FusionMonitorUI()
            log("RGB-D Fusion Monitor panel created inside Isaac Sim.")
        except Exception as e:  # noqa: BLE001
            ui_panel = None
            log(f"Fusion Monitor UI creation failed ({e}); continuing console-only.", "WARN")
            traceback.print_exc()
    viz = Visualizer()

    # ---- [VIZ] Viewport 작업 반경 시각화 (실패해도 센서 처리 / 판정에는 영향 없음) ----
    reach_viz = None
    if REACH_VIZ_ENABLED and _debug_draw is not None:
        try:
            reach_viz = ReachVisualizer()
            log("Reach visualization (debug_draw) enabled: upper-hemisphere wireframe + table ring + target line.")
        except Exception as e:  # noqa: BLE001
            reach_viz = None
            log(f"ReachVisualizer creation failed ({e}); continuing without reach visualization.", "WARN")

    # ---- loop 상태 변수 ----
    step = 0
    steps_since_reset = 0
    last_cam_frame = None
    last_print_step = -10 ** 9
    last_save_step = -10 ** 9
    reset_needed = False
    save_requested = False

    while simulation_app.is_running():
        world.step(render=True)
        key = -1
        actions = []
        if ui_panel is not None:
            try:
                ui_panel.tick()
                actions += ui_panel.pop_requests()
            except Exception as e:  # noqa: BLE001
                log(f"Fusion Monitor UI error ({e}); disabling panel, console continues.", "WARN")
                ui_panel = None

        if world.is_stopped():
            # Stop 상태: 다시 Play 하면 reset 한다 (물체는 기본 위치로 돌아감)
            reset_needed = True
            if ui_panel is not None:
                ui_panel.set_status("STOPPED - press Play to resume sensing")
            key = viz.poll()
        elif not world.is_playing():          # Pause 상태
            if ui_panel is not None:
                ui_panel.set_status("PAUSED")
            key = viz.poll()
        elif reset_needed:                    # Stop 후 다시 Play: scene 초기화
            world.reset()
            apply_robot_home(scene)
            reset_needed = False
            steps_since_reset = 0
            last_cam_frame = None
        else:                                 # 실행 중: 센서 처리
            step += 1
            steps_since_reset += 1
            frame = acquire_rgbd_frame(camera) if steps_since_reset >= WARMUP_FRAMES else None
            if frame is None:
                if steps_since_reset >= WARMUP_FRAMES:
                    RL.warn("noframe", "Camera buffers not ready yet (empty RGB/Depth); waiting...")
                key = viz.poll()
            elif frame["rendering_frame"] is not None and frame["rendering_frame"] == last_cam_frame:
                key = viz.poll()            # 새 카메라 frame 이 아니면 건너뜀 (카메라는 CAMERA_FREQUENCY 로만 갱신)
            else:
                # ---- 같은 frame 의 RGB, Depth 로 센서 융합 ----
                cam_frame = frame["rendering_frame"]
                last_cam_frame = cam_frame
                meta = {"scenario": SCENARIO, "step": step, "sim_time": float(world.current_time),
                        "cam_frame": cam_frame if cam_frame is not None else "N/A", "source": frame["source"]}
                bundle = None
                try:
                    bundle = process_rgbd(camera, frame, scene, MAX_REACH_M)
                    want_imgs = SAVE_RESULTS or save_requested
                    annotated = draw_overlay(frame["rgb"], bundle, meta) if (SHOW_OVERLAY or want_imgs) else None
                    depth_vis = (make_depth_vis(frame["depth"], bundle)
                                 if (SHOW_DEPTH_WINDOW or want_imgs) else None)
                    if step - last_print_step >= PRINT_INTERVAL_FRAMES:
                        print_results(bundle, meta)
                        last_print_step = step
                    if save_requested or (SAVE_RESULTS and step - last_save_step >= SAVE_INTERVAL_FRAMES):
                        save_results(bundle, meta, frame["rgb"], frame["depth"], annotated, depth_vis)
                        if save_requested and ui_panel is not None:
                            ui_panel.set_action_text(f"Snapshot saved (frame {step}) -> {OUTPUT_DIR}")
                        last_save_step = step
                        save_requested = False
                    key = viz.show(annotated, depth_vis)
                except Exception as e:  # noqa: BLE001
                    RL.warn("process", f"Frame processing error at step {step}: {e}\n{traceback.format_exc()}",
                            "ERROR")
                    key = viz.poll()
                # [VIZ] Target 연결선: 이번 frame 의 RGB-D 추정 위치를 사용 (GT 아님)
                #       Target 미검출 / depth 무효이면 선을 숨긴다.
                if bundle is not None and reach_viz is not None:
                    try:
                        tgt = next((r for r in bundle["results"] if r["role"] == "TARGET"), None)
                        reach_viz.update_target_line(bundle["robot_pose"][0],
                                                     tgt["p_world"] if tgt is not None else None,
                                                     bundle["final"]["status"])
                    except Exception as e:  # noqa: BLE001
                        log(f"Reach visualization error ({e}); disabling it, sensing continues.", "WARN")
                        reach_viz = None
                # 패널 갱신: 같은 bundle 을 그대로 전달 (패널 오류는 센서 처리에 영향 없음)
                if bundle is not None and ui_panel is not None:
                    try:
                        ui_panel.update(bundle, meta)
                    except Exception as e:  # noqa: BLE001
                        log(f"Fusion Monitor update failed ({e}); disabling panel, console continues.", "WARN")
                        traceback.print_exc()
                        ui_panel = None

        # ---- 사용자 요청 처리 (패널 버튼 / OpenCV 키). 요청마다 한 번만 실행 ----
        key_action = _key_to_action(key)
        if key_action:
            actions.append(key_action)
        quit_requested = False
        for action in actions:
            if action == "quit":
                quit_requested = True
            elif action.startswith("scenario:"):
                SCENARIO = action.split(":", 1)[1]
                apply_scenario(scene, SCENARIO)
                last_print_step = -10 ** 9
                if ui_panel is not None:
                    ui_panel.set_action_text(f"Scenario {SCENARIO} applied once. Objects are free to move.")
            elif action == "save":
                save_requested = True           # 다음에 처리되는 frame 의 결과를 저장
            elif action in ("reach+", "reach-"):
                delta = 0.05 if action == "reach+" else -0.05
                MAX_REACH_M = max(0.05, round(MAX_REACH_M + delta, 3))
                log(f"MAX_REACH_M -> {MAX_REACH_M:.2f} m")
                last_print_step = -10 ** 9
                if ui_panel is not None:
                    ui_panel.set_action_text(f"Maximum Range set to {MAX_REACH_M:.2f} m")
            elif action == "reachviz" and reach_viz is not None:
                reach_viz.set_enabled(not reach_viz.enabled)
                if ui_panel is not None:
                    ui_panel.set_action_text(f"Reach visualization {'ON' if reach_viz.enabled else 'OFF'}")
        if quit_requested:
            log("Quit requested.")
            break

        # ---- [VIZ] 작업 반경 경계선: 현재 MAX_REACH_M 와 로봇 위치 기준.
        #      둘 중 하나가 바뀐 경우에만 다시 그린다 (변화가 없으면 아무것도 하지 않음) ----
        if reach_viz is not None:
            try:
                reach_viz.update_boundary(get_robot_base_pose(scene), MAX_REACH_M)
            except Exception as e:  # noqa: BLE001
                log(f"Reach visualization error ({e}); disabling it, sensing continues.", "WARN")
                reach_viz = None
        # 패널의 REACH VISUALIZATION 구역 갱신
        if ui_panel is not None:
            try:
                if reach_viz is None:
                    ui_panel.update_reach(MAX_REACH_M, "OFF (debug_draw unavailable)", "N/A")
                else:
                    ui_panel.update_reach(reach_viz.radius, "ON" if reach_viz.enabled else "OFF",
                                          reach_viz.target_text if reach_viz.enabled else "OFF")
            except Exception as e:  # noqa: BLE001
                log(f"Fusion Monitor reach info update failed ({e})", "WARN")

    viz.close()
    if reach_viz is not None:
        reach_viz.shutdown()
    if ui_panel is not None:
        ui_panel.destroy()


if __name__ == "__main__":
    try:
        main()
    except Exception as _e:  # noqa: BLE001
        log(f"Fatal error: {_e}", "ERROR")
        traceback.print_exc()
    finally:
        simulation_app.close()
