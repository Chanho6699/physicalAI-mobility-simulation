# -*- coding: utf-8 -*-
"""
sensor_fusion_reachability.py
=============================
Isaac Sim 5.0 / 5.1  Standalone Python 과제 코드

주제: RGB + Depth 센서 융합 기반 "Target 식별 + 단순화된 작업 반경(Simplified Reachability) 판단"

  질문: "여러 객체 중 현재 작업 대상인 Target 은 무엇이며,
         그 Target 은 로봇팔의 설정된 작업 가능 반경(MAX_REACH_M) 안에 있는가?"

데이터 흐름 (같은 카메라, 같은 render frame t):
  RGB(t)   -> HSV 색상 검출 -> 객체별 mask / bbox / centroid(u,v)        [WHAT]
  Depth(t) -> 객체 mask 내부의 valid depth 만 선택 -> median Z            [WHERE]
  (u, v, Z) -> Camera frame(ROS optical) -> World -> Robot base frame
            -> robot distance vs MAX_REACH_M -> Target 기준 FINAL DECISION

주의 사항:
  - REACHABLE RANGE 는 "robot base 원점 기준 반경 MAX_REACH_M 구 안에 있다"는 뜻일 뿐,
    IK / grasp / 충돌 회피가 가능하다는 보장이 아니다.
  - Ground-truth(stage) object pose 는 "VALIDATION ONLY" 출력에만 사용하며
    어떤 판단(Range / FINAL)에도 사용하지 않는다.

실행 (Windows, Isaac Sim 설치 폴더에서):
  python.bat C:\\path\\to\\sensor_fusion_reachability.py --scenario A

★ 핵심 코드 지도 (에디터에서 "[CORE-" 로 검색하면 바로 이동)
  [CORE-1] acquire_rgbd_frame()      : 같은 frame 의 RGB(t) + Depth(t) 획득 (동기화)
  [CORE-2] detect_objects_rgb()      : RGB -> 객체 identity / Target 여부 / mask           (WHAT)
  [CORE-3] extract_object_depth()    : RGB mask 가 고른 pixel 의 depth 만 사용 -> median Z  (융합 지점)
  [CORE-4] pixel_depth_to_3d()       : (u, v, Z) -> camera frame -> world                  (WHERE)
  [CORE-5] world_to_robot_base()     : world -> robot base frame (역변환 방향 주의)
  [CORE-6] evaluate_reachability()   : 3D 거리 <= MAX_REACH_M ? -> Target 기준 FINAL
  [CORE-7] process_rgbd()            : 위 1~6 을 한 frame 에 연결하는 Fusion 파이프라인
  [CORE-8] depth_only_baseline()     : 단일 센서(Depth only) 비교용 baseline
  [VIZ]    ReachVisualizer           : Viewport 에 MAX_REACH_M 반경 / Target 연결선 표시 (시각화 전용)
"""

# =============================================================================
# 0. 명령행 인자 (SimulationApp 생성 전에 파싱해야 함)
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
    # Kit 이 추가 인자를 넘길 수 있으므로 parse_known_args 사용
    args, _unknown = parser.parse_known_args()
    return args


ARGS = _parse_args()

# =============================================================================
# 1. CONFIG  (과제 실험 시 이 블록만 수정하면 됨)
# =============================================================================

# ---- 실험 조건 ----------------------------------------------------------------
SCENARIO = (ARGS.scenario or "A").upper()          # "A" / "B" / "C" (실행 중 a/b/c 키로도 변경 가능)
TARGET_NAME = ARGS.target or "RedContainer"        # OBJECT_SPECS 의 name 중 하나
# MAX_REACH_M: 제조사 공식 reach 값이 아니다.
# 이번 센서 융합 실험에서 사용하는 "단순화된 작업 가능 반경" [m] (robot base 원점 중심의 구).
MAX_REACH_M = ARGS.max_reach if ARGS.max_reach is not None else 0.70

# ---- Scene geometry (world frame: Z-up, 단위 m) ----------------------------------
TABLE_TOP_Z = 0.40                                  # 작업대 윗면 높이 [m]
# x: -1.40 ~ 2.20, y: -1.50 ~ 1.50  -> robot base(0,0) 에서 가장 가까운 가장자리까지 1.40 m.
# Reach 반경 ~1.3 m 까지 작업대 안에 표시됨. 앞쪽 가장자리(y=-1.5)는 카메라(y=-1.75)보다 안쪽.
TABLE_CENTER_XY = (0.40, 0.0)
TABLE_SIZE_XY = (3.60, 3.00)
# 밝은 배경과 대비되는 저채도 dark charcoal. 파란 기가 강하면(hue ~110) BLUE 범위에 가까워지므로
# R,G,B 차이를 작게 둬서 채도(S)를 BLUE/GREEN/RED 검출 하한(S>=100)보다 훨씬 낮게 유지한다.
TABLE_COLOR = (0.18, 0.19, 0.21)

# ---- 배경 / 바닥 / 작업대 외형 (시각 요소일 뿐 판단 로직과 무관) ----------------------------
# 창고형 작업 현장: 콘크리트 슬래브 바닥 + 노란 안전선, 콘크리트 블록 하부벽 + 골강판(금속) 상부벽,
# 철골 기둥(노란 보호대), 개방형 지붕 트러스, 파렛트 랙/박스, 철제 작업대, 카메라 스탠드.
# 색 선택 규칙 (RGB 색 검출 보호):
#   - 대부분 저채도 회색/금속 -> 채도 S 가 검출 하한(S >= 100)보다 훨씬 낮음
#   - 안전 노랑: OpenCV hue ~23  -> RED(0~10, 170~179) / GREEN(40~85) 범위 밖
#   - 골판지/파렛트 갈색: hue ~17, S ~95 -> RED 범위 밖 + RED 채도 하한(120) 미만
#   - 빨강/파랑/초록 소품은 넣지 않음 (Isaac 창고 asset 을 쓰지 않는 이유: 색 소품 오검출, 지붕에 의한 조명 차단)
ENVIRONMENT_STYLE = "warehouse"             # "warehouse" 또는 "plain"(넓은 회색 바닥만)
FLOOR_SIZE_M = 60.0                         # 바닥(GroundPlane, collider) 한 변 [m]
FLOOR_JOINT_COLOR = (0.20, 0.20, 0.20)      # GroundPlane 색 = 슬래브 사이 줄눈(saw-cut) 색
FLOOR_PANEL_SIZE_M = 2.0                    # 콘크리트 슬래브 한 변 [m]
FLOOR_CONCRETE_COLORS = ((0.47, 0.47, 0.45), (0.50, 0.49, 0.47), (0.44, 0.44, 0.43))
ROOM_CENTER_XY = TABLE_CENTER_XY
ROOM_SIZE_XY = (16.0, 14.0)                 # 벽 안쪽 크기 [m]
ROOM_WALL_HEIGHT_M = 6.0
WALL_BLOCK_HEIGHT_M = 1.2                   # 하부 콘크리트 블록 높이
WALL_BLOCK_COLOR = (0.52, 0.52, 0.50)
WALL_CLADDING_COLOR = (0.68, 0.69, 0.70)    # 상부 골강판 (금속)
STEEL_DARK_COLOR = (0.28, 0.29, 0.31)       # 기둥 / 트러스 / 카메라 스탠드
STEEL_GALV_COLOR = (0.60, 0.61, 0.63)       # 아연도금 강재 (랙 기둥, 작업대 다리)
SAFETY_YELLOW = (0.95, 0.75, 0.05)          # 바닥 안전선, 기둥 보호대, 랙 빔, 작업대 테두리
CARDBOARD_COLOR = (0.55, 0.46, 0.34)
PALLET_WOOD_COLOR = (0.50, 0.42, 0.32)
WORKBENCH_TOP_THICKNESS_M = 0.06
# 선택: 실제 텍스처 재질(MDL)을 쓰고 싶으면 Content Browser 에서 .mdl 파일 우클릭 > Copy URL 후 붙여넣기.
# (빨강/파랑/초록 계열 재질은 색 검출을 방해하므로 회색 콘크리트/금속 계열만 사용)  빈 문자열 = 사용 안 함
MDL_OVERRIDES = {"floor": "", "walls": "", "workbench_top": ""}

# ---- Robot --------------------------------------------------------------------
ROBOT_PRIM_PATH = "/World/Franka"
# 아래 두 값은 "scene 생성 시 로봇을 배치"하는 데에만 사용한다.
# 런타임 좌표변환에서는 이 숫자를 쓰지 않고, 매 처리 frame 마다 stage 에서 robot base pose 를 다시 읽는다.
ROBOT_INITIAL_POSITION = (0.0, 0.0, TABLE_TOP_Z)
ROBOT_INITIAL_YAW_DEG = 0.0
# 화면 구성을 위한 Franka 대기 자세 (7 arm + 2 finger). 로봇 동작/IK 는 과제 범위가 아님.
ROBOT_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
# Franka asset 경로: 우선순위대로 시도한다.
#  - 환경변수 FRANKA_USD_PATH 또는 FRANKA_USD_OVERRIDE 에 로컬/Nucleus 전체 경로를 넣으면 그것을 먼저 사용
#  - Isaac Sim 5.0/5.1 asset 구조: /Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd
#  - Isaac Sim 4.x asset 구조:     /Isaac/Robots/Franka/franka.usd
FRANKA_USD_OVERRIDE = os.environ.get("FRANKA_USD_PATH", "")
FRANKA_USD_RELATIVE_CANDIDATES = [
    "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "/Isaac/Robots/Franka/franka.usd",
    "/Isaac/Robots/Franka/franka_instanceable.usd",
]

# ---- RGB-D Camera (eye-to-hand, 고정) --------------------------------------------
CAMERA_PRIM_PATH = "/World/RGBD_Camera"
# 확대된 작업대 기준으로 Franka 전체(대기 자세 상단 포함)와 A/B/C 모든 객체가 화면 안에 들어오도록 조정
CAMERA_POSITION = (1.00, -1.75, 1.45)               # world [m]
CAMERA_TARGET = (0.45, 0.05, 0.55)                  # 카메라가 바라보는 world 점 [m]
CAMERA_RESOLUTION = (640, 480)                      # (width, height) [px]
CAMERA_FREQUENCY = 20                               # [Hz] 카메라 annotator 갱신 주기
CAMERA_FOCAL_LENGTH_MM = 18.0                       # USD focalLength (수평 FOV ~60 deg)
CAMERA_HORIZONTAL_APERTURE_MM = 20.955              # USD horizontalAperture
CAMERA_CLIPPING_RANGE = (0.05, 50.0)                # [m]

# ---- Objects -------------------------------------------------------------------
# role 은 TARGET_NAME 으로부터 자동 결정된다 (TARGET_NAME 을 바꾸면 Target 이 바뀜).
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

# 시나리오별 초기 XY [m] (Z 는 작업대 위에 놓이도록 자동 계산). 시작 시 1회만 적용,
# 실행 중에는 사용자가 키(a/b/c)를 누를 때만 1회 적용. 매 frame 강제 재설정하지 않음.
SCENARIO_PRESETS = {
    # A: Target 이 반경 안 -> REACHABLE RANGE
    "A": {"RedContainer": (0.45, 0.10), "BlueBox": (0.70, -0.35), "GreenMetalSphere": (0.20, 0.45)},
    # B: Target 이 멀리 -> OUT OF RANGE
    "B": {"RedContainer": (1.00, 0.25), "BlueBox": (0.60, -0.20), "GreenMetalSphere": (0.45, 0.35)},
    # C: Distractor(BlueBox)가 카메라/로봇에 매우 가깝고 Target 은 반경 밖
    #    -> Depth-only "가장 가까운 것 = target" 가정은 틀리고, Fusion 은 OUT OF RANGE
    "C": {"RedContainer": (0.95, 0.30), "BlueBox": (0.30, -0.35), "GreenMetalSphere": (0.75, 0.45)},
}
if SCENARIO not in SCENARIO_PRESETS:
    SCENARIO = "A"
OBJECT_INITIAL_POSITIONS = SCENARIO_PRESETS[SCENARIO]

# ---- RGB detection (OpenCV HSV 스케일: H 0-179, S 0-255, V 0-255) ---------------
COLOR_RANGES = {
    "RED":   [((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (179, 255, 255))],  # 빨강은 hue 가 0/180 에서 wrap
    "BLUE":  [((100, 120, 50), (130, 255, 255))],
    "GREEN": [((40, 100, 50), (85, 255, 255))],
}
MIN_MASK_PIXELS = 60            # 이보다 작은 mask 는 DETECTED = NO (노이즈/거의 가려짐)

# ---- Depth ---------------------------------------------------------------------
MIN_VALID_DEPTH_PIXELS = 20     # mask 내부 valid depth 가 이보다 적으면 INVALID DEPTH
DEPTH_MAX_VALID_M = 20.0        # 이보다 먼 값은 배경/far-plane 으로 보고 제외
DEPTH_ERODE_MASK = True         # 물체 경계(배경 depth 섞임) 제외를 위해 mask 를 1px 침식 후 사용

# ---- 3D unprojection -----------------------------------------------------------
USE_OFFICIAL_UNPROJECTION = True    # Camera.get_world_points_from_image_coords() 우선 사용
UNPROJECTION_CHECK_TOL_M = 0.01     # 공식 helper 와 직접 pinhole 계산의 차이 허용치 (불일치 시 경고)

# ---- Depth-only baseline (비교용) ------------------------------------------------
# Depth 만으로 "작업대 위에 솟아있는 가장 가까운 표면"을 찾는다. (RGB 사용 안 함 -> identity 모름)
DEPTH_ONLY_STRIDE = 2                       # 연산량 절감을 위한 pixel subsampling
DEPTH_ONLY_HEIGHT_BAND = (0.015, 0.25)      # 작업대 윗면 기준 높이 범위 [m]
DEPTH_ONLY_ROBOT_EXCLUSION_RADIUS = 0.20    # robot base 주변(XY) 로봇 자기 몸체 제외 반경 [m]

# ---- 출력 / 시각화 / 저장 ----------------------------------------------------------
SHOW_FUSION_MONITOR = not ARGS.no_monitor  # Isaac Sim 내부 omni.ui "RGB-D Fusion Monitor" 패널 (기본)
MONITOR_WIDTH = 380                         # [px]
MONITOR_HEIGHT = 840
MONITOR_DOCK_RIGHT_OF_VIEWPORT = True       # True: Viewport 오른쪽에 dock 시도 (실패 시 floating 유지)
MONITOR_DOCK_RATIO = 0.27                   # dock 시 패널이 차지할 비율 (레이아웃이 어색하면 False 로)
# ---- Reach 시각화 (isaacsim.util.debug_draw, 물리/USD 에 영향 없는 시각화 전용) ----
REACH_VIZ_ENABLED = True
REACH_VIZ_TABLE_OFFSET_M = 0.005            # 작업대 표면 원을 표면보다 살짝 위에 그려 z-fighting 방지
# 색은 (r, g, b, a) 0~1. boundary 는 저채도 light blue 로 두어(S < 100/255) RGB 검출의
# RED/GREEN/BLUE HSV 범위와 겹치지 않게 했다 (debug draw 가 카메라 영상에 찍히는 경우 대비).
REACH_VIZ_BOUNDARY_COLOR = (0.65, 0.85, 0.95, 0.55)
REACH_VIZ_TABLE_RING_COLOR = (0.70, 0.88, 0.97, 0.85)
REACH_VIZ_BASE_COLOR = (1.0, 1.0, 1.0, 1.0)
REACH_VIZ_IN_COLOR = (0.15, 0.90, 0.25, 1.0)
REACH_VIZ_OUT_COLOR = (1.00, 0.20, 0.20, 1.0)
REACH_VIZ_BOUNDARY_WIDTH = 1.5             # [px]
REACH_VIZ_TABLE_RING_WIDTH = 2.5
REACH_VIZ_TARGET_LINE_WIDTH = 2.0          # 얇게: 카메라 영상에 찍혀도 3x3 morphology open 으로 제거되는 두께
REACH_VIZ_TARGET_MARKER_SIZE = 8.0         # [px]
REACH_VIZ_LATITUDES_DEG = (30.0, 60.0)     # 상부 hemisphere 위도 ring
SHOW_OVERLAY = bool(ARGS.cv_windows)        # OpenCV annotated RGB 창 (선택 fallback, 기본 끔)
SHOW_DEPTH_WINDOW = bool(ARGS.cv_windows) and not ARGS.no_depth_window  # OpenCV depth 창 (보조)
SHOW_GT_VALIDATION = True                   # GT 기반 검증 출력 (판단에는 사용하지 않음)
PRINT_INTERVAL_FRAMES = 15                  # 콘솔 표 출력 주기 [simulation step]
WARMUP_FRAMES = 20                          # 시작 직후 센서 버퍼가 비어있는 구간 skip
SAVE_RESULTS = bool(ARGS.save)              # True 면 SAVE_INTERVAL_FRAMES 마다 output/ 에 저장
SAVE_INTERVAL_FRAMES = 300
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPTH_VIS_RANGE_M = (1.0, 3.5)              # depth 컬러맵 표시 범위 [m]

# ---- Simulation ----------------------------------------------------------------
PHYSICS_DT = 1.0 / 60.0
RENDERING_DT = 1.0 / 60.0
DOME_LIGHT_INTENSITY = 700.0
KEY_LIGHT_INTENSITY = 1500.0

# =============================================================================
# 2. SimulationApp 시작 (다른 omni / isaacsim import 보다 반드시 먼저)
# =============================================================================
from isaacsim import SimulationApp  # noqa: E402

# window_width/height = Isaac Sim 앱 창 크기, width/height = viewport 렌더 해상도
simulation_app = SimulationApp({"headless": False, "window_width": 1600, "window_height": 900,
                                "width": 1280, "height": 720})

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # noqa: E402


def log(msg, level="INFO"):
    print(f"[SensorFusion][{level}] {msg}", flush=True)


class RateLimitedLog:
    """같은 종류의 경고가 매 frame 콘솔을 도배하지 않도록 key 별로 간격을 둔다."""

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
    enable_extension("isaacsim.sensors.camera")      # 기본 experience 에 포함되어 있지만 명시적으로 활성화
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

# ---- 선택 모듈 (없으면 해당 기능만 비활성화) ------------------------------------------
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

# get_assets_root_path: 5.x 공식 위치는 isaacsim.storage.native
_get_assets_root_path = None
try:
    from isaacsim.storage.native import get_assets_root_path as _get_assets_root_path
except ImportError:
    try:  # 구버전(4.x) 위치. 5.x 에서는 deprecated/없을 수 있음
        from isaacsim.core.utils.nucleus import get_assets_root_path as _get_assets_root_path
    except ImportError:
        _get_assets_root_path = None

# ---- debug_draw (Viewport reach 시각화, 선택) -------------------------------------------
# Isaac Sim 5.x 공식 extension: isaacsim.util.debug_draw
#   draw = _debug_draw.acquire_debug_draw_interface()
#   draw.draw_lines(starts, ends, colors, widths) / draw.draw_points(points, colors, sizes)
#   draw.clear_lines() / draw.clear_points() / _debug_draw.release_debug_draw_interface(draw)
_debug_draw = None
try:
    from isaacsim.core.utils.extensions import enable_extension as _enable_ext
    _enable_ext("isaacsim.util.debug_draw")
    from isaacsim.util.debug_draw import _debug_draw
except Exception as _e:  # noqa: BLE001
    _debug_draw = None
    log(f"isaacsim.util.debug_draw not available ({_e}); reach visualization disabled.", "WARN")

# ---- omni.ui (Isaac Sim 내부 패널, 선택) ---------------------------------------------
try:
    import omni.ui as ui
except Exception as _e:  # noqa: BLE001
    ui = None
    log(f"omni.ui not available ({_e}); Fusion Monitor panel disabled (console only).", "WARN")

# ---- OpenCV (선택) -------------------------------------------------------------
try:
    import cv2
except Exception as _e:  # noqa: BLE001
    cv2 = None
    log(f"OpenCV not available ({_e}). Detection falls back to numpy HSV; overlay windows/PNG saving disabled. "
        "Install: python.bat -m pip install opencv-python==4.10.0.84", "WARN")


# =============================================================================
# 3. 수학 helper (quaternion 은 Isaac Sim 규약과 같은 scalar-first (w, x, y, z))
# =============================================================================
def quat_wxyz_to_rotmat(q):
    """(w,x,y,z) quaternion -> 3x3 회전행렬 R.  p_parent = R @ p_child."""
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
    """3x3 회전행렬 -> (w,x,y,z) quaternion (Shepperd 방법)."""
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


def yaw_to_quat_wxyz(yaw_deg):
    h = math.radians(yaw_deg) / 2.0
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def look_at_quat_usd(eye, target, world_up=(0.0, 0.0, 1.0)):
    """USD 카메라 규약(로컬 -Z 가 시선, +Y 가 화면 위, +X 가 화면 오른쪽)으로
    eye 에서 target 을 바라보는 world orientation (w,x,y,z) 을 계산한다."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - eye
    if np.linalg.norm(f) < 1e-9:
        raise ValueError("CAMERA_POSITION and CAMERA_TARGET must be different points")
    f /= np.linalg.norm(f)
    up = np.asarray(world_up, dtype=float)
    if abs(float(np.dot(f, up))) > 0.999:          # 수직으로 내려다보는 특이 경우
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.column_stack([r, u, -f])                # 열 = 카메라 로컬 x, y, z 축의 world 방향
    return rotmat_to_quat_wxyz(R)


def fmt(v, nd=2, width=0):
    """None/NaN 이면 'N/A'."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        s = "N/A"
    else:
        s = f"{v:.{nd}f}"
    return s.rjust(width) if width else s


def fmt_vec(v, nd=2):
    if v is None:
        return "N/A"
    return "(" + ", ".join(f"{float(c):+.{nd}f}" for c in v) + ")"


def object_role(name):
    return "TARGET" if name == TARGET_NAME else "NON_TARGET"


def object_center_z(spec):
    """작업대 위에 놓였을 때의 물체 중심 높이."""
    if spec["shape"] == "cuboid":
        half = spec["size"][2] / 2.0
    elif spec["shape"] == "cylinder":
        half = spec["height"] / 2.0
    else:
        half = spec["radius"]
    return TABLE_TOP_Z + half


# =============================================================================
# 4. Scene 구성
# =============================================================================
def get_assets_root_safe():
    """Isaac Sim asset root(Nucleus 또는 S3) 를 얻는다. 실패하면 None (프로그램은 계속)."""
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
    """reference 가 실제로 resolve 되었는지(자식 prim 이 생겼는지) 확인."""
    try:
        return len(list(prim.GetFilteredChildren(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))) > 0
    except Exception:  # noqa: BLE001
        return len(prim.GetChildren()) > 0


def load_franka_reference(stage, assets_root):
    """Franka USD 를 후보 경로 순서대로 reference 해보고 성공한 경로를 반환. 모두 실패하면 None."""
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
    """Dome light(균일 환경광) + Distant light(주광). 색 검출이 안정적이도록 과노출을 피하는 세기."""
    UsdGeom.Xform.Define(stage, "/World/Lights")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(DOME_LIGHT_INTENSITY)
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key.CreateIntensityAttr(KEY_LIGHT_INTENSITY)
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))


def create_object(world, spec, xy):
    """OBJECT_SPECS 한 항목으로 정적(Fixed) primitive 를 만든다.
    Fixed* 를 쓰는 이유: 중력으로 굴러가지 않고, UI 에서 gizmo 로 옮기면 그 자리에 유지됨."""
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
    """여러 prim 이 함께 쓰는 PreviewSurface 재질 (슬래브/리브 수십 개가 재질 수십 개를 만들지 않도록)."""
    if PreviewSurface is None:
        return None
    try:
        return PreviewSurface(prim_path=prim_path, color=np.array(color, dtype=float),
                              roughness=float(roughness), metallic=float(metallic))
    except Exception as e:  # noqa: BLE001
        log(f"PreviewSurface {prim_path} failed ({e}); using plain color.", "WARN")
        return None


def _visual_box(prim_path, center, size, material, color, cast_shadows=True):
    """충돌/물리 없는 시각용 박스 (배경 요소). cast_shadows=False 면 RTX 그림자를 만들지 않음."""
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
    """창고형 작업 현장 배경을 만들고 MDL override 용 prim 경로 그룹을 반환한다. (모두 VisualCuboid, 물리 없음)"""
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

    # (1) 콘크리트 슬래브 바닥: 2 m 패널 3가지 톤, 8 mm 줄눈(GroundPlane 색)
    p = FLOOR_PANEL_SIZE_M
    for i in range(int(round(rx / p))):
        for j in range(int(round(ry / p))):
            k = (i * 7 + j * 3) % len(FLOOR_CONCRETE_COLORS)
            groups["floor"].append(_visual_box(
                f"{E}/Floor/Slab_{i:02d}_{j:02d}", (x0 + (i + 0.5) * p, y0 + (j + 0.5) * p, 0.002),
                (p - 0.008, p - 0.008, 0.004), m_conc[k], FLOOR_CONCRETE_COLORS[k]))

    # (2) 노란 안전선: 로봇 작업 셀 경계(작업대 + 0.6 m) + 남쪽 보행 통로 2줄
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

    # (3) 벽: 하부 콘크리트 블록(1.2 m) + 상부 골강판 + 0.6 m 간격 세로 리브. 안쪽 면 = 방 경계
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

    # (4) 철골 기둥 + 노란 보호대 (벽 안쪽)
    cols = [(min(max(x0 + 4.0 * k, x0 + 0.25), x1 - 0.25), y1 - 0.25) for k in range(5)]
    cols += [(min(max(x0 + 4.0 * k, x0 + 0.25), x1 - 0.25), y0 + 0.25) for k in range(5)]
    cols += [(x1 - 0.25, y0 + 3.5 * k) for k in range(1, 4)] + [(x0 + 0.25, y0 + 3.5 * k) for k in range(1, 4)]
    for n, (px, py) in enumerate(cols):
        _visual_box(f"{E}/Columns/Column_{n:02d}", (px, py, H / 2), (0.30, 0.30, H), m_dark, STEEL_DARK_COLOR)
        _visual_box(f"{E}/Columns/Guard_{n:02d}", (px, py, 0.5), (0.38, 0.38, 1.0), m_yellow, SAFETY_YELLOW)

    # (5) 개방형 지붕 트러스 (지붕 패널 없음 -> 조명 유지, 그림자도 만들지 않음)
    for n, (px, _) in enumerate(cols[:5]):
        _visual_box(f"{E}/Roof/Truss_{n}", (px, cy, H - 0.2), (0.25, ry, 0.40), m_dark, STEEL_DARK_COLOR,
                    cast_shadows=False)
    for n, py in enumerate((cy - ry / 4, cy + ry / 4)):
        _visual_box(f"{E}/Roof/Purlin_{n}", (cx, py, H - 0.05), (rx, 0.15, 0.20), m_dark, STEEL_DARK_COLOR,
                    cast_shadows=False)

    # (6) 파렛트 랙 (북쪽 벽): 4 bay x 3 단, 아연도금 기둥 + 노란 빔 + 나무 파렛트 + 골판지 박스
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

    # (7) 바닥에 쌓인 파렛트 (동쪽 벽 근처)
    for n, py in enumerate((cy - 3.0, cy + 3.0)):
        px = x1 - 1.4
        _visual_box(f"{E}/Racks/FloorPallet_{n}", (px, py, 0.07), (1.2, 1.0, 0.14), m_wood, PALLET_WOOD_COLOR)
        _visual_box(f"{E}/Racks/FloorBox_{n}", (px, py, 0.14 + 0.45), (1.1, 0.9, 0.9), m_card, CARDBOARD_COLOR)

    # (8) 고정 카메라 스탠드 (카메라 바로 아래 기둥. 카메라 시야(아래쪽 최대 ~50도) 밖이라 영상에 찍히지 않음)
    cam_x, cam_y, cam_z = CAMERA_POSITION
    _visual_box(f"{E}/CameraStand/BasePlate", (cam_x, cam_y, 0.01), (0.45, 0.45, 0.02), m_dark, STEEL_DARK_COLOR)
    pole_h = cam_z - 0.12 - 0.02
    _visual_box(f"{E}/CameraStand/Pole", (cam_x, cam_y, 0.02 + pole_h / 2), (0.06, 0.06, pole_h), m_dark,
                STEEL_DARK_COLOR)
    return groups


def setup_environment(world, stage):
    """넓은 바닥(GroundPlane, collider) + 창고형 작업 현장 배경.
    배경은 VisualCuboid(물리 없음)이며 센서 판단 로직과 무관하다. 장식 생성이 실패해도 바닥만으로 계속 진행.
    반환: MDL override 용 prim 경로 그룹 {"floor": [...], "walls": [...]}"""
    # (1) 넓은 바닥: asset 다운로드가 필요 없는 로컬 GroundPlane (물리 바닥 역할)
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
    """철제 작업대: 상판(FixedCuboid, collider, 윗면 = TABLE_TOP_Z) + 다리/하부 프레임 + 노란 테두리 띠.
    판단에 쓰이는 것은 TABLE_TOP_Z / TABLE_CENTER_XY / TABLE_SIZE_XY 상수뿐이며 외형은 영향 없음.
    반환: 상판 prim 경로 (MDL override 용)"""
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
        # 상판 옆면의 노란 테두리 띠 (윗면보다 1 cm 아래까지만 -> 작업 표면 높이에 영향 없음)
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
    """MDL_OVERRIDES 에 URL 이 지정된 그룹에만 실제 텍스처 재질(MDL)을 입힌다 (기본: 사용 안 함).
    Kit 명령 CreateMdlMaterialPrim / BindMaterial 사용. 실패하면 PreviewSurface 외형을 그대로 유지."""
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
    """Ground / 배경 / 작업대 / Light / Franka / 물체 3개 를 생성하고 handle 들을 dict 로 반환."""
    stage = get_current_stage()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    UsdGeom.Xform.Define(stage, "/World/Looks")
    scene = {"objects": {}, "articulation": None, "robot_xform": None,
             "robot_usd": None, "robot_is_placeholder": False}

    # (1) 넓은 바닥 + 창고형 배경
    groups = setup_environment(world, stage)

    # (2) 작업대 (윗면 높이 TABLE_TOP_Z)
    groups["workbench_top"] = [setup_workbench(world, stage)]
    apply_mdl_overrides(groups)

    # (3) Lights
    setup_lights(stage)

    # (4) Robot (Franka). 실패 시 placeholder base 로 계속 진행 (좌표변환 파이프라인 확인 가능)
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
    # robot base frame = ROBOT_PRIM_PATH prim 의 frame (Franka 의 panda_link0 가 이 원점에 위치)
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

    # (5) Objects
    for spec in OBJECT_SPECS:
        scene["objects"][spec["name"]] = create_object(world, spec, OBJECT_INITIAL_POSITIONS[spec["name"]])
    return scene


def apply_robot_home(scene):
    """world.reset() 이후 Franka 를 대기 자세로 둔다 (화면 구성용, 판단과 무관)."""
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
    """시나리오 preset 을 '한 번만' 적용 (사용자가 키를 눌렀을 때). 매 frame 호출하지 않는다."""
    preset = SCENARIO_PRESETS[name]
    for spec in OBJECT_SPECS:
        obj = scene["objects"][spec["name"]]
        xy = preset[spec["name"]]
        pos = np.array([xy[0], xy[1], object_center_z(spec)], dtype=float)
        obj.set_world_pose(position=pos)
        try:
            obj.set_default_state(position=pos)     # Stop->Play(reset) 시에도 이 배치를 유지
        except Exception:  # noqa: BLE001
            pass
    log(f"Scenario {name} applied once. Objects can now be moved freely in the UI.")


def validate_scene_layout(camera, scene):
    """시작 시 1회 실행하는 배치 점검 (경고만 출력, 판단 로직과 무관).
    - A/B/C preset 의 모든 객체 footprint 가 작업대 위에 있는지
    - 각 객체 중심과 Franka(base / 대기 자세 상단 근사점)가 카메라 화면 안으로 투영되는지"""
    K, _ = get_intrinsics(camera)
    t_wc, R_wc, _ = get_camera_pose_ros(camera)
    t_wr, _ = get_robot_base_pose(scene)
    w, h = CAMERA_RESOLUTION
    tx0, tx1 = TABLE_CENTER_XY[0] - TABLE_SIZE_XY[0] / 2, TABLE_CENTER_XY[0] + TABLE_SIZE_XY[0] / 2
    ty0, ty1 = TABLE_CENTER_XY[1] - TABLE_SIZE_XY[1] / 2, TABLE_CENTER_XY[1] + TABLE_SIZE_XY[1] / 2

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
# 5. Camera
# =============================================================================
def setup_camera():
    """외부 고정형 RGB-D 카메라 (eye-to-hand).
    - pose 는 여기서 '한 번만' 설정하며 매 frame 다시 설정하지 않는다.
    - intrinsics 가 일관되도록 USD focalLength/aperture 를 명시적으로 설정한다
      (verticalAperture = horizontalAperture * H / W  -> 정사각 pixel, fx == fy)."""
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

    # look-at 회전을 USD 카메라 축 규약으로 계산해 그대로 넣는다 (camera_axes="usd": 변환 없음)
    q_usd = look_at_quat_usd(CAMERA_POSITION, CAMERA_TARGET)
    camera.set_world_pose(position=np.array(CAMERA_POSITION, dtype=float), orientation=q_usd, camera_axes="usd")
    return camera


def get_intrinsics(camera):
    """K = [[fx,0,cx],[0,fy,cy],[0,0,1]] [px].  공식 API 우선, 실패 시 USD 속성에서 계산."""
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
    """카메라 pose 를 ROS optical frame 규약으로 반환: p_world = R_wc @ p_cam + t_wc.
    ROS optical frame: x = 화면 오른쪽, y = 화면 아래, z = 광축(앞) -> depth(Z) 와 같은 축."""
    try:
        pos, quat = camera.get_world_pose(camera_axes="ros")
        return np.asarray(pos, dtype=float).reshape(3), quat_wxyz_to_rotmat(quat), \
            "Camera.get_world_pose(camera_axes='ros')"
    except Exception as e:  # noqa: BLE001
        RL.warn("campose", f"Camera.get_world_pose(camera_axes='ros') failed: {e}; using USD transform")
    prim = get_current_stage().GetPrimAtPath(CAMERA_PRIM_PATH)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=float)
    # Gf 행렬은 row-vector 규약 (p_world_row = p_local_row @ M) -> R = M[:3,:3]^T, t = M[3,:3]
    R_usd = M[:3, :3].T
    R_usd = R_usd / np.linalg.norm(R_usd, axis=0, keepdims=True)
    R_ros = R_usd @ np.diag([1.0, -1.0, -1.0])     # USD cam(x right, y up, z back) -> ROS(x right, y down, z fwd)
    return M[3, :3].copy(), R_ros, "USD ComputeLocalToWorldTransform (fallback)"


def get_robot_base_pose(scene):
    """T_world_robot (robot base frame -> world) 를 stage 에서 현재 값으로 읽는다.
    반환 (t_wr, R_wr): p_world = R_wr @ p_robot + t_wr."""
    pos, quat = scene["robot_xform"].get_world_pose()
    return np.asarray(pos, dtype=float).reshape(3), quat_wxyz_to_rotmat(quat)


# ★★★----------------------------------------------------------------------
# ★ [CORE-1] 동기화: RGB(t) 와 Depth(t) 를 같은 카메라 / 같은 frame 에서 획득
# ★★★----------------------------------------------------------------------
def acquire_rgbd_frame(camera):
    """같은 카메라 render product 의 RGB 와 Depth 를 '같은 frame' 으로 함께 가져온다.

    1순위: camera.get_current_frame()  -> 하나의 dict 안에 'rgba' 와 'distance_to_image_plane' 이
           같은 data-acquisition 시점(rendering_frame / rendering_time)에 함께 저장된다.
    2순위: get_rgba()/get_depth() 개별 getter (같은 world.step() 직후 호출이므로 같은 render 결과).

    Depth 의미: distance_to_image_plane = 카메라 광축 방향 Z depth [m] (image plane 까지의 수직거리).
                distance_to_camera (광학 중심까지의 Euclidean range) 가 아님!
    반환: dict 또는 None(아직 버퍼가 비어 있음)."""
    rgba = depth = None
    rframe = rtime = None
    source = "camera.get_current_frame()"
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
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape[:2] != rgba.shape[:2]:
        RL.warn("shape", f"RGB {rgba.shape} and depth {depth.shape} sizes differ; skipping frame")
        return None
    rgb = rgba[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * (255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return {"rgb": np.array(rgb, copy=True), "depth": np.array(depth, copy=True),
            "rendering_frame": rframe, "rendering_time": rtime, "source": source}


# =============================================================================
# 6. RGB 처리: 색상 기반 객체 검출 (WHAT)
# =============================================================================
def rgb_to_hsv_cv_scale(rgb):
    """RGB uint8 -> HSV (OpenCV 스케일 H 0-179, S/V 0-255). OpenCV 없으면 numpy 로 동일 계산."""
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
    """작은 노이즈 제거 후 가장 큰 연결 영역만 남긴다 (OpenCV 없으면 mask 그대로)."""
    if cv2 is None or not mask.any():
        return mask
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx


# ★★★----------------------------------------------------------------------
# ★ [CORE-2] RGB 센서 역할 (WHAT): 색으로 객체 identity / TARGET 여부 / mask 결정
# ★★★----------------------------------------------------------------------
def detect_objects_rgb(rgb, object_specs):
    """RGB 영상에서 각 객체를 HSV threshold 로 검출한다.
    반환: {name: {name, role, visual, detected, pixel_count, mask, bbox(x0,y0,x1,y1), centroid(u,v), det_reason}}
    pixel 좌표: u = 열(오른쪽 +), v = 행(아래 +), 원점 = 좌상단. centroid 는 pixel 중심(+0.5) 기준.
    ※ 나중에 YOLO 등으로 교체하려면 이 함수만 같은 반환 형식으로 바꾸면 된다."""
    hsv = rgb_to_hsv_cv_scale(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = {}
    for spec in object_specs:
        name = spec["name"]
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
            det.update(detected=True, mask=mask, det_reason="ok",
                       bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                       centroid=(float(xs.mean()) + 0.5, float(ys.mean()) + 0.5))
        out[name] = det
    return out


# =============================================================================
# 7. Depth 처리: RGB mask 가 고른 pixel 의 depth 만 사용 (WHERE)
# =============================================================================
# ★★★----------------------------------------------------------------------
# ★ [CORE-3] 센서 융합 지점: RGB mask 가 어떤 Depth pixel 을 쓸지 결정 (전체 최소 depth 사용 X)
# ★★★----------------------------------------------------------------------
def extract_object_depth(depth, mask):
    """객체 mask 내부의 valid depth (finite, >0, <DEPTH_MAX_VALID_M) 의 median 을 구한다.
    bbox 전체가 아니라 mask 내부만 쓰므로 배경 depth 가 섞이지 않는다.
    반환: {depth_valid, depth_median[m], depth_valid_count, depth_reason}"""
    res = {"depth_valid": False, "depth_median": None, "depth_valid_count": 0, "depth_reason": ""}
    if mask is None:
        res["depth_reason"] = "no RGB mask"
        return res
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
# 8. (u, v, Z) -> 3D : camera frame / world frame
# =============================================================================
# ★★★----------------------------------------------------------------------
# ★ [CORE-4] Depth 센서 역할 (WHERE): (u, v, Z) -> camera frame -> world 3D 위치
# ★★★----------------------------------------------------------------------
def pixel_depth_to_3d(camera, u, v, z, K, cam_pose):
    """pixel (u,v) 와 Z depth(distance_to_image_plane) 로 3D 점을 계산한다.

    camera frame = ROS optical (x 오른쪽, y 아래, z 광축 앞). distance_to_image_plane 은 정확히 이 z 값이므로
        X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy,  Z = depth
    가 성립한다 (Euclidean range 인 distance_to_camera 였다면 이 식은 틀림).

    world 점은 공식 helper Camera.get_world_points_from_image_coords() 를 우선 사용하고,
    직접 계산값 (R_wc @ p_cam + t_wc) 과 비교해 좌표 규약이 일치하는지 매번 검증한다."""
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
# 9. World -> Robot base frame
# =============================================================================
# ★★★----------------------------------------------------------------------
# ★ [CORE-5] 좌표변환: world -> robot base (Camera depth 를 robot 거리로 그대로 쓰지 않음)
# ★★★----------------------------------------------------------------------
def world_to_robot_base(p_world, t_wr, R_wr):
    """T_world_robot = [R_wr | t_wr] 은 robot 좌표 -> world 좌표 변환 (p_w = R_wr p_r + t_wr).
    우리가 필요한 것은 역변환 T_robot_world:  p_r = R_wr^T (p_w - t_wr)."""
    return R_wr.T @ (np.asarray(p_world, dtype=float) - t_wr)


# =============================================================================
# 10. Reachability 판단 (Target 기준 FINAL)
# =============================================================================
# ★★★----------------------------------------------------------------------
# ★ [CORE-6] 최종 판단: 3D Euclidean distance <= MAX_REACH_M -> Target 기준 FINAL
# ★★★----------------------------------------------------------------------
def evaluate_reachability(results, max_reach):
    """각 객체에 IN / OUT / N/A 를 붙이고, FINAL 은 Target 객체만으로 결정한다."""
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
# 11. 단일 센서 비교 (RGB only / Depth only)
# =============================================================================
# ★★★----------------------------------------------------------------------
# ★ [CORE-8] 단일 센서 비교: Depth-only 는 거리만 알고 identity 를 모름
# ★★★----------------------------------------------------------------------
def depth_only_baseline(depth, K, cam_pose, robot_pose, max_reach):
    """Depth 만 사용: 작업대 위로 솟은 표면 중 카메라에 가장 가까운 점을 찾는다.
    (작업대 높이/범위, robot base 위치는 scene 보정 정보이며 RGB 는 전혀 사용하지 않음.)
    -> 거리는 알 수 있지만 그것이 어떤 물체(Target?)인지는 알 수 없다."""
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
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    pw = pc @ R_wc.T + t_wc
    tx0, tx1 = TABLE_CENTER_XY[0] - TABLE_SIZE_XY[0] / 2, TABLE_CENTER_XY[0] + TABLE_SIZE_XY[0] / 2
    ty0, ty1 = TABLE_CENTER_XY[1] - TABLE_SIZE_XY[1] / 2, TABLE_CENTER_XY[1] + TABLE_SIZE_XY[1] / 2
    m = ((pw[:, 2] > TABLE_TOP_Z + DEPTH_ONLY_HEIGHT_BAND[0]) & (pw[:, 2] < TABLE_TOP_Z + DEPTH_ONLY_HEIGHT_BAND[1])
         & (pw[:, 0] > tx0) & (pw[:, 0] < tx1) & (pw[:, 1] > ty0) & (pw[:, 1] < ty1)
         & (np.hypot(pw[:, 0] - t_wr[0], pw[:, 1] - t_wr[1]) > DEPTH_ONLY_ROBOT_EXCLUSION_RADIUS))
    if not m.any():
        return out
    idx = np.flatnonzero(m)[int(np.argmin(z[m]))]
    p_r = world_to_robot_base(pw[idx], t_wr, R_wr)
    dist = float(np.linalg.norm(p_r))
    out.update(available=True, nearest_depth=float(z[idx]), pixel=(float(u[idx]), float(v[idx])),
               robot_dist=dist, naive_status="REACHABLE RANGE" if dist <= max_reach else "OUT OF RANGE")
    return out


def compute_single_sensor_baselines(results, depth, K, cam_pose, robot_pose, max_reach, final):
    target = next((r for r in results if r["role"] == "TARGET"), None)
    rgb_only = {"target_detected": bool(target and target["detected"]),
                "centroid": target["centroid"] if target else None,
                "distance": None, "reachability": "UNKNOWN"}
    d_only = depth_only_baseline(depth, K, cam_pose, robot_pose, max_reach)
    # 설명용: depth-only 가 고른 pixel 이 실제로 무엇인지는 RGB mask 가 있어야만 알 수 있음
    d_only["actually_is"] = "unknown surface"
    if d_only["available"]:
        pu, pv = int(d_only["pixel"][0]), int(d_only["pixel"][1])
        for r in results:
            if r["mask"] is not None and r["mask"][pv, pu]:
                d_only["actually_is"] = r["name"]
                break
    agree = None
    if d_only["available"] and final["status"] in ("REACHABLE RANGE", "OUT OF RANGE"):
        agree = d_only["naive_status"] == final["status"]
    return {"rgb_only": rgb_only, "depth_only": d_only, "naive_agrees_with_fusion": agree}


def attach_ground_truth_validation(results, scene, robot_pose):
    """[VALIDATION ONLY] stage 의 GT 물체 중심 위치로 거리를 계산해 오차를 보여준다.
    이 값은 range_status / FINAL 판단에 절대 사용하지 않는다.
    (센서 추정점은 '보이는 표면'이므로 물체 중심 대비 수 cm 차이가 정상)"""
    t_wr, R_wr = robot_pose
    for r in results:
        r["gt_robot_dist"] = None
        try:
            pos, _ = scene["objects"][r["name"]].get_world_pose()
            r["gt_robot_dist"] = float(np.linalg.norm(world_to_robot_base(pos, t_wr, R_wr)))
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# 12. 한 frame 처리 (Fusion 핵심)
# =============================================================================
# ★★★----------------------------------------------------------------------
# ★ [CORE-7] Fusion 파이프라인: CORE-2 -> 3 -> 4 -> 5 -> 6 을 한 frame 에 연결
# ★★★----------------------------------------------------------------------
def process_rgbd(camera, frame, scene, max_reach):
    """RGB(t) 와 Depth(t) (같은 frame dict) 를 융합해 객체별 결과와 FINAL 을 만든다."""
    rgb, depth = frame["rgb"], frame["depth"]
    K, k_src = get_intrinsics(camera)
    cam_pose = get_camera_pose_ros(camera)
    robot_pose = get_robot_base_pose(scene)          # 매 처리 시 stage 에서 현재 robot base pose 를 읽음

    detections = detect_objects_rgb(rgb, OBJECT_SPECS)                   # RGB -> WHAT
    results = []
    for spec in OBJECT_SPECS:
        r = dict(detections[spec["name"]])
        r.update({"p_cam": None, "p_world": None, "p_robot": None, "robot_dist": None,
                  "unproj_method": None, "unproj_check_m": None})
        r.update(extract_object_depth(depth, r["mask"]))                 # RGB mask 가 Depth pixel 을 선택
        if r["detected"] and r["depth_valid"]:
            u, v = r["centroid"]
            r.update(pixel_depth_to_3d(camera, u, v, r["depth_median"], K, cam_pose))   # -> camera / world
            r["p_robot"] = world_to_robot_base(r["p_world"], *robot_pose)               # -> robot base
            r["robot_dist"] = float(np.linalg.norm(r["p_robot"]))
        results.append(r)

    final = evaluate_reachability(results, max_reach)
    baselines = compute_single_sensor_baselines(results, depth, K, cam_pose, robot_pose, max_reach, final)
    if SHOW_GT_VALIDATION:
        attach_ground_truth_validation(results, scene, robot_pose)
    return {"results": results, "final": final, "baselines": baselines,
            "K": K, "K_source": k_src, "cam_pose": cam_pose, "robot_pose": robot_pose}


# =============================================================================
# 13. 콘솔 출력
# =============================================================================
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
    """센서 의미 / 단위 / 좌표계 / intrinsics / pose 를 시작 시 1회 출력."""
    K, k_src = get_intrinsics(camera)
    t_wc, R_wc, pose_src = get_camera_pose_ros(camera)
    t_wr, R_wr = get_robot_base_pose(scene)
    f_expected = np.asarray(CAMERA_TARGET, float) - np.asarray(CAMERA_POSITION, float)
    f_expected /= np.linalg.norm(f_expected)
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
# 14. Overlay / Depth 시각화
# =============================================================================
STATUS_COLORS_BGR = {"REACHABLE RANGE": (0, 200, 0), "OUT OF RANGE": (0, 0, 255)}


def _put(img, text, org, color=(255, 255, 255), scale=0.45, thick=1, bg=(0, 0, 0)):
    """배경 박스가 있는 텍스트 (가독성)."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(rgb, bundle, meta):
    """annotated RGB (최종 결과 화면). 객체별 bbox/contour, 이름, 역할, depth, robot 거리, IN/OUT,
    상단에 TARGET / MAX REACH / TARGET DIST / FINAL, 하단에 RGB-only / Depth-only / Fusion 비교."""
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
        y = y0 - 8 - 16 * (len(lines) - 1)
        if y < 14:
            y = y1 + 18
        for text, c in lines:
            _put(img, text, (max(0, x0), y), c)
            y += 16
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
    """Depth 보조 시각화 (최종 결과 화면 아님). 가까울수록 밝게, invalid 는 검정."""
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
    """OpenCV 창 관리. GUI 오류가 나면 창만 끄고 simulation / 콘솔 출력은 계속한다."""

    def __init__(self):
        self.enabled = cv2 is not None and (SHOW_OVERLAY or SHOW_DEPTH_WINDOW)
        if (SHOW_OVERLAY or SHOW_DEPTH_WINDOW) and cv2 is None:
            log("Overlay requested but OpenCV is missing -> console only.", "WARN")

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
# 14-B. [VIZ] Viewport Reach 시각화 (isaacsim.util.debug_draw) - 시각화 전용
# =============================================================================
def _ring_segments(center, u_axis, v_axis, radius, n_seg, a0=0.0, a1=2.0 * math.pi):
    """center 를 중심으로 (u_axis, v_axis) 평면의 원/호를 n_seg 개 선분으로 나눈 (starts, ends)."""
    ang = np.linspace(a0, a1, n_seg + 1)
    pts = (np.asarray(center, dtype=float)[None, :]
           + radius * (np.cos(ang)[:, None] * np.asarray(u_axis, dtype=float)[None, :]
                       + np.sin(ang)[:, None] * np.asarray(v_axis, dtype=float)[None, :]))
    return pts[:-1], pts[1:]


class ReachVisualizer:
    """MAX_REACH_M 작업 반경과 Target 연결선을 Viewport 에 그린다.

    - isaacsim.util.debug_draw 는 USD prim / collider / rigid body 를 만들지 않는 순수 시각화이다.
      -> 물리 scene 과 센서 판단에 영향 없음.
    - 판정은 full 3D sphere 기준 (robot base 원점으로부터 3D Euclidean distance <= MAX_REACH_M)이고,
      visualization 은 가독성을 위해 upper hemisphere 만 표시한다.
      (작업대 아래쪽 반구는 그리지 않을 뿐, 판정에서 제외되는 것이 아님)
    - debug_draw 의 clear_lines()/clear_points() 는 그려진 '모든' 선/점을 지우므로,
      boundary 선분은 (반경 / robot pose 가 바뀔 때만) 다시 계산해 캐시하고,
      변경이 생겼을 때만 clear -> boundary + target line 을 한 번에 다시 그린다 (매 frame 누적 없음)."""

    def __init__(self):
        if _debug_draw is None:
            raise RuntimeError("isaacsim.util.debug_draw is not available")
        self._draw = _debug_draw.acquire_debug_draw_interface()
        self.enabled = True
        self.radius = None
        self._boundary_key = None
        self._boundary = ([], [], [], [])       # starts, ends, colors, widths (캐시)
        self._base_world = None
        self._target_key = None
        self._target = None                     # (start, end, color) 또는 None
        self.target_text = "N/A"

    @staticmethod
    def _f3(p):
        return (float(p[0]), float(p[1]), float(p[2]))

    def _build_boundary(self, t_wr, R_wr, r):
        """robot base frame 의 축(ex, ey, ez)을 따라 상부 hemisphere wireframe + 작업대 표면 원을 만든다."""
        ex, ey, ez = R_wr[:, 0], R_wr[:, 1], R_wr[:, 2]
        segs = []                                                   # (starts, ends, color, width)
        bc, bw = REACH_VIZ_BOUNDARY_COLOR, REACH_VIZ_BOUNDARY_WIDTH
        # (1) XZ / YZ 방향 반원 arc (robot base frame 기준, 위쪽 절반만)
        segs.append((*_ring_segments(t_wr, ex, ez, r, 24, 0.0, math.pi), bc, bw))
        segs.append((*_ring_segments(t_wr, ey, ez, r, 24, 0.0, math.pi), bc, bw))
        # (2) 위도 ring (30 deg, 60 deg)
        for lat in REACH_VIZ_LATITUDES_DEG:
            el = math.radians(lat)
            segs.append((*_ring_segments(t_wr + r * math.sin(el) * ez, ex, ey, r * math.cos(el), 40), bc, bw))
        # (3) 작업대 표면 원: z = TABLE_TOP_Z + offset 평면과 판정 구(sphere)의 교선.
        #     robot base 가 작업대 위에 있으면 dz ~ 0 이므로 반경 ~= MAX_REACH_M.
        z_plane = TABLE_TOP_Z + REACH_VIZ_TABLE_OFFSET_M
        dz = z_plane - t_wr[2]
        if abs(dz) < r:
            r_table = math.sqrt(r * r - dz * dz)
            segs.append((*_ring_segments((t_wr[0], t_wr[1], z_plane), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                         r_table, 72), REACH_VIZ_TABLE_RING_COLOR, REACH_VIZ_TABLE_RING_WIDTH))
        # (4) robot base 평면의 equator: 작업대 원과 거의 겹치면(1 cm 이내) 생략
        if abs(dz) > 0.01:
            segs.append((*_ring_segments(t_wr, ex, ey, r, 64), bc, bw))
        # (5) robot base 중심 cross marker
        for axis in (ex, ey, ez):
            segs.append((np.array([t_wr - 0.04 * axis]), np.array([t_wr + 0.04 * axis]), REACH_VIZ_BASE_COLOR, 2.0))
        starts, ends, colors, widths = [], [], [], []
        for s, e, c, w in segs:
            starts += [self._f3(p) for p in s]
            ends += [self._f3(p) for p in e]
            colors += [tuple(c)] * len(s)
            widths += [float(w)] * len(s)
        return starts, ends, colors, widths

    def update_boundary(self, robot_pose, max_reach):
        """반경 또는 robot base pose 가 바뀐 경우에만 boundary 를 다시 계산하고 다시 그린다."""
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
        """RGB-D 로 추정한 Target p_world 와 robot base 를 잇는 선. GT pose 는 사용하지 않는다.
        REACHABLE = 초록, OUT OF RANGE = 빨강, 그 외(NOT DETECTED / DEPTH INVALID) = 선 없음."""
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

    def set_enabled(self, on):
        self.enabled = bool(on)
        self._redraw()

    def clear(self):
        """debug_draw 의 모든 선/점 제거."""
        self._draw.clear_lines()
        self._draw.clear_points()

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

    def shutdown(self):
        try:
            self.clear()
            _debug_draw.release_debug_draw_interface(self._draw)
        except Exception:  # noqa: BLE001
            pass


# =============================================================================
# 15. Isaac Sim 내부 결과 패널 (omni.ui) - 표시 전용, 계산은 하지 않음
# =============================================================================
def _abgr(r, g, b, a=255):
    """omni.ui 색상 정수 형식은 0xAABBGGRR."""
    return (a << 24) | (b << 16) | (g << 8) | r


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
    """거리 값 표기: 없으면 N/A."""
    return "N/A" if v is None else f"{v:.3f} m"


class FusionMonitorUI:
    """'RGB-D Fusion Monitor' : Isaac Sim GUI 안의 omni.ui Window.

    - 위젯은 생성자에서 한 번만 만들고, update() 에서는 Label.text / style 만 바꾼다.
    - 표시하는 값은 process_rgbd() 가 반환한 bundle(results / final / baselines) 그대로이며,
      GT 값은 표시하지 않는다.
    - 버튼은 동작을 직접 실행하지 않고 요청(request) 큐에 넣는다. main loop 가 simulation step 사이에
      꺼내서 apply_scenario() 등을 '한 번' 실행한다 -> 센서 처리와 UI 가 분리됨."""

    def __init__(self, title="RGB-D Fusion Monitor"):
        self._requests = []
        self._ticks = 0
        self._dock_attempted = False
        self.obj_widgets = {}
        self.window = ui.Window(title, width=MONITOR_WIDTH, height=MONITOR_HEIGHT)
        try:   # floating 기본 위치: 화면 오른쪽 (Stage/Property 패널 왼쪽)
            self.window.position_x = max(0, 1600 - MONITOR_WIDTH - 430)
            self.window.position_y = 40
        except Exception:  # noqa: BLE001
            pass
        self._build()

    # ---- 위젯 생성 helper ----
    @staticmethod
    def _section(text):
        ui.Spacer(height=8)
        ui.Label(text, height=20, style={"font_size": 15, "color": UI_HEADER})
        ui.Line(height=2, style={"color": UI_GRAY})
        ui.Spacer(height=2)

    @staticmethod
    def _kv(key, key_width=120):
        with ui.HStack(height=20):
            ui.Label(key, width=key_width, style={"color": UI_GRAY})
            value = ui.Label("N/A", style={"color": UI_WHITE})
        return value

    def _request(self, action):
        self._requests.append(action)
        self.lbl_action.text = f"Requested: {action}"

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
        """매 loop 호출. 몇 frame 뒤 (창이 실제로 만들어진 뒤) 한 번만 docking 을 시도한다."""
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

    def pop_requests(self):
        reqs, self._requests = self._requests, []
        return reqs

    def set_status(self, text):
        self.lbl_meta.text = text

    def set_action_text(self, text):
        self.lbl_action.text = text

    def update_reach(self, radius, boundary_on, line_text):
        """REACH VISUALIZATION 항목 갱신 (텍스트만 변경)."""
        self.lbl_reach_radius.text = "N/A" if radius is None else f"{radius:.2f} m"
        self.lbl_reach_boundary.text = boundary_on
        self.lbl_reach_boundary.style = {"color": UI_GREEN if boundary_on == "ON" else UI_GRAY}
        self.lbl_reach_line.text = line_text

    def update(self, bundle, meta):
        """bundle 의 실제 센서융합 결과만으로 text/style 을 갱신 (위젯 재생성 없음)."""
        results, final, base = bundle["results"], bundle["final"], bundle["baselines"]
        target = next((r for r in results if r["role"] == "TARGET"), None)
        status = final["status"]
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
# 16. 결과 저장
# =============================================================================
CSV_FIELDS = ["wall_time", "scenario", "frame", "camera_frame", "sim_time", "object", "role", "detected",
              "pixel_count", "valid_depth_px", "depth_m", "u", "v", "cam_x", "cam_y", "cam_z",
              "world_x", "world_y", "world_z", "robot_x", "robot_y", "robot_z", "robot_distance_m",
              "range_status", "max_reach_m", "final_result", "depth_only_nearest_m", "depth_only_naive_result"]


def save_results(bundle, meta, rgb, depth, annotated, depth_vis):
    """output/ 에 annotated RGB, depth 시각화, 결과 txt, results.csv(누적) 를 저장한다."""
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
        else:
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
    """OpenCV 창(선택 fallback) 키 입력 -> 패널 버튼과 같은 action 문자열."""
    if key in (ord("q"), 27):
        return "quit"
    if key in (ord("a"), ord("b"), ord("c")):
        return "scenario:" + chr(key).upper()
    return {ord("s"): "save", ord("]"): "reach+", ord("["): "reach-"}.get(key)


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
    camera.add_distance_to_image_plane_to_frame()   # Depth(Z) annotator 를 RGB 와 같은 render product 에 부착
    apply_robot_home(scene)
    print_startup_info(camera, scene)
    validate_scene_layout(camera, scene)

    # ---- 결과 표시 계층 (센서 처리와 분리). 실패해도 console-only 로 계속 ----
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

    # ---- [VIZ] Viewport reach 시각화 (실패해도 센서 처리 / 판단에는 영향 없음) ----
    reach_viz = None
    if REACH_VIZ_ENABLED and _debug_draw is not None:
        try:
            reach_viz = ReachVisualizer()
            log("Reach visualization (debug_draw) enabled: upper-hemisphere wireframe + table ring + target line.")
        except Exception as e:  # noqa: BLE001
            reach_viz = None
            log(f"ReachVisualizer creation failed ({e}); continuing without reach visualization.", "WARN")

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
            # Stop 버튼 -> 다시 Play 시 reset (물체는 default state 로 돌아감)
            reset_needed = True
            if ui_panel is not None:
                ui_panel.set_status("STOPPED - press Play to resume sensing")
            key = viz.poll()
        elif not world.is_playing():          # Pause
            if ui_panel is not None:
                ui_panel.set_status("PAUSED")
            key = viz.poll()
        elif reset_needed:
            world.reset()
            apply_robot_home(scene)
            reset_needed = False
            steps_since_reset = 0
            last_cam_frame = None
        else:
            step += 1
            steps_since_reset += 1
            frame = acquire_rgbd_frame(camera) if steps_since_reset >= WARMUP_FRAMES else None
            if frame is None:
                if steps_since_reset >= WARMUP_FRAMES:
                    RL.warn("noframe", "Camera buffers not ready yet (empty RGB/Depth); waiting...")
                key = viz.poll()
            elif frame["rendering_frame"] is not None and frame["rendering_frame"] == last_cam_frame:
                key = viz.poll()            # 새 카메라 frame 이 없으면 처리 생략 (CAMERA_FREQUENCY 기준)
            else:
                # ---- 같은 frame 의 RGB(t), Depth(t) 로 센서융합 ----
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
                # [VIZ] Target 연결선: process_rgbd() 의 RGB-D 추정 p_world 와 같은 bundle 의 robot base 위치 사용
                #       (stage GT pose 사용 안 함). NOT DETECTED / DEPTH INVALID 이면 선을 숨긴다.
                if bundle is not None and reach_viz is not None:
                    try:
                        tgt = next((r for r in bundle["results"] if r["role"] == "TARGET"), None)
                        reach_viz.update_target_line(bundle["robot_pose"][0],
                                                     tgt["p_world"] if tgt is not None else None,
                                                     bundle["final"]["status"])
                    except Exception as e:  # noqa: BLE001
                        log(f"Reach visualization error ({e}); disabling it, sensing continues.", "WARN")
                        reach_viz = None
                # UI 갱신: 위 bundle 을 그대로 전달 (UI 오류는 센서 처리에 영향 없음)
                if bundle is not None and ui_panel is not None:
                    try:
                        ui_panel.update(bundle, meta)
                    except Exception as e:  # noqa: BLE001
                        log(f"Fusion Monitor update failed ({e}); disabling panel, console continues.", "WARN")
                        traceback.print_exc()
                        ui_panel = None

        # ---- 사용자 요청 처리 (패널 버튼 / OpenCV 키). 요청 시 '한 번만' 실행 ----
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
                save_requested = True           # 다음 처리 frame 의 결과를 저장
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

        # ---- [VIZ] reach boundary: 현재 MAX_REACH_M 와 stage 에서 읽은 robot base pose 사용.
        #      둘 중 하나가 바뀐 경우에만 내부에서 clear 후 다시 그림 (변화 없으면 아무것도 안 함) ----
        if reach_viz is not None:
            try:
                reach_viz.update_boundary(get_robot_base_pose(scene), MAX_REACH_M)
            except Exception as e:  # noqa: BLE001
                log(f"Reach visualization error ({e}); disabling it, sensing continues.", "WARN")
                reach_viz = None
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
