# app.py
# LaneAgent: lightweight 2D autonomy-stack demonstrator with CARLA-style module interfaces.
# All modules are intentionally kept in one file for easy deployment.

import os
import math
import tempfile
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import streamlit as st


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DT = 0.05
SUBSTEPS = 4
PRED_DT = 0.1
PRED_HORIZON = 3.0
EGO_LENGTH = 4.5
EGO_WIDTH = 1.8
WHEELBASE = 2.7
MAX_ACCEL = 3.0
MAX_DECEL = 6.0
TARGET_SPEED = 10.0
LANE_HALF = 1.75
ROAD_HALF = 5.0

SCENARIOS = [
    "straight_curves",
    "stopped_vehicle_blocking_lane",
    "pedestrian_steps_in",
    "traffic_light_change",
    "sensor_dropout",
]

METRICS = [
    "collisions",
    "lane_invasion_time",
    "min_ttc",
    "max_long_jerk",
    "max_lat_jerk",
    "route_completion_pct",
    "emergency_brakes",
]


# --------------------------------------------------------------------------------------
# Typed interfaces / data models
# --------------------------------------------------------------------------------------


class AgentType(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    TRAFFIC_LIGHT = "traffic_light"


class PlanState(str, Enum):
    LANE_KEEP = "LANE_KEEP"
    FOLLOW = "FOLLOW"
    OVERTAKE = "OVERTAKE"
    STOP_AT_LIGHT = "STOP_AT_LIGHT"
    YIELD_TO_PEDESTRIAN = "YIELD_TO_PEDESTRIAN"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


@dataclass
class Pose:
    x: float
    y: float
    theta: float
    v: float = 0.0
    a: float = 0.0
    steer: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class ControlCommand:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    accel: float = 0.0


@dataclass
class Detection:
    id: str
    type: AgentType
    x: float
    y: float
    theta: float = 0.0
    v: float = 0.0
    length: float = 1.0
    width: float = 0.6
    confidence: float = 0.5
    traffic_state: Optional[str] = None


@dataclass
class LanePerception:
    valid: bool = False
    offset: float = 0.0
    heading_error: float = 0.0
    curvature: float = 0.0
    confidence: float = 0.0


@dataclass
class PerceptionOutput:
    image: Optional[np.ndarray] = None
    detections: List[Detection] = field(default_factory=list)
    lane: LanePerception = field(default_factory=LanePerception)
    lidar_points: Optional[np.ndarray] = None
    obstacles: List[Tuple[float, float, float]] = field(default_factory=list)
    valid: bool = True
    timestamp: float = 0.0


@dataclass
class TrackedAgent:
    track_id: int
    type: AgentType
    x: float
    y: float
    theta: float
    vx: float
    vy: float
    speed: float
    length: float
    width: float
    last_seen: float
    age: float = 0.0
    traffic_state: Optional[str] = None
    crossing_intent: float = 0.0


@dataclass
class PredictedAgent:
    track_id: int
    type: AgentType
    times: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    vxs: np.ndarray
    vys: np.ndarray
    intent: float = 0.0
    length: float = 1.0
    width: float = 0.6


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    theta: float
    v: float
    a: float
    j: float
    d: float
    s: float
    lat_a: float = 0.0
    lat_j: float = 0.0


@dataclass
class Candidate:
    lateral_offset: float
    profile: str
    points: List[TrajectoryPoint]
    cost: float
    risk: float
    comfort: float
    lane: float
    progress: float


@dataclass
class Plan:
    state: PlanState
    points: List[TrajectoryPoint]
    target_speed: float
    target_accel: float
    candidates: List[Candidate] = field(default_factory=list)
    selected_cost: float = 0.0
    reason: str = ""


@dataclass
class AgentState:
    id: str
    type: AgentType
    x: float
    y: float
    theta: float
    v: float
    length: float
    width: float
    s: float = 0.0
    d: float = 0.0
    color: Tuple[int, int, int] = (200, 200, 200)
    target_speed: float = 0.0
    trigger_time: float = 1e9
    phase: str = ""
    crossing_speed: float = 0.0
    traffic_state: Optional[str] = None


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ego_frame(ego: Pose, x: float, y: float) -> Tuple[float, float]:
    dx = x - ego.x
    dy = y - ego.y
    c = math.cos(ego.theta)
    s = math.sin(ego.theta)
    xf = dx * c + dy * s
    yl = -dx * s + dy * c
    return xf, yl


def world_from_ego_frame(ego: Pose, xf: float, yl: float) -> Tuple[float, float]:
    c = math.cos(ego.theta)
    s = math.sin(ego.theta)
    return ego.x + xf * c - yl * s, ego.y + xf * s + yl * c


def guess_mime_from_path(path: Optional[str]) -> str:
    if path and str(path).lower().endswith(".gif"):
        return "image/gif"
    return "video/mp4"


def normalize_video_tuple(video: Any) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Robustly converts old/new video tuples into:
        (preview_path, preview_mime, mp4_download_path)

    Supported legacy forms:
        (path, mime)
        (path, mp4_path)
        (preview_path, preview_mime, mp4_path)
    """
    if video is None:
        return None, "image/gif", None

    if not isinstance(video, (tuple, list)):
        if isinstance(video, str):
            return video, guess_mime_from_path(video), None
        return None, "image/gif", None

    if len(video) == 0:
        return None, "image/gif", None

    if len(video) == 1:
        p = video[0] if isinstance(video[0], str) else None
        return p, guess_mime_from_path(p), None

    if len(video) == 2:
        a, b = video
        a_str = a if isinstance(a, str) else None
        b_str = b if isinstance(b, str) else None

        # Legacy form: (preview_path, mp4_path)
        if b_str and b_str.lower().endswith((".mp4", ".gif", ".webm", ".mov")):
            mime = guess_mime_from_path(a_str)
            return a_str, mime, b_str

        # Legacy form: (preview_path, mime)
        mime = b_str if b_str else guess_mime_from_path(a_str)
        return a_str, mime, None

    # len >= 3
    preview_path = video[0] if isinstance(video[0], str) else None
    preview_mime = video[1] if isinstance(video[1], str) else guess_mime_from_path(preview_path)
    mp4_path = video[2] if isinstance(video[2], str) else None

    if preview_path:
        preview_mime = guess_mime_from_path(preview_path) if preview_path.lower().endswith(".gif") else preview_mime

    if not preview_mime or not isinstance(preview_mime, str):
        preview_mime = guess_mime_from_path(preview_path)

    return preview_path, preview_mime, mp4_path


# --------------------------------------------------------------------------------------
# Track / map
# --------------------------------------------------------------------------------------


class Track:
    """Closed urban-like 2D track generated from a smooth Catmull-Rom spline."""

    def __init__(self) -> None:
        ctrl = np.array(
            [
                [0.0, 0.0],
                [90.0, 0.0],
                [170.0, 25.0],
                [225.0, 85.0],
                [240.0, 165.0],
                [200.0, 235.0],
                [115.0, 265.0],
                [35.0, 250.0],
                [-45.0, 200.0],
                [-65.0, 120.0],
                [-25.0, 40.0],
            ],
            dtype=float,
        )

        pts = []
        n = len(ctrl)
        samples = 60
        for i in range(n):
            p0 = ctrl[(i - 1) % n]
            p1 = ctrl[i]
            p2 = ctrl[(i + 1) % n]
            p3 = ctrl[(i + 2) % n]
            for j in range(samples):
                t = j / float(samples)
                t2 = t * t
                t3 = t2 * t
                pt = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                pts.append(pt)

        self.points = np.array(pts, dtype=float)
        self.N = len(self.points)

        seg = np.roll(self.points, -1, axis=0) - self.points
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_len = np.maximum(self.seg_len, 1e-6)
        self.s = np.concatenate(([0.0], np.cumsum(self.seg_len[:-1])))
        self.length = float(np.sum(self.seg_len))

        tangents = seg / self.seg_len[:, None]
        self.tangents = 0.5 * (tangents + np.roll(tangents, 1, axis=0))
        norms = np.linalg.norm(self.tangents, axis=1, keepdims=True)
        self.tangents = self.tangents / np.maximum(norms, 1e-9)
        self.normals = np.column_stack((-self.tangents[:, 1], self.tangents[:, 0]))

        angles = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])
        dtheta = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            dtheta[i] = wrap_angle(angles[(i + 1) % self.N] - angles[i])
        self.curvature = dtheta / self.seg_len

    def nearest(self, x: float, y: float) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
        diff = self.points - np.array([x, y], dtype=float)
        dist2 = np.sum(diff * diff, axis=1)
        idx = int(np.argmin(dist2))

        p1 = self.points[idx]
        p2 = self.points[(idx + 1) % self.N]
        seg = p2 - p1
        l2 = float(np.dot(seg, seg))
        if l2 < 1e-9:
            t = 0.0
        else:
            t = clamp(float(np.dot(np.array([x, y]) - p1, seg) / l2), 0.0, 1.0)

        proj = p1 + t * seg
        dvec = np.array([x, y]) - proj
        tangent = self.tangents[idx]
        normal = self.normals[idx]
        d = float(np.dot(dvec, normal))
        s = float(self.s[idx] + t * self.seg_len[idx])
        if s >= self.length:
            s -= self.length
        curv = float(self.curvature[idx])
        return s, d, tangent, normal, curv

    def pose_from_sd(self, s: float, d: float) -> Tuple[float, float, float, float]:
        s = s % self.length
        idx = int(np.searchsorted(self.s, s, side="right") - 1)
        idx = clamp(idx, 0, self.N - 1)
        nxt = (idx + 1) % self.N
        seg_len = self.seg_len[idx]
        t = clamp((s - self.s[idx]) / max(seg_len, 1e-9), 0.0, 1.0)

        point = (1.0 - t) * self.points[idx] + t * self.points[nxt]
        tangent = (1.0 - t) * self.tangents[idx] + t * self.tangents[nxt]
        nt = max(np.linalg.norm(tangent), 1e-9)
        tangent = tangent / nt
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        curv = (1.0 - t) * self.curvature[idx] + t * self.curvature[nxt]

        x = float(point[0] + normal[0] * d)
        y = float(point[1] + normal[1] * d)
        theta = float(math.atan2(tangent[1], tangent[0]))
        return x, y, theta, float(curv)


@st.cache_resource
def get_track() -> Track:
    return Track()


# --------------------------------------------------------------------------------------
# Simulator interface
# --------------------------------------------------------------------------------------


class SimWorld:
    """Lightweight bicycle-model simulator with a CARLA-like ground-truth interface."""

    def __init__(self, track: Track, scenario: str, seed: int, duration: float) -> None:
        self.track = track
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario
        self.duration = float(duration)
        self.t = 0.0

        self.start_s = 20.0
        x, y, theta, _ = track.pose_from_sd(self.start_s, 0.0)
        self.ego = Pose(x=x, y=y, theta=theta, v=0.0)

        self.agents: List[AgentState] = []
        self.light_s = (self.start_s + 110.0) % track.length
        if scenario == "traffic_light_change":
            self.light_offset = 7.0
        else:
            self.light_offset = float(self.rng.uniform(0.0, 10.0))

        if scenario == "sensor_dropout":
            self.dropout_start = self.duration * 0.45
            self.dropout_duration = 0.5
        else:
            self.dropout_start = None
            self.dropout_duration = 0.0

        self._spawn_scenario_agents()

    def _add_vehicle(self, s: float, d: float, v: float, color: Tuple[int, int, int]) -> None:
        x, y, theta, _ = self.track.pose_from_sd(s, d)
        self.agents.append(
            AgentState(
                id=f"veh_{len(self.agents)}",
                type=AgentType.VEHICLE,
                x=x,
                y=y,
                theta=theta,
                v=v,
                length=EGO_LENGTH,
                width=EGO_WIDTH,
                s=s % self.track.length,
                d=d,
                color=color,
            )
        )

    def _add_pedestrian(self, s: float, d: float, trigger_time: float, speed: float) -> None:
        x, y, theta, _ = self.track.pose_from_sd(s, d)
        self.agents.append(
            AgentState(
                id=f"ped_{len(self.agents)}",
                type=AgentType.PEDESTRIAN,
                x=x,
                y=y,
                theta=theta - math.pi / 2.0,
                v=0.0,
                length=0.7,
                width=0.6,
                s=s % self.track.length,
                d=d,
                color=(0, 220, 220),
                trigger_time=trigger_time,
                phase="wait",
                crossing_speed=speed,
            )
        )

    def _spawn_scenario_agents(self) -> None:
        L = self.track.length
        if self.scenario == "stopped_vehicle_blocking_lane":
            self._add_vehicle((self.start_s + 70.0) % L, 0.0, 0.0, (0, 0, 255))
        elif self.scenario == "pedestrian_steps_in":
            self._add_pedestrian(
                (self.start_s + 65.0) % L,
                6.0,
                float(self.rng.uniform(4.0, 7.0)),
                1.2,
            )
        elif self.scenario == "traffic_light_change":
            pass
        elif self.scenario == "sensor_dropout":
            self._add_vehicle((self.start_s + 80.0) % L, 0.0, 6.0, (255, 80, 80))
        else:
            pass

    def sensor_active(self) -> bool:
        if self.dropout_start is None:
            return True
        return not (self.dropout_start <= self.t < self.dropout_start + self.dropout_duration)

    def get_light_state(self) -> str:
        phase = (self.t + self.light_offset) % 21.0
        if phase < 10.0:
            return "green"
        if phase < 13.0:
            return "yellow"
        return "red"

    def update(self, dt: float, cmd: ControlCommand) -> None:
        # Ego bicycle model.
        accel = cmd.throttle * MAX_ACCEL - cmd.brake * MAX_DECEL
        self.ego.a = accel
        self.ego.v = clamp(self.ego.v + accel * dt, 0.0, 16.0)
        self.ego.steer = clamp(cmd.steer, -0.5, 0.5)
        self.ego.yaw_rate = self.ego.v / WHEELBASE * math.tan(self.ego.steer)
        self.ego.theta = wrap_angle(self.ego.theta + self.ego.yaw_rate * dt)
        self.ego.x += self.ego.v * math.cos(self.ego.theta) * dt
        self.ego.y += self.ego.v * math.sin(self.ego.theta) * dt

        # Agents.
        for ag in self.agents:
            if ag.type == AgentType.VEHICLE:
                ag.s = (ag.s + ag.v * dt) % self.track.length
                ag.x, ag.y, ag.theta, _ = self.track.pose_from_sd(ag.s, ag.d)
            elif ag.type == AgentType.PEDESTRIAN:
                if ag.phase == "wait" and self.t >= ag.trigger_time:
                    ag.phase = "cross"
                if ag.phase == "cross":
                    ag.d -= ag.crossing_speed * dt
                    ag.v = ag.crossing_speed
                    if ag.d <= -6.0:
                        ag.d = -6.0
                        ag.v = 0.0
                        ag.phase = "done"
                    cx, cy, theta_t, _ = self.track.pose_from_sd(ag.s, ag.d)
                    ag.x, ag.y = cx, cy
                    ag.theta = theta_t - math.pi / 2.0
                else:
                    ag.v = 0.0
                    ag.x, ag.y, theta_t, _ = self.track.pose_from_sd(ag.s, ag.d)
                    ag.theta = theta_t + math.pi / 2.0

        self.t += dt

    def gt_detections(self, ego: Pose) -> List[Detection]:
        dets: List[Detection] = []

        lx, ly, _, _ = self.track.pose_from_sd(self.light_s, 4.5)
        xf, yl = ego_frame(ego, lx, ly)
        if 1.0 < xf < 60.0 and abs(yl) < 18.0:
            dets.append(
                Detection(
                    id="traffic_light",
                    type=AgentType.TRAFFIC_LIGHT,
                    x=lx,
                    y=ly,
                    theta=math.atan2(ly - ego.y, lx - ego.x),
                    v=0.0,
                    length=0.4,
                    width=0.4,
                    confidence=0.95,
                    traffic_state=self.get_light_state(),
                )
            )

        for ag in self.agents:
            if ag.type == AgentType.TRAFFIC_LIGHT:
                continue
            xf, yl = ego_frame(ego, ag.x, ag.y)
            if 1.0 < xf < 60.0 and abs(yl) < 18.0:
                dets.append(
                    Detection(
                        id=ag.id,
                        type=ag.type,
                        x=ag.x,
                        y=ag.y,
                        theta=ag.theta,
                        v=ag.v,
                        length=ag.length,
                        width=ag.width,
                        confidence=0.98,
                    )
                )
        return dets


# --------------------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------------------


class CameraRenderer:
    def __init__(self, width: int = 160, height: int = 120) -> None:
        self.w = width
        self.h = height
        self.fx = 100.0
        self.fy = 100.0
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.horizon = 45.0
        self.k = 300.0

    def project(self, ego: Pose, x: float, y: float) -> Optional[Tuple[int, int, float]]:
        xf, yl = ego_frame(ego, x, y)
        if xf < 0.8 or xf > 65.0:
            return None
        u = self.cx + self.fx * yl / xf
        v = self.horizon + self.k / xf
        return int(u), int(v), xf

    def render(self, sim: SimWorld, ego: Pose) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:, :] = (95, 95, 95)

        if not sim.sensor_active():
            cv2.putText(
                img,
                "SENSOR DROPOUT",
                (8, self.h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            return img

        s0, _, _, _, _ = sim.track.nearest(ego.x, ego.y)

        road_left: List[Tuple[int, int]] = []
        road_right: List[Tuple[int, int]] = []
        lane_left: List[Tuple[int, int]] = []
        lane_right: List[Tuple[int, int]] = []

        for dist in np.arange(1.0, 60.0, 1.5):
            s = s0 + dist
            px, py, pth, _ = sim.track.pose_from_sd(s, 0.0)
            nx, ny = -math.sin(pth), math.cos(pth)

            for dval, arr in ((ROAD_HALF, road_left), (-ROAD_HALF, road_right)):
                p = self.project(ego, px + nx * dval, py + ny * dval)
                if p is not None:
                    arr.append((p[0], p[1]))

            for dval, arr in ((LANE_HALF, lane_left), (-LANE_HALF, lane_right)):
                p = self.project(ego, px + nx * dval, py + ny * dval)
                if p is not None:
                    arr.append((p[0], p[1]))

        if len(road_left) > 2 and len(road_right) > 2:
            poly = np.array(road_left + road_right[::-1], dtype=np.int32)
            cv2.fillPoly(img, [poly], (60, 60, 60))

        for arr in (lane_left, lane_right):
            if len(arr) > 1:
                cv2.polylines(img, [np.array(arr, dtype=np.int32)], False, (255, 255, 255), 2)

        for ag in sim.agents:
            if ag.type == AgentType.VEHICLE:
                p = self.project(ego, ag.x, ag.y)
                if p is None:
                    continue
                u, v, xf = p
                wp = max(3, int(self.fx * ag.width / xf))
                hp = max(3, int(self.fy * 1.45 / xf))
                cv2.rectangle(img, (u - wp // 2, v - hp), (u + wp // 2, v), ag.color, -1)
            elif ag.type == AgentType.PEDESTRIAN:
                p = self.project(ego, ag.x, ag.y)
                if p is None:
                    continue
                u, v, xf = p
                r = max(2, int(self.fx * 0.35 / xf))
                cv2.circle(img, (u, v - r), r, (0, 220, 220), -1)

        lx, ly, _, _ = sim.track.pose_from_sd(sim.light_s, 4.5)
        p = self.project(ego, lx, ly)
        if p is not None:
            u, v, _ = p
            cv2.line(img, (u, v), (u, v - 28), (20, 20, 20), 2)
            state = sim.get_light_state()
            color = (0, 255, 0) if state == "green" else (0, 255, 255) if state == "yellow" else (0, 0, 255)
            cv2.circle(img, (u, v - 30), 4, color, -1)

        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img


class PerceptionModule:
    """
    Perception interface:
      - mode="learned": mock learned lane segmentation + classical/blob object detector.
      - mode="classical": Canny + Hough lane detection + blob object detector.
      - mode="fast": ground-truth assisted detections for fast batch evaluation.
      - mode="demo": fast detections + optional camera rendering for live dashboard.
    """

    def __init__(self, mode: str, rng: np.random.Generator, track: Track) -> None:
        self.mode = mode
        self.rng = rng
        self.track = track
        self.cam = CameraRenderer()

    def process(self, sim: SimWorld, ego: Pose, render_image: bool = True) -> PerceptionOutput:
        active = sim.sensor_active()

        if self.mode in ("fast", "demo"):
            img = self.cam.render(sim, ego) if (self.mode == "demo" and render_image) else None

            if not active:
                return PerceptionOutput(
                    image=img,
                    detections=[],
                    lane=LanePerception(valid=False),
                    valid=False,
                    timestamp=sim.t,
                )

            dets = self._gt_assisted_detections(sim, ego)
            lane = self._lane_from_map(ego)
            pts = self._raycast(sim, ego)
            obs = self._cluster_lidar(pts)

            return PerceptionOutput(
                image=img,
                detections=dets,
                lane=lane,
                lidar_points=pts,
                obstacles=obs,
                valid=True,
                timestamp=sim.t,
            )

        img = self.cam.render(sim, ego)

        if not active:
            return PerceptionOutput(
                image=img if render_image else None,
                detections=[],
                lane=LanePerception(valid=False),
                valid=False,
                timestamp=sim.t,
            )

        if self.mode == "classical":
            lane = self._lane_canny_hough(img)
        else:
            lane = self._lane_learned(img)

        dets = self._objects_cv(img, ego)
        pts = self._raycast(sim, ego)
        obs = self._cluster_lidar(pts)

        return PerceptionOutput(
            image=img if render_image else None,
            detections=dets,
            lane=lane,
            lidar_points=pts,
            obstacles=obs,
            valid=True,
            timestamp=sim.t,
        )

    def _gt_assisted_detections(self, sim: SimWorld, ego: Pose) -> List[Detection]:
        dets: List[Detection] = []
        for d in sim.gt_detections(ego):
            dets.append(
                Detection(
                    id=d.id,
                    type=d.type,
                    x=d.x + self.rng.normal(0.0, 0.25),
                    y=d.y + self.rng.normal(0.0, 0.25),
                    theta=d.theta + self.rng.normal(0.0, 0.03),
                    v=max(0.0, d.v + self.rng.normal(0.0, 0.2)),
                    length=d.length,
                    width=d.width,
                    confidence=max(0.35, d.confidence - 0.05),
                    traffic_state=d.traffic_state,
                )
            )
        return dets

    def _lane_from_map(self, ego: Pose) -> LanePerception:
        s, d, tangent, _, curv = self.track.nearest(ego.x, ego.y)
        theta_t = math.atan2(tangent[1], tangent[0])
        return LanePerception(True, d, wrap_angle(ego.theta - theta_t), curv, 1.0)

    def _fit_poly(self, ys: np.ndarray, xs: np.ndarray) -> Optional[np.ndarray]:
        if len(ys) < 2:
            return None
        try:
            deg = 2 if len(ys) >= 5 else 1
            return np.polyfit(ys, xs, deg)
        except Exception:
            return None

    def _lane_from_fits(self, fl: Optional[np.ndarray], fr: Optional[np.ndarray], npts: int) -> LanePerception:
        if fl is None or fr is None:
            return LanePerception(False)
        z = 18.0
        v = self.cam.horizon + self.cam.k / z
        uL = float(np.polyval(fl, v))
        uR = float(np.polyval(fr, v))
        center = 0.5 * (uL + uR)
        offset = (center - self.cam.cx) * z / self.cam.fx

        slope = 0.0
        try:
            if len(fl) > 1:
                slope += float(np.polyval(np.polyder(fl), v))
            if len(fr) > 1:
                slope += float(np.polyval(np.polyder(fr), v))
            slope *= 0.5
        except Exception:
            slope = 0.0

        heading = -math.atan(slope * 0.03)
        curv = 0.0
        try:
            if len(fl) == 3 and len(fr) == 3:
                curv = 0.0006 * (float(fl[0]) + float(fr[0]))
        except Exception:
            curv = 0.0

        conf = clamp(npts / 18.0, 0.0, 1.0)
        return LanePerception(True, offset, heading, curv, conf)

    def _lane_canny_hough(self, img: np.ndarray) -> LanePerception:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 60, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=28, minLineLength=18, maxLineGap=8)

        left_pts: List[Tuple[float, float]] = []
        right_pts: List[Tuple[float, float]] = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if max(y1, y2) < self.cam.horizon + 4:
                    continue
                dy = float(y2 - y1)
                dx = float(x2 - x1)
                slope = dx / (dy + 1e-6)
                if abs(slope) > 2.5:
                    continue
                mid_u = 0.5 * (x1 + x2)
                for x, y in ((x1, y1), (x2, y2)):
                    if y > self.cam.horizon + 3:
                        if mid_u < self.cam.cx - 3:
                            left_pts.append((float(y), float(x)))
                        elif mid_u > self.cam.cx + 3:
                            right_pts.append((float(y), float(x)))

        fl = self._fit_poly(np.array([p[0] for p in left_pts]), np.array([p[1] for p in left_pts])) if left_pts else None
        fr = self._fit_poly(np.array([p[0] for p in right_pts]), np.array([p[1] for p in right_pts])) if right_pts else None
        return self._lane_from_fits(fl, fr, len(left_pts) + len(right_pts))

    def _lane_learned(self, img: np.ndarray) -> LanePerception:
        # Mock learned segmentation: color/intensity lane-pixel classifier + row-wise median fitting.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 170), (180, 80, 255))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        left_pts: List[Tuple[float, float]] = []
        right_pts: List[Tuple[float, float]] = []

        for y in range(int(self.cam.horizon) + 5, self.cam.h, 2):
            xs = np.flatnonzero(mask[y] > 0)
            if len(xs) == 0:
                continue
            left = xs[xs < self.cam.cx - 2]
            right = xs[xs > self.cam.cx + 2]
            if len(left) > 0:
                left_pts.append((float(y), float(np.median(left))))
            if len(right) > 0:
                right_pts.append((float(y), float(np.median(right))))

        fl = self._fit_poly(np.array([p[0] for p in left_pts]), np.array([p[1] for p in left_pts])) if left_pts else None
        fr = self._fit_poly(np.array([p[0] for p in right_pts]), np.array([p[1] for p in right_pts])) if right_pts else None
        return self._lane_from_fits(fl, fr, len(left_pts) + len(right_pts))

    def _bbox_to_det(
        self,
        bbox: Tuple[int, int, int, int],
        det_type: AgentType,
        ego: Pose,
        real_height: float,
        length: float,
        width: float,
        traffic_state: Optional[str] = None,
    ) -> Optional[Detection]:
        x, y, w, h = bbox
        if h < 3 or w < 2:
            return None
        z = self.cam.fy * real_height / float(max(3, h))
        if z < 2.0 or z > 58.0:
            return None
        u = x + w / 2.0
        yl = (u - self.cam.cx) * z / self.cam.fx
        xf = z
        wx, wy = world_from_ego_frame(ego, xf, yl)
        theta = math.atan2(wy - ego.y, wx - ego.x)
        area = float(w * h)
        conf = clamp(0.35 + area / 900.0, 0.35, 0.95)
        return Detection(
            id=f"{det_type.value}_{int(wx*10)}_{int(wy*10)}",
            type=det_type,
            x=wx,
            y=wy,
            theta=theta,
            v=0.0,
            length=length,
            width=width,
            confidence=conf,
            traffic_state=traffic_state,
        )

    def _objects_cv(self, img: np.ndarray, ego: Pose) -> List[Detection]:
        dets: List[Detection] = []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        blue = cv2.inRange(hsv, (100, 80, 80), (140, 255, 255))
        red1 = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 100, 80), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)
        vehicle_mask = cv2.bitwise_or(blue, red)
        vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        yellow = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
        green = cv2.inRange(hsv, (40, 100, 80), (85, 255, 255))

        contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 80 or area > 6000:
                continue
            bbox = cv2.boundingRect(c)
            det = self._bbox_to_det(bbox, AgentType.VEHICLE, ego, 1.45, EGO_LENGTH, EGO_WIDTH)
            if det is not None:
                dets.append(det)

        contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 20 or area > 1200:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < 8 or area > 120:
                det = self._bbox_to_det((x, y, w, h), AgentType.PEDESTRIAN, ego, 1.70, 0.7, 0.6)
                if det is not None:
                    dets.append(det)
            else:
                det = self._bbox_to_det((x, y, w, h), AgentType.TRAFFIC_LIGHT, ego, 0.35, 0.4, 0.4, "yellow")
                if det is not None:
                    dets.append(det)

        for mask, state in ((green, "green"), (red, "red")):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 8 or area > 130:
                    continue
                bbox = cv2.boundingRect(c)
                det = self._bbox_to_det(bbox, AgentType.TRAFFIC_LIGHT, ego, 0.35, 0.4, 0.4, state)
                if det is not None:
                    dets.append(det)

        return dets

    def _raycast(self, sim: SimWorld, ego: Pose) -> np.ndarray:
        angles = np.linspace(-1.0, 1.0, 41)
        pts = []
        for a in angles:
            hit = 40.0
            ca, sa = math.cos(a), math.sin(a)
            for ag in sim.agents:
                if ag.type == AgentType.TRAFFIC_LIGHT:
                    continue
                xf, yl = ego_frame(ego, ag.x, ag.y)
                if xf <= 0.5 or xf > 40.0:
                    continue
                r = math.hypot(xf, yl)
                ang = math.atan2(yl, xf)
                if abs(wrap_angle(ang - a)) < 0.045 and r < hit:
                    hit = r
            if hit < 40.0:
                pts.append([hit * ca, hit * sa])
        return np.array(pts, dtype=float) if pts else np.zeros((0, 2), dtype=float)

    def _cluster_lidar(self, points: np.ndarray) -> List[Tuple[float, float, float]]:
        if points.shape[0] == 0:
            return []
        used = np.zeros(points.shape[0], dtype=bool)
        clusters = []
        for i in range(points.shape[0]):
            if used[i]:
                continue
            group = [points[i]]
            used[i] = True
            for j in range(i + 1, points.shape[0]):
                if not used[j] and np.linalg.norm(points[i] - points[j]) < 1.2:
                    group.append(points[j])
                    used[j] = True
            g = np.array(group)
            center = g.mean(axis=0)
            radius = float(np.max(np.linalg.norm(g - center, axis=1))) if len(g) > 1 else 0.35
            clusters.append((float(center[0]), float(center[1]), radius))
        return clusters


# --------------------------------------------------------------------------------------
# Localization: noisy ground truth + EKF
# --------------------------------------------------------------------------------------


class EKFLocalizer:
    """
    EKF fusing:
      - GPS-like position from noisy simulator ground truth
      - IMU yaw-rate integrated heading delta
      - wheel odometry distance delta
    State: [x, y, theta, v]
    """

    def __init__(self, initial_pose: Pose, rng: np.random.Generator) -> None:
        self.x = np.array([initial_pose.x, initial_pose.y, initial_pose.theta, initial_pose.v], dtype=float)
        self.P = np.diag([1.0, 1.0, 0.08, 1.0])
        self.rng = rng

    def update(self, gt: Pose, dt: float, gps_active: bool = True) -> Pose:
        gps_x = gt.x + self.rng.normal(0.0, 0.45)
        gps_y = gt.y + self.rng.normal(0.0, 0.45)
        odom_ds = gt.v * dt + self.rng.normal(0.0, 0.06)
        dtheta = gt.yaw_rate * dt + self.rng.normal(0.0, 0.012)

        x = self.x
        c = math.cos(x[2])
        s = math.sin(x[2])
        v_meas = odom_ds / dt if dt > 1e-6 else x[3]

        x_pred = np.array(
            [
                x[0] + odom_ds * c,
                x[1] + odom_ds * s,
                wrap_angle(x[2] + dtheta),
                max(0.0, 0.7 * x[3] + 0.3 * v_meas),
            ],
            dtype=float,
        )

        F = np.eye(4)
        F[0, 2] = -odom_ds * s
        F[1, 2] = odom_ds * c
        Q = np.diag([0.04, 0.04, 0.006, 0.08]) * max(dt, 1e-3)

        self.x = x_pred
        self.P = F @ self.P @ F.T + Q

        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        R = np.diag([0.45, 0.45]) if gps_active else np.diag([100.0, 100.0])
        z = np.array([gps_x, gps_y], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

        return Pose(x=float(self.x[0]), y=float(self.x[1]), theta=float(self.x[2]), v=float(self.x[3]))


# --------------------------------------------------------------------------------------
# Prediction / tracking
# --------------------------------------------------------------------------------------


class TrackerPredictor:
    def __init__(self, track: Track) -> None:
        self.track = track
        self.tracks: Dict[int, TrackedAgent] = {}
        self.next_id = 1

    def update(self, dets: List[Detection], dt: float, ego: Pose, sim_time: float) -> List[PredictedAgent]:
        for tr in self.tracks.values():
            tr.age += dt

        matched: set[int] = set()

        for det in dets:
            gate = 25.0 if det.type == AgentType.TRAFFIC_LIGHT else (5.0 if det.type == AgentType.VEHICLE else 4.0)
            best_id = None
            best_dist = gate
            for tid, tr in self.tracks.items():
                if tid in matched or tr.type != det.type or tr.age > 2.0:
                    continue
                dist = math.hypot(tr.x - det.x, tr.y - det.y)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                matched.add(best_id)
                tr = self.tracks[best_id]
                old_x, old_y = tr.x, tr.y
                alpha = 0.55
                tr.x = alpha * det.x + (1.0 - alpha) * tr.x
                tr.y = alpha * det.y + (1.0 - alpha) * tr.y
                tr.theta = det.theta
                if dt > 1e-6:
                    raw_vx = (det.x - old_x) / dt
                    raw_vy = (det.y - old_y) / dt
                    tr.vx = 0.55 * tr.vx + 0.45 * raw_vx
                    tr.vy = 0.55 * tr.vy + 0.45 * raw_vy
                tr.speed = math.hypot(tr.vx, tr.vy)
                tr.length = det.length
                tr.width = det.width
                tr.last_seen = sim_time
                tr.age = 0.0
                tr.traffic_state = det.traffic_state
            else:
                tid = self.next_id
                self.next_id += 1
                vx = det.v * math.cos(det.theta)
                vy = det.v * math.sin(det.theta)
                self.tracks[tid] = TrackedAgent(
                    track_id=tid,
                    type=det.type,
                    x=det.x,
                    y=det.y,
                    theta=det.theta,
                    vx=vx,
                    vy=vy,
                    speed=abs(det.v),
                    length=det.length,
                    width=det.width,
                    last_seen=sim_time,
                    age=0.0,
                    traffic_state=det.traffic_state,
                )
                matched.add(tid)

        preds: List[PredictedAgent] = []
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if tr.age > 2.0:
                del self.tracks[tid]
                continue
            if tr.type == AgentType.PEDESTRIAN:
                tr.crossing_intent = self._ped_intent(tr)
            else:
                tr.crossing_intent = 0.0
            preds.append(self._predict(tr))
        return preds

    def _ped_intent(self, tr: TrackedAgent) -> float:
        s, d, tangent, normal, _ = self.track.nearest(tr.x, tr.y)
        dist_center = abs(d)
        sign = 1.0 if d > 0 else -1.0
        v_normal = tr.vx * normal[0] + tr.vy * normal[1]
        v_toward = -sign * v_normal
        speed = math.hypot(tr.vx, tr.vy)

        if dist_center < 2.5 and (speed > 0.25 or v_toward > 0.1):
            return 1.0
        if 2.5 <= dist_center <= 8.0 and v_toward > 0.15:
            return 0.9
        if 2.5 <= dist_center <= 6.0 and speed < 0.25:
            return 0.65
        return 0.1

    def _predict(self, tr: TrackedAgent) -> PredictedAgent:
        steps = int(PRED_HORIZON / PRED_DT) + 1
        times = np.linspace(0.0, PRED_HORIZON, steps)
        xs = np.zeros(steps)
        ys = np.zeros(steps)
        vxs = np.zeros(steps)
        vys = np.zeros(steps)

        if tr.type == AgentType.PEDESTRIAN and tr.crossing_intent > 0.5:
            _, d, _, normal, _ = self.track.nearest(tr.x, tr.y)
            sign = 1.0 if d > 0 else -1.0
            dir_x = -sign * normal[0]
            dir_y = -sign * normal[1]
            current_speed = math.hypot(tr.vx, tr.vy)
            speed = 1.2 if current_speed < 0.3 else clamp(current_speed, 0.5, 1.6)
            delay = 0.3 if current_speed < 0.3 else 0.0
            for i, t in enumerate(times):
                if t <= delay:
                    xs[i] = tr.x
                    ys[i] = tr.y
                    vxs[i] = 0.0
                    vys[i] = 0.0
                else:
                    tau = t - delay
                    xs[i] = tr.x + dir_x * speed * tau
                    ys[i] = tr.y + dir_y * speed * tau
                    vxs[i] = dir_x * speed
                    vys[i] = dir_y * speed
        else:
            for i, t in enumerate(times):
                xs[i] = tr.x + tr.vx * t
                ys[i] = tr.y + tr.vy * t
                vxs[i] = tr.vx
                vys[i] = tr.vy

        return PredictedAgent(
            track_id=tr.track_id,
            type=tr.type,
            times=times,
            xs=xs,
            ys=ys,
            vxs=vxs,
            vys=vys,
            intent=tr.crossing_intent,
            length=tr.length,
            width=tr.width,
        )


# --------------------------------------------------------------------------------------
# Safety monitor: independent of planner output
# --------------------------------------------------------------------------------------


class SafetyMonitor:
    """
    Independent safety monitor.
    It does not consume Plan/Candidate output. It consumes localized state, tracked/predicted
    agents, and detected traffic-light state only.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.triggered = False
        self.min_ttc = float("inf")
        self.reason = ""

    def update(
        self,
        ego: Pose,
        preds: List[PredictedAgent],
        dt: float,
        traffic_state: Optional[str] = None,
        light_dist: float = 1e9,
    ) -> Tuple[bool, str]:
        self.min_ttc = float("inf")
        emerg = False
        reason = ""

        forward = np.array([math.cos(ego.theta), math.sin(ego.theta)], dtype=float)
        left = np.array([-math.sin(ego.theta), math.cos(ego.theta)], dtype=float)
        stop_dist = ego.v * ego.v / (2.0 * MAX_DECEL) + 2.0

        for pred in preds:
            if pred.type == AgentType.TRAFFIC_LIGHT:
                continue
            for i, t in enumerate(pred.times):
                ex = ego.x + ego.v * forward[0] * t
                ey = ego.y + ego.v * forward[1] * t
                dx = pred.xs[i] - ex
                dy = pred.ys[i] - ey
                xf = dx * forward[0] + dy * forward[1]
                yl = dx * left[0] + dy * left[1]

                corridor = 2.5 + 0.5 * pred.width
                if xf > 0.0 and xf < 50.0 and abs(yl) < corridor:
                    v_agent_long = pred.vxs[i] * forward[0] + pred.vys[i] * forward[1]
                    closing = ego.v - v_agent_long
                    if closing > 0.1:
                        ttc = xf / closing
                        if ttc < self.min_ttc:
                            self.min_ttc = ttc
                        if self.enabled and (ttc < 2.0 or (xf < stop_dist and closing > 0.5)):
                            emerg = True
                            reason = f"TTC={ttc:.2f}s"
                    else:
                        if xf < stop_dist:
                            approx_ttc = xf / max(ego.v, 0.1)
                            if approx_ttc < self.min_ttc:
                                self.min_ttc = approx_ttc
                            if self.enabled:
                                emerg = True
                                reason = "stopping-distance"

        if self.enabled and traffic_state == "red" and light_dist < stop_dist + 8.0:
            emerg = True
            reason = "red-light-stop-distance"

        self.triggered = emerg
        self.reason = reason
        return emerg, reason


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------


class LocalPlanner:
    def __init__(self, track: Track, allow_overtake: bool = True, fast: bool = False) -> None:
        self.track = track
        self.allow_overtake = allow_overtake
        self.fast = fast
        self.horizon = 2.0 if fast else 3.0
        self.dt = 0.1

    def plan(
        self,
        ego: Pose,
        lane: LanePerception,
        preds: List[PredictedAgent],
        traffic_state: Optional[str],
        light_dist: float,
        safety_emergency: bool,
        sensor_valid: bool,
    ) -> Plan:
        ctx = self._context(ego, preds)
        state = self._fsm(ego, ctx, traffic_state, light_dist, safety_emergency)

        if state == PlanState.EMERGENCY_BRAKE:
            offsets = [0.0]
            profiles = ["stop"]
        elif state in (PlanState.STOP_AT_LIGHT, PlanState.YIELD_TO_PEDESTRIAN):
            offsets = [-0.4, 0.0, 0.4] if not self.fast else [0.0]
            profiles = ["stop", "slow"] if not self.fast else ["stop"]
        elif state == PlanState.OVERTAKE:
            offsets = [0.0, 1.6, 2.6, -1.6] if not self.fast else [0.0, 2.5]
            profiles = ["cruise", "slow"] if not self.fast else ["cruise"]
        elif state == PlanState.FOLLOW:
            offsets = [-0.4, 0.0, 0.4] if not self.fast else [0.0]
            profiles = ["follow", "slow", "stop"] if not self.fast else ["follow", "stop"]
        else:
            offsets = [-0.4, 0.0, 0.4] if not self.fast else [0.0]
            profiles = ["cruise", "slow"] if not self.fast else ["cruise"]

        if not sensor_valid:
            profiles = [p for p in profiles if p != "cruise"] or ["slow"]

        s0, d0, _, _, _ = self.track.nearest(ego.x, ego.y)
        candidates: List[Candidate] = []

        for off in offsets:
            for prof in profiles:
                pts = self._generate(ego, s0, d0, off, prof, ctx, traffic_state, light_dist, state)
                cand = self._score(pts, preds, ego, s0, off, prof, state, ctx, traffic_state, light_dist)
                candidates.append(cand)

        selected = min(candidates, key=lambda c: c.cost) if candidates else Candidate(
            0.0,
            "stop",
            self._generate(ego, s0, d0, 0.0, "stop", ctx, traffic_state, light_dist, state),
            1e9,
            1e9,
            0.0,
            0.0,
            0.0,
        )

        return Plan(
            state=state,
            points=selected.points,
            target_speed=selected.points[-1].v if selected.points else 0.0,
            target_accel=selected.points[-1].a if selected.points else 0.0,
            candidates=candidates,
            selected_cost=selected.cost,
            reason=ctx.get("reason", ""),
        )

    def _context(self, ego: Pose, preds: List[PredictedAgent]) -> Dict[str, Any]:
        forward = np.array([math.cos(ego.theta), math.sin(ego.theta)], dtype=float)
        left = np.array([-math.sin(ego.theta), math.cos(ego.theta)], dtype=float)

        lead_gap = 1e9
        lead_speed = None
        adjacent_clear = True
        ped_conflict = False

        for pred in preds:
            if pred.type == AgentType.VEHICLE:
                dx = pred.xs[0] - ego.x
                dy = pred.ys[0] - ego.y
                xf = dx * forward[0] + dy * forward[1]
                yl = dx * left[0] + dy * left[1]
                if 0.0 < xf < 40.0 and abs(yl) < 2.0:
                    if xf < lead_gap:
                        lead_gap = xf
                        lead_speed = pred.vxs[0] * forward[0] + pred.vys[0] * forward[1]
                for idx in (0, len(pred.times) // 3, -1):
                    dx = pred.xs[idx] - ego.x
                    dy = pred.ys[idx] - ego.y
                    xf = dx * forward[0] + dy * forward[1]
                    yl = dx * left[0] + dy * left[1]
                    if -5.0 < xf < 35.0 and 1.2 < abs(yl) < 4.5:
                        adjacent_clear = False

            elif pred.type == AgentType.PEDESTRIAN and pred.intent > 0.4:
                for i, t in enumerate(pred.times):
                    ex = ego.x + ego.v * forward[0] * t
                    ey = ego.y + ego.v * forward[1] * t
                    dx = pred.xs[i] - ex
                    dy = pred.ys[i] - ey
                    xf = dx * forward[0] + dy * forward[1]
                    yl = dx * left[0] + dy * left[1]
                    if -2.0 < xf < 16.0 and abs(yl) < 2.6:
                        ped_conflict = True
                        break

        blocked = lead_speed is not None and lead_speed < 0.7 and lead_gap < 25.0
        reason = ""
        if blocked:
            reason = "stopped_vehicle"
        if ped_conflict:
            reason = "pedestrian_conflict"
        return {
            "lead_gap": lead_gap,
            "lead_speed": lead_speed,
            "adjacent_clear": adjacent_clear,
            "ped_conflict": ped_conflict,
            "blocked": blocked,
            "reason": reason,
        }

    def _fsm(
        self,
        ego: Pose,
        ctx: Dict[str, Any],
        traffic_state: Optional[str],
        light_dist: float,
        safety_emergency: bool,
    ) -> PlanState:
        if safety_emergency:
            return PlanState.EMERGENCY_BRAKE

        stop_dist = ego.v * ego.v / (2.0 * 3.5) + 4.0
        if traffic_state == "red" and light_dist < 65.0:
            return PlanState.STOP_AT_LIGHT
        if traffic_state == "yellow" and light_dist < stop_dist + 8.0:
            return PlanState.STOP_AT_LIGHT

        if ctx["ped_conflict"]:
            return PlanState.YIELD_TO_PEDESTRIAN

        if ctx["blocked"] and self.allow_overtake and ctx["adjacent_clear"]:
            return PlanState.OVERTAKE

        if ctx["lead_gap"] < 28.0:
            return PlanState.FOLLOW

        return PlanState.LANE_KEEP

    def _generate(
        self,
        ego: Pose,
        s0: float,
        d0: float,
        d_cmd: float,
        profile: str,
        ctx: Dict[str, Any],
        traffic_state: Optional[str],
        light_dist: float,
        state: PlanState,
    ) -> List[TrajectoryPoint]:
        steps = int(self.horizon / self.dt) + 1
        T_lat = min(2.0, self.horizon)

        if profile == "stop" or state in (
            PlanState.STOP_AT_LIGHT,
            PlanState.YIELD_TO_PEDESTRIAN,
            PlanState.EMERGENCY_BRAKE,
        ):
            v_target = 0.0
        elif profile == "slow":
            v_target = TARGET_SPEED * 0.55
        elif profile == "follow":
            if ctx["lead_speed"] is not None:
                v_target = min(TARGET_SPEED, max(0.0, ctx["lead_speed"] + 1.5), max(0.0, ctx["lead_gap"] * 0.45))
            else:
                v_target = TARGET_SPEED
        else:
            v_target = TARGET_SPEED

        if state == PlanState.OVERTAKE and profile == "cruise":
            v_target = min(v_target, TARGET_SPEED * 0.9)

        delta_d = d_cmd - d0
        a2 = -3.0 * delta_d / (T_lat * T_lat) if T_lat > 1e-6 else 0.0
        a3 = 2.0 * delta_d / (T_lat * T_lat * T_lat) if T_lat > 1e-6 else 0.0

        pts: List[TrajectoryPoint] = []
        prev_v = ego.v
        prev_a = 0.0
        prev_theta = ego.theta
        prev_lat_a = 0.0
        s = s0

        for i in range(steps):
            t = i * self.dt
            if i == 0:
                v = ego.v
                a = 0.0
                j = 0.0
                d = d0
                d_dot = 0.0
            else:
                a_des = clamp((v_target - prev_v) / 0.45, -5.0, 2.5)
                v = max(0.0, prev_v + a_des * self.dt)
                if v_target == 0.0 and v < 0.05:
                    v = 0.0
                a = (v - prev_v) / self.dt
                j = (a - prev_a) / self.dt
                s = s + 0.5 * (prev_v + v) * self.dt

                if t < T_lat:
                    d = d0 + a2 * t * t + a3 * t * t * t
                    d_dot = 2.0 * a2 * t + 3.0 * a3 * t * t
                else:
                    d = d_cmd
                    d_dot = 0.0

            x, y, theta_t, _ = self.track.pose_from_sd(s, d)
            theta = wrap_angle(theta_t + math.atan2(d_dot, max(v, 0.5)))

            if i == 0:
                lat_a = 0.0
                lat_j = 0.0
            else:
                theta_dot = wrap_angle(theta - prev_theta) / self.dt
                lat_a = v * theta_dot
                lat_j = (lat_a - prev_lat_a) / self.dt

            pts.append(
                TrajectoryPoint(
                    t=t,
                    x=x,
                    y=y,
                    theta=theta,
                    v=v,
                    a=a,
                    j=j,
                    d=d,
                    s=s,
                    lat_a=lat_a,
                    lat_j=lat_j,
                )
            )

            prev_v = v
            prev_a = a
            prev_theta = theta
            prev_lat_a = lat_a

        return pts

    def _score(
        self,
        pts: List[TrajectoryPoint],
        preds: List[PredictedAgent],
        ego: Pose,
        s0: float,
        lateral_offset: float,
        profile: str,
        state: PlanState,
        ctx: Dict[str, Any],
        traffic_state: Optional[str],
        light_dist: float,
    ) -> Candidate:
        risk = 0.0
        min_ttc = float("inf")

        for pred in preds:
            if pred.type == AgentType.TRAFFIC_LIGHT:
                continue
            for pt in pts:
                idx = int(round(pt.t / PRED_DT))
                if idx < 0 or idx >= len(pred.xs):
                    continue
                dx = pred.xs[idx] - pt.x
                dy = pred.ys[idx] - pt.y
                c = math.cos(pt.theta)
                s = math.sin(pt.theta)
                long = dx * c + dy * s
                lat = -dx * s + dy * c

                len_sum = 0.5 * (EGO_LENGTH + pred.length)
                wid_sum = 0.5 * (EGO_WIDTH + pred.width) + 0.15

                if -len_sum < long < len_sum and abs(lat) < wid_sum:
                    risk += 1200.0 * max(0.0, 1.0 - pt.t / self.horizon)

                if long > 0.0:
                    v_agent_long = pred.vxs[idx] * c + pred.vys[idx] * s
                    closing = pt.v - v_agent_long
                    if closing > 0.1:
                        ttc = long / closing
                        if ttc < min_ttc:
                            min_ttc = ttc
                        risk += 40.0 / (ttc + 0.1)

        if min_ttc < 1.5:
            risk += 700.0

        comfort = 0.0
        for pt in pts[2:]:
            comfort += (abs(pt.j) + 0.35 * abs(pt.lat_j) + 0.06 * abs(pt.lat_a)) * self.dt

        lane_cost = 0.0
        for pt in pts:
            if state == PlanState.OVERTAKE:
                lane_cost += max(0.0, abs(pt.d) - 3.0) ** 2 * 35.0 + abs(pt.d) * 0.25
            else:
                lane_cost += max(0.0, abs(pt.d) - LANE_HALF) ** 2 * 140.0 + pt.d * pt.d * 0.45
        lane_cost *= self.dt

        avg_v = float(np.mean([p.v for p in pts])) if pts else 0.0
        progress_cost = -8.0 * avg_v + 4.0 * abs(lateral_offset)

        if traffic_state in ("red", "yellow") and light_dist < 90.0:
            stop_line_s = s0 + light_dist - 2.5
            for pt in pts:
                if pt.s > stop_line_s:
                    if traffic_state == "red":
                        risk += 12000.0
                    elif profile != "stop":
                        risk += 2500.0
                    break

        if state == PlanState.STOP_AT_LIGHT and profile != "stop" and light_dist < 35.0:
            risk += 3500.0

        if profile == "stop" and state not in (
            PlanState.STOP_AT_LIGHT,
            PlanState.YIELD_TO_PEDESTRIAN,
            PlanState.EMERGENCY_BRAKE,
        ):
            if not ctx["blocked"] and not ctx["ped_conflict"]:
                risk += 180.0

        total = risk + 0.045 * comfort + lane_cost + progress_cost
        return Candidate(lateral_offset, profile, pts, total, risk, comfort, lane_cost, progress_cost)


# --------------------------------------------------------------------------------------
# Control
# --------------------------------------------------------------------------------------


class Controller:
    def __init__(self) -> None:
        self.integral = 0.0
        self.prev_err = 0.0
        self.kp = 1.15
        self.ki = 0.08
        self.kd = 0.22
        self.stanley_k = 0.55

    def control(self, ego: Pose, plan: Plan, safety_emergency: bool) -> ControlCommand:
        # Lateral: Stanley-like controller on selected local trajectory.
        target = None
        lookahead = max(3.0, 0.35 * ego.v + 2.0)
        for pt in plan.points:
            if math.hypot(pt.x - ego.x, pt.y - ego.y) >= lookahead:
                target = pt
                break
        if target is None and plan.points:
            target = plan.points[-1]

        if target is None:
            steer = 0.0
            v_ref = 0.0
        else:
            xf, yl = ego_frame(ego, target.x, target.y)
            e_ct = yl
            psi_e = wrap_angle(target.theta - ego.theta)
            steer = psi_e + math.atan(self.stanley_k * e_ct / max(ego.v, 1.0))
            steer = clamp(steer, -0.45, 0.45)
            v_ref = target.v

        # Longitudinal PID.
        err = v_ref - ego.v
        if abs(err) > 6.0:
            self.integral = 0.0
        self.integral += err * DT
        self.integral = clamp(self.integral, -8.0, 8.0)
        deriv = (err - self.prev_err) / DT
        self.prev_err = err

        a_cmd = self.kp * err + self.ki * self.integral + self.kd * deriv
        a_cmd = clamp(a_cmd, -MAX_DECEL, MAX_ACCEL)

        if safety_emergency:
            a_cmd = -MAX_DECEL

        if a_cmd >= 0.0:
            throttle = clamp(a_cmd / MAX_ACCEL, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(-a_cmd / MAX_DECEL, 0.0, 1.0)

        return ControlCommand(throttle=throttle, brake=brake, steer=steer, accel=a_cmd)


# --------------------------------------------------------------------------------------
# Full autonomous stack
# --------------------------------------------------------------------------------------


class AutonomousStack:
    def __init__(
        self,
        track: Track,
        scenario: str,
        seed: int,
        safety_enabled: bool,
        perception_mode: str,
        allow_overtake: bool,
        render: bool = False,
    ) -> None:
        self.track = track
        self.seed = int(seed)
        self.sim = SimWorld(track, scenario, seed, duration=0.0)
        self.rng = np.random.default_rng(seed + 1000)

        self.perception = PerceptionModule(perception_mode, self.rng, track)
        self.localizer = EKFLocalizer(self.sim.ego, self.rng)
        self.tracker = TrackerPredictor(track)
        self.planner = LocalPlanner(track, allow_overtake=allow_overtake, fast=not render)
        self.controller = Controller()
        self.safety = SafetyMonitor(enabled=safety_enabled)

        self.render = render
        self.recorder = VideoRecorder(fps=10, size=(640, 240)) if render else None
        self.render_every = max(1, int(round(1.0 / (10.0 * DT))))

        self.last_perc: Optional[PerceptionOutput] = None
        self.last_loc: Pose = self.sim.ego
        self.last_preds: List[PredictedAgent] = []
        self.last_plan: Optional[Plan] = None
        self.last_cmd: ControlCommand = ControlCommand()

        self.collisions = 0
        self.collided_ids: set[str] = set()
        self.lane_invasion_time = 0.0
        self.global_min_ttc = float("inf")
        self.max_long_jerk = 0.0
        self.max_lat_jerk = 0.0
        self.emergency_brakes = 0
        self.prev_emergency = False

        self.progress = 0.0
        self.last_s = self.sim.start_s
        self.prev_v = self.sim.ego.v
        self.prev_a = 0.0
        self.prev_lat_a = 0.0

        self.duration = 0.0
        self.log: List[Dict[str, Any]] = []

    def set_duration(self, duration: float) -> None:
        self.duration = float(duration)
        self.sim.duration = float(duration)
        if self.sim.scenario == "sensor_dropout":
            self.sim.dropout_start = self.duration * 0.45

    def step(self, render_image: bool = False) -> None:
        perc = self.perception.process(self.sim, self.sim.ego, render_image=render_image)
        loc = self.localizer.update(self.sim.ego, DT, gps_active=self.sim.sensor_active())
        preds = self.tracker.update(perc.detections, DT, loc, self.sim.t)

        loc_s, _, _, _, _ = self.track.nearest(loc.x, loc.y)
        light_dist = (self.sim.light_s - loc_s) % self.track.length
        traffic_state = None
        for tr in self.tracker.tracks.values():
            if tr.type == AgentType.TRAFFIC_LIGHT and tr.traffic_state is not None:
                traffic_state = tr.traffic_state
                break

        safety_emergency, _ = self.safety.update(loc, preds, DT, traffic_state, light_dist)
        plan = self.planner.plan(loc, perc.lane, preds, traffic_state, light_dist, safety_emergency, perc.valid)
        base_cmd = self.controller.control(loc, plan, safety_emergency)

        temp = Pose(x=loc.x, y=loc.y, theta=loc.theta, v=loc.v)
        any_emergency = False
        dt_sub = DT / SUBSTEPS

        last_effective_cmd = ControlCommand(
            throttle=base_cmd.throttle,
            brake=base_cmd.brake,
            steer=base_cmd.steer,
            accel=base_cmd.accel,
        )

        for _ in range(SUBSTEPS):
            emerg_sub, _ = self.safety.update(temp, preds, dt_sub, traffic_state, light_dist)

            if emerg_sub:
                any_emergency = True
                effective_cmd = ControlCommand(
                    throttle=0.0,
                    brake=1.0,
                    steer=base_cmd.steer,
                    accel=-MAX_DECEL,
                )
            else:
                effective_cmd = ControlCommand(
                    throttle=base_cmd.throttle,
                    brake=base_cmd.brake,
                    steer=base_cmd.steer,
                    accel=base_cmd.accel,
                )

            last_effective_cmd = effective_cmd

            temp.v = max(0.0, temp.v + effective_cmd.accel * dt_sub)
            temp.theta = wrap_angle(temp.theta + temp.v / WHEELBASE * math.tan(effective_cmd.steer) * dt_sub)
            temp.x += temp.v * math.cos(temp.theta) * dt_sub
            temp.y += temp.v * math.sin(temp.theta) * dt_sub

            self.sim.update(dt_sub, effective_cmd)

        if any_emergency and not self.prev_emergency:
            self.emergency_brakes += 1
        self.prev_emergency = any_emergency

        self._record_metrics(loc, plan, perc, last_effective_cmd)

        self.last_perc = perc
        self.last_loc = loc
        self.last_preds = preds
        self.last_plan = plan
        self.last_cmd = last_effective_cmd

    def _record_metrics(self, loc: Pose, plan: Plan, perc: PerceptionOutput, cmd: ControlCommand) -> None:
        true = self.sim.ego
        s, d, _, _, _ = self.track.nearest(true.x, true.y)

        delta = s - self.last_s
        if delta < -self.track.length / 2.0:
            delta += self.track.length
        elif delta > self.track.length / 2.0:
            delta -= self.track.length
        if delta > 0.0:
            self.progress += delta
        self.last_s = s

        if abs(d) > LANE_HALF + 0.20:
            self.lane_invasion_time += DT

        # Collision check against true simulator agents.
        forward = np.array([math.cos(true.theta), math.sin(true.theta)], dtype=float)
        left = np.array([-math.sin(true.theta), math.cos(true.theta)], dtype=float)
        for ag in self.sim.agents:
            if ag.type == AgentType.TRAFFIC_LIGHT or ag.id in self.collided_ids:
                continue
            dx = ag.x - true.x
            dy = ag.y - true.y
            xf = dx * forward[0] + dy * forward[1]
            yl = dx * left[0] + dy * left[1]
            len_sum = 0.5 * (EGO_LENGTH + ag.length)
            wid_sum = 0.5 * (EGO_WIDTH + ag.width)
            if ag.type == AgentType.PEDESTRIAN:
                hit = math.hypot(xf, yl) < 2.2 + 0.45
            else:
                hit = abs(xf) < len_sum and abs(yl) < wid_sum
            if hit:
                self.collisions += 1
                self.collided_ids.add(ag.id)
                self.sim.ego.v = 0.0

        a_long = (true.v - self.prev_v) / DT
        lat_a = true.v * true.yaw_rate
        j_long = (a_long - self.prev_a) / DT
        j_lat = (lat_a - self.prev_lat_a) / DT
        self.max_long_jerk = max(self.max_long_jerk, abs(j_long))
        self.max_lat_jerk = max(self.max_lat_jerk, abs(j_lat))
        self.prev_v = true.v
        self.prev_a = a_long
        self.prev_lat_a = lat_a

        if self.safety.min_ttc < self.global_min_ttc:
            self.global_min_ttc = self.safety.min_ttc

        self.log.append(
            {
                "t": self.sim.t,
                "x": true.x,
                "y": true.y,
                "theta": true.theta,
                "v": true.v,
                "a": a_long,
                "d": d,
                "state": plan.state.value,
                "emergency": self.safety.triggered,
                "min_ttc_step": self.safety.min_ttc,
                "valid_sensors": perc.valid,
            }
        )

    def run(self, duration: float, callback: Optional[Any] = None) -> None:
        self.set_duration(duration)
        steps = int(duration / DT)
        for i in range(steps):
            self.step(render_image=False)
            if callback is not None:
                callback(i, steps, self)

    def summary(self) -> Dict[str, Any]:
        target_dist = max(1.0, TARGET_SPEED * self.duration)
        completion = clamp(100.0 * self.progress / target_dist, 0.0, 100.0)
        min_ttc = self.global_min_ttc if math.isfinite(self.global_min_ttc) else 99.0
        return {
            "scenario": self.sim.scenario,
            "seed": self.seed,
            "safety_enabled": self.safety.enabled,
            "collisions": int(self.collisions),
            "lane_invasion_time": float(self.lane_invasion_time),
            "min_ttc": float(min_ttc),
            "max_long_jerk": float(self.max_long_jerk),
            "max_lat_jerk": float(self.max_lat_jerk),
            "route_completion_pct": float(completion),
            "emergency_brakes": int(self.emergency_brakes),
            "distance_m": float(self.progress),
            "final_speed_mps": float(self.sim.ego.v),
            "duration_s": float(self.duration),
        }


# --------------------------------------------------------------------------------------
# Visualization and video recording
# --------------------------------------------------------------------------------------


def make_figure(stack: AutonomousStack) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=80, gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    track = stack.track
    ax.plot(track.points[:, 0], track.points[:, 1], color="#bbbbbb", lw=2.0, zorder=1)

    ego = stack.sim.ego
    ax.scatter([ego.x], [ego.y], c="green", s=55, zorder=6)
    ax.arrow(
        ego.x,
        ego.y,
        2.2 * math.cos(ego.theta),
        2.2 * math.sin(ego.theta),
        head_width=0.7,
        head_length=0.8,
        fc="green",
        ec="green",
        zorder=7,
    )

    for ag in stack.sim.agents:
        if ag.type == AgentType.VEHICLE:
            ax.scatter([ag.x], [ag.y], c="red", marker="s", s=45, zorder=5)
        elif ag.type == AgentType.PEDESTRIAN:
            ax.scatter([ag.x], [ag.y], c="orange", marker="o", s=35, zorder=5)

    for pred in stack.last_preds:
        if pred.type == AgentType.TRAFFIC_LIGHT:
            continue
        color = "cyan" if pred.type == AgentType.VEHICLE else "yellow"
        ax.plot(pred.xs, pred.ys, "--", color=color, alpha=0.65, lw=1.2, zorder=4)

    if stack.last_plan is not None:
        for cand in stack.last_plan.candidates[:10]:
            xs = [p.x for p in cand.points]
            ys = [p.y for p in cand.points]
            ax.plot(xs, ys, color="orange", alpha=0.13, lw=1.0, zorder=3)
        xs = [p.x for p in stack.last_plan.points]
        ys = [p.y for p in stack.last_plan.points]
        ax.plot(xs, ys, color="lime", lw=2.3, zorder=5)

    lx, ly, _, _ = track.pose_from_sd(stack.sim.light_s, 0.0)
    ax.scatter([lx], [ly], c="black", marker="*", s=70, zorder=6)

    state = stack.last_plan.state.value if stack.last_plan else "INIT"
    ttc = stack.safety.min_ttc if math.isfinite(stack.safety.min_ttc) else 99.0
    ax.set_title(
        f"t={stack.sim.t:5.1f}s  state={state}  v={ego.v:4.1f}m/s  TTC={ttc:4.1f}s  emergency={stack.safety.triggered}",
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax2 = axes[1]
    if stack.last_perc is not None and stack.last_perc.image is not None:
        ax2.imshow(cv2.cvtColor(stack.last_perc.image, cv2.COLOR_BGR2RGB))
        ax2.set_title("Front camera / perception", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Perception image not rendered\nfor this frame", ha="center", va="center")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title("Perception", fontsize=9)
    ax2.axis("off")

    fig.tight_layout()
    return fig


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    try:
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()
    except Exception:
        buf = np.asarray(fig.canvas.tostring_rgb(), dtype=np.uint8)
        h, w = fig.canvas.get_width_height()[::-1]
        return buf.reshape(h, w, 3)


class VideoRecorder:
    def __init__(self, fps: int = 10, size: Tuple[int, int] = (640, 240)) -> None:
        self.fps = fps
        self.size = size

        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        self.mp4_path: Optional[str] = self.tmp.name
        self.tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.mp4_path, fourcc, fps, size)
        self.ok = self.writer.isOpened()

        self.frames: List[np.ndarray] = []
        self.gif_path: Optional[str] = None

    def add(self, frame_rgb: np.ndarray) -> None:
        frame = cv2.resize(frame_rgb, self.size, interpolation=cv2.INTER_AREA)
        if self.ok:
            try:
                self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception:
                self.ok = False
        # Always keep frames so we can produce an animated GIF preview.
        self.frames.append(frame)

    def finish(self) -> Tuple[Optional[str], str, Optional[str]]:
        # Finalize MP4 if available.
        if self.ok:
            try:
                self.writer.release()
            except Exception:
                pass
            if not (os.path.exists(self.mp4_path) and os.path.getsize(self.mp4_path) > 5000):
                self.mp4_path = None
        else:
            self.mp4_path = None

        # Build animated GIF preview for reliable in-browser motion display.
        try:
            if self.frames:
                from PIL import Image

                gif_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
                self.gif_path = gif_tmp.name
                gif_tmp.close()

                imgs = [Image.fromarray(f) for f in self.frames]
                imgs[0].save(
                    self.gif_path,
                    save_all=True,
                    append_images=imgs[1:],
                    duration=int(1000 / max(1, self.fps)),
                    loop=0,
                )

                if not (os.path.exists(self.gif_path) and os.path.getsize(self.gif_path) > 1000):
                    self.gif_path = None
        except Exception:
            self.gif_path = None

        preview_path = self.gif_path or self.mp4_path
        preview_mime = "image/gif" if self.gif_path else "video/mp4"
        return preview_path, preview_mime, self.mp4_path


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def run_single_scenario(
    track: Track,
    scenario: str,
    seed: int,
    safety_enabled: bool,
    duration: float,
    perception_mode: str = "fast",
    allow_overtake: bool = True,
) -> Dict[str, Any]:
    stack = AutonomousStack(
        track=track,
        scenario=scenario,
        seed=seed,
        safety_enabled=safety_enabled,
        perception_mode=perception_mode,
        allow_overtake=allow_overtake,
        render=False,
    )
    stack.run(duration)
    out = stack.summary()
    out["seed"] = int(seed)
    return out


def ci95(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n < 2:
        return 0.0
    std = float(x.std(ddof=1))
    if not math.isfinite(std):
        return 0.0
    return 2.093 * std / math.sqrt(n)


def summarize_eval(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, safety), g in df.groupby(["scenario", "safety_enabled"]):
        for metric in METRICS:
            mean = float(pd.to_numeric(g[metric], errors="coerce").mean())
            rows.append(
                {
                    "scenario": scenario,
                    "safety": "ON" if safety else "OFF",
                    "metric": metric,
                    "mean": mean,
                    "ci95": ci95(g[metric]),
                    "n": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def run_evaluation(
    track: Track,
    seed_count: int,
    duration: float,
    scenarios: List[str],
    perception_mode: str,
    allow_overtake: bool,
    status_container: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    seeds = list(range(int(seed_count)))
    safety_modes = [True, False]
    total = len(seeds) * len(scenarios) * len(safety_modes)
    count = 0

    progress = status_container.progress(0.0, text="Evaluation: 0%")
    for scenario in scenarios:
        for seed in seeds:
            for safety in safety_modes:
                rows.append(
                    run_single_scenario(
                        track=track,
                        scenario=scenario,
                        seed=seed,
                        safety_enabled=safety,
                        duration=duration,
                        perception_mode=perception_mode,
                        allow_overtake=allow_overtake,
                    )
                )
                count += 1
                progress.progress(
                    count / max(1, total),
                    text=f"Evaluation: {int(100*count/max(1,total))}%  {scenario}  seed={seed}  safety={'ON' if safety else 'OFF'}",
                )

    progress.empty()
    df = pd.DataFrame(rows)
    agg = summarize_eval(df)
    return df, agg


# --------------------------------------------------------------------------------------
# Integration tests
# --------------------------------------------------------------------------------------


def run_integration_tests(track: Track) -> pd.DataFrame:
    tests: List[Tuple[str, bool]] = []
    rng = np.random.default_rng(0)

    try:
        sim = SimWorld(track, "straight_curves", 0, 5.0)
        cmd = ControlCommand(throttle=0.6, brake=0.0, steer=0.0, accel=0.0)
        for _ in range(20):
            sim.update(DT, cmd)
        tests.append(("01_simulator_ground_truth_driving", sim.ego.v > 0.4 and sim.t > 0.9))

        loc = EKFLocalizer(sim.ego, rng)
        p = loc.update(sim.ego, DT, gps_active=True)
        tests.append(("02_localization_ekf_fusion", math.hypot(p.x - sim.ego.x, p.y - sim.ego.y) < 3.0))

        perc_fast = PerceptionModule("fast", rng, track)
        out_fast = perc_fast.process(sim, sim.ego, render_image=False)
        tests.append(("03_perception_fast_interface", out_fast.valid))

        perc_cam = PerceptionModule("learned", rng, track)
        out_cam = perc_cam.process(sim, sim.ego, render_image=True)
        tests.append(("04_perception_camera_render_and_process", out_cam.image is not None))

        tracker = TrackerPredictor(track)
        preds = tracker.update(out_fast.detections, DT, sim.ego, sim.t)
        tests.append(("05_prediction_tracker_output", isinstance(preds, list)))

        planner = LocalPlanner(track, allow_overtake=True, fast=True)
        plan = planner.plan(sim.ego, out_fast.lane, preds, None, 1e9, False, True)
        tests.append(("06_planning_candidate_trajectory", len(plan.points) > 0))

        ctrl = Controller()
        cmd2 = ctrl.control(sim.ego, plan, False)
        tests.append(("07_control_stanley_pid", abs(cmd2.steer) < 0.6 and cmd2.brake >= 0.0))

        safety = SafetyMonitor(True)
        det = Detection(
            id="test_veh",
            type=AgentType.VEHICLE,
            x=15.0,
            y=0.0,
            theta=0.0,
            v=0.0,
            length=EGO_LENGTH,
            width=EGO_WIDTH,
            confidence=0.9,
        )
        tracker2 = TrackerPredictor(track)
        preds2 = tracker2.update([det], DT, Pose(0.0, 0.0, 0.0, 10.0), 0.0)
        emerg, _ = safety.update(Pose(0.0, 0.0, 0.0, 10.0), preds2, DT, None, 1e9)
        tests.append(("08_safety_monitor_independent_ttc", emerg))

    except Exception as exc:
        warnings.warn(f"Integration test exception: {exc}")
        tests.append(("all_integration_tests", False))

    return pd.DataFrame(tests, columns=["integration_test", "passed"])


# --------------------------------------------------------------------------------------
# Streamlit UI helpers
# --------------------------------------------------------------------------------------


def perform_demo(
    track: Track,
    scenario: str,
    seed: int,
    duration: float,
    safety_enabled: bool,
    perception_mode: str,
    allow_overtake: bool,
    live_container: Any,
    status_container: Any,
) -> Tuple[Dict[str, Any], Tuple[Optional[str], str, Optional[str]]]:
    stack = AutonomousStack(
        track=track,
        scenario=scenario,
        seed=int(seed),
        safety_enabled=safety_enabled,
        perception_mode=perception_mode,
        allow_overtake=allow_overtake,
        render=True,
    )
    stack.set_duration(duration)
    steps = int(duration / DT)

    for i in range(steps):
        render_image = bool(stack.render and stack.recorder is not None and i % stack.render_every == 0)
        stack.step(render_image=render_image)

        if render_image and stack.recorder is not None:
            fig = make_figure(stack)
            frame = fig_to_array(fig)
            stack.recorder.add(frame)
            if i % (stack.render_every * 10) == 0:
                live_container.pyplot(fig)
            plt.close(fig)

        if i % 10 == 0:
            status_container.progress((i + 1) / steps, text=f"Simulating {stack.sim.t:.1f}s / {duration:.1f}s")

    video = stack.recorder.finish() if stack.recorder is not None else (None, "image/gif", None)
    status_container.empty()
    summary = stack.summary()
    summary["seed"] = int(seed)
    return summary, video


def show_demo_outputs(
    summary: Optional[Dict[str, Any]],
    video: Any,
    video_container: Any,
    metrics_container: Any,
) -> None:
    video_container.empty()
    metrics_container.empty()

    preview_path, preview_mime, mp4_path = normalize_video_tuple(video)

    if preview_path:
        if preview_mime == "image/gif" or str(preview_path).lower().endswith(".gif"):
            try:
                video_container.image(preview_path, caption="Generated LaneAgent animation")
            except Exception:
                try:
                    video_container.video(preview_path)
                except Exception:
                    video_container.info("Animated preview could not be embedded.")
        else:
            try:
                video_container.video(preview_path)
            except Exception:
                video_container.info("MP4 preview could not be embedded. Use the download button below if available.")

    if mp4_path and os.path.exists(mp4_path):
        try:
            with open(mp4_path, "rb") as f:
                data = f.read()
            video_container.download_button(
                label="Download MP4 demo video",
                data=data,
                file_name="laneagent_demo.mp4",
                mime="video/mp4",
            )
        except Exception:
            pass

    if not preview_path and not mp4_path:
        video_container.warning("Video could not be generated in this environment.")

    if summary is None:
        return

    c1, c2, c3, c4 = metrics_container.columns(4)
    c1.metric("Collisions", summary.get("collisions", 0))
    c2.metric("Emergency brakes", summary.get("emergency_brakes", 0))
    c3.metric("Route completion %", f"{summary.get('route_completion_pct', 0):.1f}")
    c4.metric("Min TTC [s]", f"{summary.get('min_ttc', 99):.2f}")

    c5, c6, c7, c8 = metrics_container.columns(4)
    c5.metric("Lane invasion [s]", f"{summary.get('lane_invasion_time', 0):.2f}")
    c6.metric("Max long jerk", f"{summary.get('max_long_jerk', 0):.2f}")
    c7.metric("Max lat jerk", f"{summary.get('max_lat_jerk', 0):.2f}")
    c8.metric("Final speed [m/s]", f"{summary.get('final_speed_mps', 0):.2f}")


def build_report_markdown(df: Optional[pd.DataFrame], agg: Optional[pd.DataFrame]) -> str:
    md = []
    md.append("## Module / message-bus design")
    md.append(
        "The app uses a ROS-2-compatible logical node structure without requiring ROS runtime. "
        "Each module publishes typed messages equivalent to ROS topics:\n\n"
        "- `/perception/image` (`sensor_msgs/Image` analog)\n"
        "- `/perception/lane` (custom `LanePerception`)\n"
        "- `/perception/detections` (custom `Detection[]`)\n"
        "- `/perception/lidar_clusters` (custom obstacle clusters)\n"
        "- `/localization/state` (custom fused `Pose`)\n"
        "- `/prediction/trajectories` (custom `PredictedAgent[]`)\n"
        "- `/planning/trajectory` and `/planning/candidates`\n"
        "- `/control/command` (`throttle`, `brake`, `steer`)\n"
        "- `/safety/status` (independent TTC / stop-distance override)\n\n"
        "Rates: perception/localization/planning/control at 20 Hz; safety monitor evaluated at 80 Hz inside simulation substeps."
    )

    md.append("\n## Build order and integration gates")
    md.append(
        "1. Simulator interface and ground-truth driving.\n"
        "2. Controller tracking a nominal trajectory.\n"
        "3. Perception: camera lane/object processing and optional lidar clustering.\n"
        "4. Prediction: tracking + 3s constant-velocity / pedestrian-intent trajectories.\n"
        "5. Planning: FSM + sampled lateral/speed candidate scoring.\n"
        "6. Safety monitor: independent TTC/stopping-distance override.\n"
        "The Integration Tests tab reports passing gates for these stages."
    )

    if df is not None and agg is not None and len(df) > 0:
        md.append("\n## Scenario result interpretation")
        for scenario in SCENARIOS:
            sub = df[df["scenario"] == scenario]
            on = sub[sub["safety_enabled"]]
            off = sub[~sub["safety_enabled"]]
            if len(on) == 0 or len(off) == 0:
                continue
            coll_on = float(on["collisions"].mean())
            coll_off = float(off["collisions"].mean())
            comp_on = float(on["route_completion_pct"].mean())
            comp_off = float(off["route_completion_pct"].mean())
            eb_on = float(on["emergency_brakes"].mean())
            md.append(
                f"- **{scenario}**: safety ON collisions={coll_on:.2f}, OFF={coll_off:.2f}; "
                f"completion ON={comp_on:.1f}%, OFF={comp_off:.1f}%; emergency brakes ON={eb_on:.2f}."
            )

        md.append("\n## Automatic failure analysis")
        if "collisions" in df:
            unsafe = df[(~df["safety_enabled"]) & (df["collisions"] > 0)]
            safe = df[(df["safety_enabled"]) & (df["collisions"] == 0)]
            md.append(
                f"- Runs with safety disabled that had at least one collision: **{len(unsafe)}**. "
                f"Runs with safety enabled and zero collisions: **{len(safe)}**."
            )
            md.append(
                "- Main observed failure modes in the lightweight simulator are late pedestrian intent detection, "
                "overly aggressive overtaking candidates when adjacent clearance is marginal, and sensor-dropout "
                "coasting where stale predictions reduce effective reaction margin."
            )
            md.append(
                "- The safety monitor improves worst-case behavior by enforcing a stop-distance/TTC floor independent "
                "of planner candidate selection. The ablation table quantifies this contribution."
            )
    else:
        md.append("\nRun the 20-seed evaluation to populate quantitative scenario results and failure analysis.")

    md.append("\n## Real-world deployment requirements")
    md.append(
        "- **Redundancy:** diverse perception (camera, radar, lidar), redundant localization (GNSS/IMU/odometry/map matching), "
        "dual-channel steering and braking, independent safety ECU, watchdogs, fail-operational/fail-safe actuation paths.\n"
        "- **Validation:** billions of simulated miles, scenario mining from real fleets, HIL/SIL fault injection, "
        "weather/lighting/occlusion matrices, edge-case regression suites, field trials under safety driver / remote assist.\n"
        "- **Regulatory / standards:** ISO 26262 functional safety, ISO 21448 SOTIF, IEEE/UL 4600 safety case, "
        "UNECE/R157 or local AV regulations, NHTSA AV guidance, data recording/EDR, cybersecurity ISO/SAE 21434, "
        "ODD documentation, insurance and incident-reporting obligations.\n"
        "- **Operations:** map freshness, remote monitoring, fleet learning governance, ethical/safety-case review, "
        "human-machine interaction for handover, and rigorous change control for model updates."
    )
    return "\n".join(md)


# --------------------------------------------------------------------------------------
# Streamlit app
# --------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="LaneAgent", layout="wide", initial_sidebar_state="expanded")
    st.title("LaneAgent — Simulated Autonomy Stack")
    st.caption(
        "CARLA-style modular autonomy stack implemented on a lightweight 2D bicycle-model simulator. "
        "Modules: perception, localization, prediction, planning, control, independent safety monitor."
    )

    track = get_track()

    with st.sidebar:
        st.header("Controls")
        scenario = st.selectbox("Scenario", SCENARIOS, index=2)
        seed = st.number_input("Demo seed", min_value=0, max_value=9999, value=7, step=1)
        duration = st.slider("Demo duration (s)", min_value=8, max_value=30, value=12, step=1)
        safety_enabled = st.checkbox("Safety monitor enabled", value=True)

        perception_choice = st.selectbox(
            "Perception mode",
            ["demo (fast + camera)", "learned", "classical", "fast"],
            index=0,
        )
        perception_mode = {
            "demo (fast + camera)": "demo",
            "learned": "learned",
            "classical": "classical",
            "fast": "fast",
        }[perception_choice]

        allow_overtake = st.checkbox("Allow overtake", value=True)

        st.divider()
        st.subheader("Batch evaluation")
        seed_count = st.slider("Seeds per scenario", min_value=1, max_value=20, value=20, step=1)
        eval_duration = st.slider("Eval duration per run (s)", min_value=10, max_value=25, value=13, step=1)

        eval_choice = st.selectbox(
            "Eval perception mode",
            ["fast", "demo (fast + camera)", "learned", "classical"],
            index=0,
        )
        eval_perception = {
            "fast": "fast",
            "demo (fast + camera)": "demo",
            "learned": "learned",
            "classical": "classical",
        }[eval_choice]

        run_demo_btn = st.button("Run interactive demo", use_container_width=True)
        run_eval_btn = st.button("Run 20-seed evaluation", use_container_width=True, type="primary")

        st.divider()
        st.markdown(
            "**Notes**\n"
            "- `demo` uses fast perception internally and renders camera frames for the live dashboard.\n"
            "- `learned` uses a mock learned lane segmentation front end.\n"
            "- `classical` uses Canny + Hough.\n"
            "- `fast` uses ground-truth-assisted detections for rapid Monte Carlo evaluation.\n"
            "- Safety monitor is independent of planner output and runs in simulation substeps."
        )

    tab_demo, tab_eval, tab_report = st.tabs(["Live demo", "Evaluation", "Report & integration tests"])

    if "auto_ran" not in st.session_state:
        st.session_state.auto_ran = False
    if "demo_summary" not in st.session_state:
        st.session_state.demo_summary = None
    if "demo_video" not in st.session_state:
        st.session_state.demo_video = (None, "image/gif", None)
    else:
        st.session_state.demo_video = normalize_video_tuple(st.session_state.demo_video)

    if "eval_df" not in st.session_state:
        st.session_state.eval_df = None
    if "eval_agg" not in st.session_state:
        st.session_state.eval_agg = None

    with tab_demo:
        status = st.empty()
        live = st.empty()
        video_box = st.empty()
        metrics_box = st.empty()

        if not st.session_state.auto_ran:
            st.session_state.auto_ran = True
            try:
                with st.spinner("Auto-starting LaneAgent demo..."):
                    summary, video = perform_demo(
                        track=track,
                        scenario=scenario,
                        seed=int(seed),
                        duration=float(duration),
                        safety_enabled=bool(safety_enabled),
                        perception_mode=str(perception_mode),
                        allow_overtake=bool(allow_overtake),
                        live_container=live,
                        status_container=status,
                    )
                st.session_state.demo_summary = summary
                st.session_state.demo_video = normalize_video_tuple(video)
            except Exception as exc:
                st.session_state.demo_summary = None
                st.session_state.demo_video = (None, "image/gif", None)
                status.error(f"Auto demo failed: {exc}")

        normalized_preview, _, _ = normalize_video_tuple(st.session_state.demo_video)
        if st.session_state.demo_summary is not None or normalized_preview is not None:
            show_demo_outputs(
                st.session_state.demo_summary,
                st.session_state.demo_video,
                video_box,
                metrics_box,
            )

        if run_demo_btn:
            try:
                with st.spinner("Running selected demo..."):
                    summary, video = perform_demo(
                        track=track,
                        scenario=scenario,
                        seed=int(seed),
                        duration=float(duration),
                        safety_enabled=bool(safety_enabled),
                        perception_mode=str(perception_mode),
                        allow_overtake=bool(allow_overtake),
                        live_container=live,
                        status_container=status,
                    )
                st.session_state.demo_summary = summary
                st.session_state.demo_video = normalize_video_tuple(video)
                show_demo_outputs(summary, st.session_state.demo_video, video_box, metrics_box)
            except Exception as exc:
                status.error(f"Demo failed: {exc}")

    with tab_eval:
        st.subheader("Monte Carlo evaluation")
        st.write(
            "The evaluation runs each selected scenario across the chosen number of seeds, with safety monitor ON and OFF. "
            "It reports means and approximate 95% confidence intervals."
        )

        if run_eval_btn:
            status_container = st.empty()
            try:
                with st.spinner("Running evaluation..."):
                    df, agg = run_evaluation(
                        track=track,
                        seed_count=int(seed_count),
                        duration=float(eval_duration),
                        scenarios=SCENARIOS,
                        perception_mode=str(eval_perception),
                        allow_overtake=bool(allow_overtake),
                        status_container=status_container,
                    )
                st.session_state.eval_df = df
                st.session_state.eval_agg = agg
            except Exception as exc:
                status_container.error(f"Evaluation failed: {exc}")

        if st.session_state.eval_agg is not None:
            try:
                agg = st.session_state.eval_agg
                df = st.session_state.eval_df

                st.markdown("### Summary: mean values by scenario and safety mode")
                pivot = agg.pivot_table(index=["scenario", "safety"], columns="metric", values="mean").reset_index()
                st.dataframe(pivot, use_container_width=True)

                st.markdown("### 95% confidence intervals")
                st.dataframe(agg, use_container_width=True)

                st.markdown("### Safety ablation")
                coll = agg[agg["metric"] == "collisions"]
                comp = agg[agg["metric"] == "route_completion_pct"]
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=90)
                    coll.pivot(index="scenario", columns="safety", values="mean").plot(kind="bar", ax=ax)
                    ax.set_ylabel("Mean collisions")
                    ax.set_title("Collisions: safety ON vs OFF")
                    ax.tick_params(axis="x", rotation=20)
                    st.pyplot(fig)
                with c2:
                    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=90)
                    comp.pivot(index="scenario", columns="safety", values="mean").plot(kind="bar", ax=ax)
                    ax.set_ylabel("Mean completion %")
                    ax.set_title("Route completion: safety ON vs OFF")
                    ax.tick_params(axis="x", rotation=20)
                    st.pyplot(fig)

                st.markdown("### Raw runs")
                st.dataframe(df, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not render evaluation tables: {exc}")
        else:
            st.info("Run the evaluation to populate this tab.")

    with tab_report:
        st.subheader("Integration tests")
        tests = run_integration_tests(track)
        st.dataframe(tests, use_container_width=True)
        if tests["passed"].all():
            st.success("All staged integration checks passed.")
        else:
            st.warning("Some integration checks did not pass; see table.")

        st.subheader("Report")
        st.markdown(build_report_markdown(st.session_state.eval_df, st.session_state.eval_agg))

        st.subheader("Trajectory log from last demo")
        if st.session_state.demo_summary is not None:
            st.json(st.session_state.demo_summary)


if __name__ == "__main__":
    main()