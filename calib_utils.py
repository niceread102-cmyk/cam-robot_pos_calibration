"""Shared helpers: config (YAML + dotted command-line overrides), xArm pose <-> 4x4 matrix, rigid fit, I/O.

Naming: T_a_b = pose of frame b in frame a (maps points in b to a: p_a = T_a_b @ p_b).
Units everywhere: m and deg (robot_lite6 / camera_photoneo convert the devices' mm at their boundary).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config.yaml")


# ---------------------------------------------------------------- config

def load_config(path: str | None = None, overrides=()) -> dict:
    path = os.path.abspath(os.path.expanduser(path or DEFAULT_CONFIG))
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        node = cfg
        keys = key.strip().split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(value)
    cfg["_path"] = path
    cfg["_dir"] = os.path.dirname(path)
    return cfg


def cli(description: str, add_args=None):
    """(args, cfg) for a script: --config <yaml>, script-specific options, then dotted overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="YAML config (default: config.yaml next to this script)")
    if add_args:
        add_args(parser)
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. robot.speed=30")
    args = parser.parse_args()
    return args, load_config(args.config, args.overrides)


def resolve_path(cfg: dict, p: str) -> str:
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(cfg["_dir"], p))


# ---------------------------------------------------------------- transforms

def pose_to_T(pose) -> np.ndarray:
    """xArm-style pose [x y z (m) roll pitch yaw (deg)] -> 4x4. xArm RPY = fixed axes X, Y, Z (R = Rz @ Ry @ Rx)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T) -> list:
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return [float(v) for v in (*T[:3, 3], *rpy)]


def inv_T(T) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def transform_points(T, pts) -> np.ndarray:
    return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]


def rot_angle_deg(R) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))


def flange_from_tcp(tcp_pose, controller_tcp_offset) -> np.ndarray:
    """T_base_flange from the TCP pose the controller reported and the TCP offset it was using at the time."""
    return pose_to_T(tcp_pose) @ inv_T(pose_to_T(controller_tcp_offset))


def tool_tip(cfg: dict, snapshot: dict, tool=None) -> np.ndarray:
    """Tool tip position in the base frame: flange (controller TCP offset removed) * tool (default robot.tcp_offset)."""
    T_base_flange = flange_from_tcp(snapshot["tcp_pose"], snapshot["controller_tcp_offset"])
    return (T_base_flange @ pose_to_T(cfg["robot"]["tcp_offset"] if tool is None else tool))[:3, 3]


def mean_T(Ts) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_matrix([t[:3, :3] for t in Ts]).mean().as_matrix()
    T[:3, 3] = np.mean([t[:3, 3] for t in Ts], axis=0)
    return T


def rigid_fit(src, dst):
    """Kabsch: T (4x4) with dst ~= T @ src, and the RMS residual (same units as the points)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cd - R @ cs
    rms = float(np.sqrt(np.mean(np.sum((transform_points(T, src) - dst) ** 2, axis=1))))
    return T, rms


def pose_is_safe(cfg: dict, pose) -> str | None:
    """None if the TCP position is inside safety.*, else the reason."""
    s = cfg["safety"]
    p = np.asarray(pose[:3], float)
    for i, key in enumerate(("workspace_x", "workspace_y", "workspace_z")):
        lo, hi = s[key]
        if not lo <= p[i] <= hi:
            return f"{'xyz'[i]}={p[i]:.4f} outside {key} [{lo}, {hi}]"
    if np.linalg.norm(p) > s["max_reach"]:
        return f"|p|={np.linalg.norm(p):.4f} > max_reach {s['max_reach']}"
    return None


# ---------------------------------------------------------------- I/O

def new_session_dir(cfg: dict, method: str) -> str:
    d = os.path.join(resolve_path(cfg, cfg["output"]["data_dir"]),
                     f"{method}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    return d


def latest_session_dir(cfg: dict, method: str) -> str:
    dirs = sorted(glob.glob(os.path.join(resolve_path(cfg, cfg["output"]["data_dir"]), f"{method}_*")))
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "samples.json"))]
    if not dirs:
        raise FileNotFoundError(f"no {method}_* session with samples.json in {cfg['output']['data_dir']}")
    return dirs[-1]


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(data), f, indent=1)
    os.replace(tmp, path)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_result(cfg: dict, T_base_cam: np.ndarray, extra: dict) -> tuple[str, str]:
    """Write T_base_cam (+ extra info) to output.result_file and return the static TF command."""
    ros = cfg["output"]["ros"]
    t_m = T_base_cam[:3, 3]
    q = Rotation.from_matrix(T_base_cam[:3, :3]).as_quat()  # x y z w
    cmd = ("ros2 run tf2_ros static_transform_publisher "
           f"--x {t_m[0]:.6f} --y {t_m[1]:.6f} --z {t_m[2]:.6f} "
           f"--qx {q[0]:.8f} --qy {q[1]:.8f} --qz {q[2]:.8f} --qw {q[3]:.8f} "
           f"--frame-id {ros['parent_frame']} --child-frame-id {ros['child_frame']}")
    result = {
        "T_base_cam": {
            "description": "camera (Photoneo CameraSpace) -> robot base (link_base): p_base[m] = matrix @ p_cam[m]. "
                           "Independent of the tool: TCP = flange * robot.tcp_offset is applied when moving",
            "matrix": np.round(T_base_cam, 8).tolist(),
            "translation": np.round(t_m, 6).tolist(),
            "quaternion_xyzw": np.round(q, 8).tolist(),
            "pose_xyz_rpy_deg": np.round(T_to_pose(T_base_cam), 6).tolist(),
        },
        **_jsonable(extra),
        "config": cfg["_path"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "ros_static_tf": cmd,
    }
    path = resolve_path(cfg, cfg["output"]["result_file"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False, default_flow_style=None, width=120)
    return path, cmd


def load_result(cfg: dict) -> np.ndarray:
    with open(resolve_path(cfg, cfg["output"]["result_file"])) as f:
        return np.asarray(yaml.safe_load(f)["T_base_cam"]["matrix"], float)
