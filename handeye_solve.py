"""Hand-eye method (eye-to-hand), solve: T_base_cam from flange poses and board poses (OpenCV calibrateHandEye).

Flange pose of each sample = reported TCP pose with the controller's TCP offset (recorded per sample) removed.
Eye-to-hand trick: pass base->flange (= inv(T_base_flange)) as "gripper2base"; the result is cam->base = T_base_cam.
Quality: every sample predicts the board corners twice (robot chain T_base_flange * T_flange_board vs camera chain
T_base_cam * measured corners); the mean distance per sample is its error. Outliers are dropped one at a time.

    python handeye_solve.py                    # latest data/handeye_* session
    python handeye_solve.py --data data/handeye_20261027_150000 handeye.outlier_err=0.002
"""
import itertools
import os

import cv2
import numpy as np

from calib_utils import (cli, flange_from_tcp, inv_T, latest_session_dir, load_json, mean_T, rot_angle_deg,
                         transform_points, write_result)
from chessboard import Chessboard

METHODS = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK, "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
           "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF, "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}


def add_args(p):
    p.add_argument("--data", default=None, help="session folder (default: the latest data/handeye_*)")


def solve(samples, method):
    T_fb = [inv_T(s["T_base_flange"]) for s in samples]
    R_c2g, t_c2g = cv2.calibrateHandEye([T[:3, :3] for T in T_fb], [T[:3, 3] for T in T_fb],
                                        [s["T_cam_board"][:3, :3] for s in samples],
                                        [s["T_cam_board"][:3, 3] for s in samples], method=METHODS[method])
    T = np.eye(4)
    T[:3, :3] = R_c2g
    T[:3, 3] = t_c2g.ravel()
    return T


def evaluate(T_base_cam, samples, board):
    """(per-sample mean corner error (m), T_flange_board mean)."""
    T_flange_board = mean_T([inv_T(s["T_base_flange"]) @ T_base_cam @ s["T_cam_board"] for s in samples])
    errs = []
    for s in samples:
        ok = ~np.isnan(s["corners_cam"][:, 0])
        robot_chain = transform_points(s["T_base_flange"] @ T_flange_board, board.model[ok])
        camera_chain = transform_points(T_base_cam, s["corners_cam"][ok])
        errs.append(float(np.mean(np.linalg.norm(robot_chain - camera_chain, axis=1))))
    return np.array(errs), T_flange_board


def best_solution(samples, methods, board):
    best = None
    for m in methods:
        try:
            T = solve(samples, m)
        except cv2.error as e:
            print(f"  {m:<10} failed: {e}")
            continue
        errs, T_fb = evaluate(T, samples, board)
        print(f"  {m:<10} corner error mean {errs.mean() * 1000:7.3f} mm  max {errs.max() * 1000:7.3f} mm")
        if best is None or errs.mean() < best[2].mean():
            best = (m, T, errs, T_fb)
    if best is None:
        raise SystemExit("every hand-eye method failed")
    return best


def main():
    args, cfg = cli(__doc__.splitlines()[0], add_args)
    hcfg = cfg["handeye"]
    session = args.data or latest_session_dir(cfg, "handeye")
    board = Chessboard(cfg)
    samples = []
    for s in load_json(os.path.join(session, "samples.json"))["samples"]:
        samples.append({"idx": s["idx"],
                        "T_base_flange": flange_from_tcp(s["tcp_pose"], s["controller_tcp_offset"]),
                        "T_cam_board": np.asarray(s["T_cam_board"], float),
                        "corners_cam": np.asarray(s["corners_cam"], float)})
    print(f"session {session}: {len(samples)} samples")
    if len(samples) < int(hcfg["min_samples"]):
        raise SystemExit(f"{len(samples)} samples < handeye.min_samples")

    spread = max(rot_angle_deg(a["T_base_flange"][:3, :3].T @ b["T_base_flange"][:3, :3])
                 for a, b in itertools.combinations(samples, 2))
    print(f"flange rotation spread {spread:.1f} deg")
    if spread < float(hcfg["min_rotation_spread_deg"]):
        print("[WARN] small rotation spread: the translation of the result is poorly determined")

    dropped = []
    while True:
        method, T, errs, T_fb = best_solution(samples, hcfg["methods"], board)
        worst = int(np.argmax(errs))
        lim = hcfg.get("outlier_err")
        if lim is None or errs[worst] <= float(lim) or len(samples) - 1 < int(hcfg["min_samples"]):
            break
        print(f"  drop sample {samples[worst]['idx']} (error {errs[worst] * 1000:.3f} mm)")
        dropped.append(samples.pop(worst)["idx"])

    print(f"\nbest: {method}, {len(samples)} samples (dropped {dropped})")
    for s, e in zip(samples, errs):
        print(f"  sample {s['idx']:3d}: {e * 1000:7.3f} mm")
    extra = {
        "method": f"handeye/{method}",
        "session": session,
        "samples_used": [s["idx"] for s in samples],
        "samples_dropped": dropped,
        "corner_error_m": {"mean": float(errs.mean()), "max": float(errs.max())},
        "T_flange_board": T_fb,
    }
    path, cmd = write_result(cfg, T, extra)
    np.set_printoptions(precision=6, suppress=True)
    print(f"\nT_base_cam (m):\n{T}\nwritten: {path}\n\n{cmd}")


if __name__ == "__main__":
    main()
