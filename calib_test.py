"""Calibration test on the real robot with the result in output.result_file (T_base_cam).

The camera measures the board (arm out of view); every corner in calib_test.corners is mapped to the base frame with
T_base_cam and the TCP tip goes calib_test.hover above it (along the board normal). At each corner you can:
  Enter = look only and go on
  t     = teach mode: put the TCP tip exactly on the corner by hand, Enter -> error = tip - predicted corner (mm)
The measured errors (and their mean = systematic offset) are printed and saved to data/calib_test_<time>/.

    python calib_test.py
    python calib_test.py calib_test.hover=0.04 calib_test.corners="[[2,2],[8,5]]"
"""
import os

import numpy as np

from calib_utils import cli, load_result, new_session_dir, save_json, tool_tip, transform_points
from camera_photoneo import PhotoneoCamera
from chessboard import Chessboard, capture_board
from robot_lite6 import Lite6, RobotError


def main():
    args, cfg = cli(__doc__.splitlines()[0])
    tcfg = cfg["calib_test"]
    board = Chessboard(cfg)
    T_base_cam = load_result(cfg)
    corners = [tuple(int(v) for v in c) for c in tcfg["corners"]]
    for c in corners:
        board.index(*c)
    hover = float(tcfg["hover"])
    out = new_session_dir(cfg, "calib_test")
    report = {"result_file": cfg["output"]["result_file"], "T_base_cam": T_base_cam, "points": []}

    with Lite6(cfg) as robot, PhotoneoCamera(cfg) as cam:
        input("로봇을 카메라 시야(보드 위) 밖으로 치운 뒤 Enter -> 보드 촬영 ")
        res = capture_board(cfg, cam, board, int(tcfg["board_shots"]), out, "board", corners)
        T_base_board = T_base_cam @ res["T_cam_board"]
        normal = T_base_board[:3, 2] * (1.0 if T_base_board[2, 2] >= 0 else -1.0)  # board normal, pointing up
        print(f"board normal in base: {np.round(normal, 4).tolist()} (tilt {np.degrees(np.arccos(normal[2])):.2f} deg)")

        for c in corners:
            p_cam = res["corners_cam"][board.index(*c)]
            if np.isnan(p_cam[0]):
                print(f"\ncorner {list(c)}: no 3D point, skipped")
                continue
            p_base = transform_points(T_base_cam, p_cam[None])[0]
            above = [*(p_base + hover * normal), *tcfg["tcp_rpy"]]
            print(f"\ncorner {list(c)}: predicted base {np.round(p_base * 1000, 2).tolist()} mm")
            cmd = input(f"  Enter = 코너 {hover * 1000:.0f} mm 위로 이동 | s = 건너뜀 | q = 종료 > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                continue
            try:
                robot.move_tcp(above)
            except RobotError as e:
                print(f"  [SKIP] {e}")
                continue
            entry = {"corner": list(c), "predicted": p_base}
            cmd = input("  도착. Enter = 다음 | t = 손으로 TCP 끝을 코너에 맞춰 오차 측정 > ").strip().lower()
            if cmd == "t":
                robot.teach_mode(True)
                input("  teach mode ON: TCP 끝을 코너 정중앙에 맞춘 뒤 Enter ")
                snap = robot.snapshot()
                robot.teach_mode(False)
                err = tool_tip(cfg, snap) - p_base
                entry.update(snap, error=err)
                print(f"  error (tip - predicted) dx {err[0] * 1000:+.2f}  dy {err[1] * 1000:+.2f}  "
                      f"dz {err[2] * 1000:+.2f}  |e| {np.linalg.norm(err) * 1000:.2f} mm")
                tip = robot.tcp_pose()
                robot.move_tcp([*(np.asarray(tip[:3]) + hover * normal), *tip[3:]])  # lift off
            report["points"].append(entry)
            save_json(os.path.join(out, "calib_test.json"), report)

    errs = np.array([p["error"] for p in report["points"] if "error" in p])
    if len(errs):
        norms = np.linalg.norm(errs, axis=1)
        print(f"\n{len(errs)} measured corners: |e| mean {norms.mean() * 1000:.2f} mm, max {norms.max() * 1000:.2f} mm")
        print(f"mean error vector (systematic) {np.round(errs.mean(axis=0) * 1000, 2).tolist()} mm")
        report["summary"] = {"mean_m": norms.mean(), "max_m": norms.max(), "mean_vector_m": errs.mean(axis=0)}
        save_json(os.path.join(out, "calib_test.json"), report)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
