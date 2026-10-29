"""Hand-eye method, data collection: chessboard fixed to the flange, the arm visits poses, the camera measures the board.

handeye.mode: generate (center x offsets) | list (handeye.list) | manual (teach mode, Enter to capture).
Writes data/handeye_<time>/samples.json (+ images). Then run handeye_solve.py.

    python handeye_collect.py
    python handeye_collect.py --ik-only              # only check which generated poses are reachable
    python handeye_collect.py handeye.mode=manual
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from calib_utils import T_to_pose, cli, new_session_dir, pose_to_T, save_json
from camera_photoneo import PhotoneoCamera
from chessboard import Chessboard, DetectionError, capture_board
from robot_lite6 import Lite6, RobotError


def add_args(p):
    p.add_argument("--ik-only", action="store_true", help="check the poses (safety box + IK) and exit, no motion")


def target_poses(hcfg) -> list:
    if hcfg["mode"] == "list":
        return [list(map(float, p)) for p in hcfg["list"]]
    g = hcfg["generate"]
    T_c = pose_to_T(g["center"])
    poses = []
    for dp in g["position_offsets"]:
        for dr in g["orientation_offsets"]:
            T = T_c.copy()
            T[:3, 3] += np.asarray(dp, float)
            T[:3, :3] = T_c[:3, :3] @ Rotation.from_euler("xyz", dr, degrees=True).as_matrix()  # in the TCP frame
            poses.append(T_to_pose(T))
    return poses


def capture_sample(cfg, cam, board, out, idx, robot, target=None):
    try:
        res = capture_board(cfg, cam, board, 1, out, f"pose_{idx:03d}")
    except DetectionError as e:
        print(f"  [SKIP] {e}")
        return None
    return {"idx": idx, "target": target, **robot.snapshot(),
            "T_cam_board": res["T_cam_board"], "corners_cam": res["corners_cam"],
            "fit_rms": res["shots"][0]["fit_rms"]}


def main():
    args, cfg = cli(__doc__.splitlines()[0], add_args)
    hcfg = cfg["handeye"]
    board = Chessboard(cfg)

    with Lite6(cfg) as robot:
        poses = []
        if hcfg["mode"] != "manual":
            for p in target_poses(hcfg):
                why = robot.reachable(p)
                print(f"  {'OK  ' if why is None else 'SKIP'} {np.round(p, 4).tolist()}" + (f"  ({why})" if why else ""))
                if why is None:
                    poses.append(p)
            print(f"{len(poses)} reachable poses")
            if args.ik_only:
                return
            if len(poses) < int(hcfg["min_samples"]):
                raise SystemExit(f"only {len(poses)} reachable poses < handeye.min_samples")

        out = new_session_dir(cfg, "handeye")
        path = os.path.join(out, "samples.json")
        data = {"method": "handeye", "board": cfg["board"], "robot_tcp_offset_config": cfg["robot"]["tcp_offset"],
                "samples": []}
        print(f"session: {out}")

        with PhotoneoCamera(cfg) as cam:
            if hcfg["mode"] == "manual":
                robot.teach_mode(True)
                print("teach mode ON: 로봇을 손으로 옮긴 뒤 정지시키고 Enter = 촬영 | q = 종료")
                while input(f"[{len(data['samples'])} samples] > ").strip().lower() != "q":
                    s = capture_sample(cfg, cam, board, out, len(data["samples"]), robot)
                    if s:
                        data["samples"].append(s)
                        save_json(path, data)
                robot.teach_mode(False)
            else:
                input(f"\n{len(poses)} 개 자세로 로봇이 움직입니다. 주변을 비우고 Enter ")
                try:
                    for i, p in enumerate(poses):
                        print(f"[{i + 1}/{len(poses)}] -> {np.round(p, 4).tolist()}")
                        try:
                            robot.move_tcp(p)
                        except RobotError as e:
                            print(f"  [SKIP] {e}")
                            continue
                        s = capture_sample(cfg, cam, board, out, i, robot, p)
                        if s:
                            data["samples"].append(s)
                            save_json(path, data)
                    if hcfg.get("return_to_start", True):
                        robot.move_tcp(poses[0])
                except KeyboardInterrupt:
                    robot.stop()
                    print("\ninterrupted: arm stopped")

    print(f"\n{len(data['samples'])} samples saved: {path}\nnext: python handeye_solve.py --data {out}")


if __name__ == "__main__":
    main()
