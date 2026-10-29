"""Touch method, data collection: board fixed on the table, camera measures it, the TCP tip touches its corners.

  1. connect Lite6 (controller TCP offset -> 0 with robot.controller_tcp_offset_zero); the touching tool =
     robot.tcp_offset, recorded with the session (touch_solve uses it)
  2. keep the arm out of the camera view -> touch.camera_shots captures of the board (averaged)
  3. teach mode: move the TCP tip by hand onto each corner in touch.corners and press Enter
  4. (touch.recheck_camera) board captured again to check it did not move
Writes data/touch_<time>/samples.json (+ images). Then run touch_solve.py.

    python touch_collect.py
    python touch_collect.py touch.corners="[[0,0],[6,0],[0,5],[6,5]]"
"""
import os

import numpy as np

from calib_utils import cli, new_session_dir, save_json
from camera_photoneo import PhotoneoCamera
from chessboard import Chessboard, capture_board
from robot_lite6 import Lite6


def main():
    args, cfg = cli(__doc__.splitlines()[0])
    tcfg = cfg["touch"]
    board = Chessboard(cfg)
    corners = [tuple(int(v) for v in c) for c in tcfg["corners"]]
    for c in corners:
        board.index(*c)  # validates the index
    if len(corners) < 3:
        raise SystemExit("touch.corners: at least 3 corners are needed (4+ recommended)")

    out = new_session_dir(cfg, "touch")
    path = os.path.join(out, "samples.json")
    data = {"method": "touch", "board": cfg["board"], "robot_tcp_offset_config": cfg["robot"]["tcp_offset"],
            "camera": None, "touches": [], "camera_after": None}
    print(f"session: {out}")

    with Lite6(cfg) as robot, PhotoneoCamera(cfg) as cam:
        input("\n로봇을 카메라 시야(보드 위) 밖으로 치운 뒤 Enter -> 보드 촬영 ")
        print(f"capturing board x{tcfg['camera_shots']}")
        data["camera"] = capture_board(cfg, cam, board, int(tcfg["camera_shots"]), out, "board", corners)
        print(f"board shots spread: {data['camera']['spread_m'] * 1000:.3f} mm, {data['camera']['spread_deg']:.3f} deg"
              f"  (어느 코너가 어디인지: {out}/board_00.png)")
        save_json(path, data)

        robot.teach_mode(True)
        print("\nteach mode ON: 로봇을 손으로 움직여 TCP 끝을 코너 정중앙에 맞추세요.")
        print("  Enter = 기록 | s = 이 코너 건너뜀 | q = 종료")
        i = 0
        while i < len(corners):
            c = corners[i]
            cmd = input(f"[{i + 1}/{len(corners)}] corner [{c[0]},{c[1]}] > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                i += 1
                continue
            snap = robot.snapshot()
            data["touches"].append({"corner": list(c), **snap})
            save_json(path, data)
            print(f"  tip {np.round(np.array(snap['tip_pose'][:3]) * 1000, 2).tolist()} mm")
            i += 1
        robot.teach_mode(False)

        if tcfg.get("recheck_camera", True) and data["touches"]:
            input("\n로봇을 다시 시야 밖으로 치운 뒤 Enter -> 보드 재촬영 (보드가 밀렸는지 확인) ")
            data["camera_after"] = capture_board(cfg, cam, board, int(tcfg["camera_shots"]), out, "board_after")
            shift = np.linalg.norm(data["camera_after"]["T_cam_board"][:3, 3] - data["camera"]["T_cam_board"][:3, 3])
            flag = "[WARN] 보드가 움직였습니다. 다시 수집하세요." if shift > float(tcfg["max_board_shift"]) else "OK"
            print(f"board shift during touching: {shift * 1000:.3f} mm  {flag}")
            save_json(path, data)

    print(f"\n{len(data['touches'])} touches saved: {path}\nnext: python touch_solve.py --data {out}")


if __name__ == "__main__":
    main()
