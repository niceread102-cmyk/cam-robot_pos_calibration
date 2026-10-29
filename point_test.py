"""Calibration test without the board: click points in the camera image, the TCP goes above them, measure the error.

Put small, sharp features on the table (a cross drawn on tape, a screw head, a coin centre ...).
The arm first goes to robot.home_joints_deg (the lite6_control home pose) and always returns there when the
script ends: normal end, error, Ctrl+C, Ctrl+Z, terminal closed (a second Ctrl+C during that return stops the arm).
Per round:
  1. the camera captures (arm at home)
  2. click the features in the image window: left click = add, right click = remove last, Enter = done, Esc = quit
     (no display: type the pixel "u v" lines in the terminal instead)
  3. each point -> base frame with T_base_cam -> TCP point_test.hover above it (base z), then
       Enter = look only and go on
       t     = teach mode: put the TCP tip exactly on the feature by hand, Enter -> error = tip - predicted (mm)
  4. back to home, next round
Errors (and their mean = systematic offset) are printed and saved to data/point_test_<time>/.

    python point_test.py
    python point_test.py point_test.hover=0.05
"""
import os
import warnings

import cv2
import numpy as np

from calib_utils import cli, load_result, new_session_dir, save_json, tool_tip, transform_points
from camera_photoneo import PhotoneoCamera, texture_to_gray8
from chessboard import Chessboard
from robot_lite6 import Interrupted, Lite6, RobotError, catch_exit_signals

WIN = "point_test: left click = add | right click = undo | Enter = done | Esc = quit"


def capture_points(cfg, cam, shots: int):
    """(gray8 texture of the first shot, point cloud averaged over the shots, m)."""
    gray, clouds = None, []
    for _ in range(shots):
        texture, points = cam.capture()
        if gray is None:
            gray = texture_to_gray8(texture, cfg["camera"]["texture_percentile"])
        clouds.append(points)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels stay NaN
        return gray, np.nanmean(np.stack(clouds), axis=0)


def draw(gray, uvs, cursor=None):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    s = max(1, img.shape[1] // 800)
    if cursor is not None:
        u, v = cursor
        cv2.line(img, (u - 15 * s, v), (u + 15 * s, v), (0, 255, 0), 1)
        cv2.line(img, (u, v - 15 * s), (u, v + 15 * s), (0, 255, 0), 1)
    for i, (u, v) in enumerate(uvs):
        cv2.drawMarker(img, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 20 * s, 2 * s)
        cv2.putText(img, str(i + 1), (int(u) + 8 * s, int(v) - 8 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                    (0, 255, 255), 2 * s)
    return img


def pick_gui(gray):
    """Clicked pixels [(u, v), ...]; None = quit."""
    uvs, state = [], {"cursor": None, "dirty": True}

    def on_mouse(event, u, v, *_):
        state["cursor"], state["dirty"] = (u, v), True
        if event == cv2.EVENT_LBUTTONDOWN:
            uvs.append((u, v))
        elif event == cv2.EVENT_RBUTTONDOWN and uvs:
            uvs.pop()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, int(1280 * gray.shape[0] / gray.shape[1]))
    cv2.setMouseCallback(WIN, on_mouse)
    try:
        while True:
            if state["dirty"]:
                cv2.imshow(WIN, draw(gray, uvs, state["cursor"]))
                state["dirty"] = False
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):
                return uvs
            if key in (27, ord("q")):
                return None
    finally:
        cv2.destroyWindow(WIN)
        cv2.waitKey(1)


def pick_terminal(image_path):
    print(f"image: {image_path}\n  pixel 'u v' per line, empty line = done, q = quit")
    uvs = []
    while True:
        line = input(f"  point {len(uvs) + 1} u v > ").strip().lower()
        if line == "q":
            return None
        if not line:
            return uvs
        try:
            u, v = (float(x) for x in line.replace(",", " ").split())
            uvs.append((u, v))
        except ValueError:
            print("  u v (pixels)")


def main():
    args, cfg = cli(__doc__.splitlines()[0])
    T_base_cam = load_result(cfg)
    out = new_session_dir(cfg, "point_test")
    report = {"result_file": cfg["output"]["result_file"], "T_base_cam": T_base_cam, "points": []}
    catch_exit_signals()

    with Lite6(cfg) as robot:
        try:
            input(f"Enter = home {cfg['robot']['home_joints_deg']} deg 로 이동 (Ctrl+C = 종료) ")
            robot.move_home()
            with PhotoneoCamera(cfg) as cam:
                rounds(cfg, robot, cam, T_base_cam, out, report)
        except (KeyboardInterrupt, Interrupted) as e:
            print(f"\n중단 ({type(e).__name__} {e})")
        finally:
            robot.go_home()

    errs = np.array([p["error"] for p in report["points"] if "error" in p])
    if len(errs):
        norms = np.linalg.norm(errs, axis=1)
        print(f"\n{len(errs)} measured points: |e| mean {norms.mean() * 1000:.2f} mm, max {norms.max() * 1000:.2f} mm")
        print(f"mean error vector (systematic) {np.round(errs.mean(axis=0) * 1000, 2).tolist()} mm")
        report["summary"] = {"mean_m": norms.mean(), "max_m": norms.max(), "mean_vector_m": errs.mean(axis=0)}
        save_json(os.path.join(out, "point_test.json"), report)
    print(f"saved: {out}")


def rounds(cfg, robot, cam, T_base_cam, out, report):
    pcfg = cfg["point_test"]
    hover = float(pcfg["hover"])
    up = np.array([0.0, 0.0, 1.0])
    local3d = Chessboard(cfg).corner_points  # local linear fit of the point cloud around a pixel (board.corner_*)
    gui = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    n_round = 0
    while True:
        n_round += 1
        gray, points = capture_points(cfg, cam, int(pcfg["shots"]))
        img_path = os.path.join(out, f"round{n_round:02d}.png")
        cv2.imwrite(img_path, gray)
        uvs = None
        if gui:
            try:
                uvs = pick_gui(gray)
            except cv2.error as e:
                print(f"[WARN] no image window ({e.err}), type the pixels instead")
                gui = False
        if not gui:
            uvs = pick_terminal(img_path)
        if uvs is None:
            break
        cv2.imwrite(os.path.join(out, f"round{n_round:02d}_picked.png"), draw(gray, uvs))
        p_cams = local3d(points, np.asarray(uvs, float)) if uvs else np.empty((0, 3))

        quit_all = False
        for i, (uv, p_cam) in enumerate(zip(uvs, p_cams)):
            name = f"round {n_round} point {i + 1} (u {uv[0]:.0f}, v {uv[1]:.0f})"
            if np.isnan(p_cam[0]):
                print(f"\n{name}: no valid 3D point around this pixel, skipped")
                continue
            p_base = transform_points(T_base_cam, p_cam[None])[0]
            above = [*(p_base + hover * up), *pcfg["tcp_rpy"]]
            print(f"\n{name}\n  camera {np.round(p_cam * 1000, 2).tolist()} mm -> base "
                  f"{np.round(p_base * 1000, 2).tolist()} mm")
            cmd = input(f"  Enter = 점 {hover * 1000:.0f} mm 위로 이동 | s = 건너뜀 | q = 종료 > ").strip().lower()
            if cmd == "q":
                quit_all = True
                break
            if cmd == "s":
                continue
            try:
                robot.move_tcp(above)
            except RobotError as e:
                print(f"  [SKIP] {e}")
                continue
            entry = {"round": n_round, "uv": list(uv), "camera": p_cam, "predicted": p_base}
            cmd = input("  도착. Enter = 다음 | t = 손으로 TCP 끝을 점에 맞춰 오차 측정 > ").strip().lower()
            if cmd == "t":
                robot.teach_mode(True)
                input("  teach mode ON: TCP 끝을 점 정중앙에 맞춘 뒤 Enter ")
                snap = robot.snapshot()
                robot.teach_mode(False)
                err = tool_tip(cfg, snap) - p_base
                entry.update(snap, error=err)
                print(f"  error (tip - predicted) dx {err[0] * 1000:+.2f}  dy {err[1] * 1000:+.2f}  "
                      f"dz {err[2] * 1000:+.2f}  |e| {np.linalg.norm(err) * 1000:.2f} mm")
                tip = robot.tcp_pose()
                robot.move_tcp([*(np.asarray(tip[:3]) + hover * up), *tip[3:]])
            report["points"].append(entry)
            save_json(os.path.join(out, "point_test.json"), report)

        if quit_all:
            break
        print("\nhome 으로 복귀 (다음 촬영)")
        robot.move_home()
        if input("Enter = 다음 촬영 (물체 위치를 바꿔도 됨) | q = 종료 > ").strip().lower() == "q":
            break


if __name__ == "__main__":
    main()
