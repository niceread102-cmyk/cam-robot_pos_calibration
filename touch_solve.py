"""Touch method, solve: T_base_cam from the touched corners and the camera's board measurement.

TCP tip of each touch = flange (reported pose with the controller's TCP offset removed) * the tool used for touching:
touch.tool_offset, or (null) the robot.tcp_offset recorded with the session. T_base_cam itself does not depend on the
tool; changing robot.tcp_offset later (another tool) needs no new calibration.
  board_model  : T_base_board = fit(board model corners -> tips), T_base_cam = T_base_board * inv(T_cam_board)
  camera_points: T_base_cam   = fit(camera-measured corner points -> tips)
Both are computed and reported; touch.solve_with picks the one written to output.result_file.

    python touch_solve.py                       # latest data/touch_* session, tool recorded with it
    python touch_solve.py --data data/touch_20261027_150000 touch.tool_offset="[0,0,0.1305,0,0,0]"
"""
import os

import numpy as np

from calib_utils import cli, inv_T, latest_session_dir, load_json, rigid_fit, tool_tip, transform_points, write_result
from chessboard import Chessboard


def add_args(p):
    p.add_argument("--data", default=None, help="session folder (default: the latest data/touch_*)")


def main():
    args, cfg = cli(__doc__.splitlines()[0], add_args)
    session = args.data or latest_session_dir(cfg, "touch")
    data = load_json(os.path.join(session, "samples.json"))
    board = Chessboard(cfg)
    tcfg = cfg["touch"]
    touches = data["touches"]
    if len(touches) < 3:
        raise SystemExit(f"{session}: {len(touches)} touches, need at least 3")

    # TCP tips: flange (the controller's own offset removed) * the touching tool.
    tool = tcfg.get("tool_offset")
    tool_src = "touch.tool_offset"
    if tool is None and "robot_tcp_offset_config" in data:
        tool, tool_src = data["robot_tcp_offset_config"], "robot.tcp_offset recorded with the session"
    elif tool is None:
        tool, tool_src = cfg["robot"]["tcp_offset"], "robot.tcp_offset (none recorded with the session)"
    tool = [float(v) for v in tool]
    tips, model, cam_pts, names = [], [], [], []
    for t in touches:
        tips.append(tool_tip(cfg, t, tool))
        c, r = t["corner"]
        model.append(board.model_point(c, r))
        cam_pts.append(np.asarray(data["camera"]["corners_cam"], float)[board.index(c, r)])
        names.append(f"[{c},{r}]")
    tips, model, cam_pts = np.array(tips), np.array(model), np.array(cam_pts)
    if np.linalg.matrix_rank(model - model.mean(axis=0), tol=1e-6) < 2:
        raise SystemExit("touched corners are collinear: touch corners that span both board directions")

    T_cam_board = np.asarray(data["camera"]["T_cam_board"], float)
    T_base_board, rms_model = rigid_fit(model, tips)
    candidates = {"board_model": T_base_board @ inv_T(T_cam_board)}
    ok_cam = ~np.isnan(cam_pts[:, 0])
    if ok_cam.sum() >= 3:
        candidates["camera_points"], rms_cam = rigid_fit(cam_pts[ok_cam], tips[ok_cam])
    else:
        rms_cam = None

    print(f"session {session}: {len(tips)} touches, touching tool {tool} ({tool_src})")
    if len(tips) == 3:
        print("[WARN] 3 touches: the fit is exact, RMS says nothing. Touch 4+ corners.")
    print(f"fit RMS  board_model {rms_model * 1000:.3f} mm"
          + (f" | camera_points {rms_cam * 1000:.3f} mm" if rms_cam is not None else ""))

    per_method = {}
    for name, T in candidates.items():
        err = np.full(len(tips), np.nan)
        err[ok_cam] = np.linalg.norm(transform_points(T, cam_pts[ok_cam]) - tips[ok_cam], axis=1)
        per_method[name] = err
        print(f"\n{name}: |T_base_cam * camera corner - TCP tip| per touch (mm)")
        for n, e in zip(names, err):
            print(f"  {n:>8} {e * 1000:7.3f}")
        print(f"  mean {np.nanmean(err) * 1000:.3f}  max {np.nanmax(err) * 1000:.3f}")
    if "camera_points" in candidates:
        T_a, T_b = candidates["board_model"], candidates["camera_points"]
        d_t = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
        print(f"\nboard_model vs camera_points: camera origin differs by {d_t * 1000:.3f} mm")

    choice = tcfg.get("solve_with", "board_model")
    if choice not in candidates:
        raise SystemExit(f"touch.solve_with={choice!r} not available (have {list(candidates)})")
    rms = rms_model if choice == "board_model" else rms_cam
    if rms > float(tcfg["max_fit_rms"]):
        print(f"[WARN] {choice} fit RMS {rms * 1000:.3f} mm > touch.max_fit_rms: re-touch carefully, check "
              "the touching tool (touch.tool_offset) and board.square_size")

    extra = {
        "method": f"touch/{choice}",
        "session": session,
        "calibration_tool_offset": tool,
        "fit_rms_m": rms,
        "errors_m": {n: float(e) for n, e in zip(names, per_method[choice])},
        "T_base_board": T_base_board,
        "board_shots_spread": {"m": data["camera"]["spread_m"], "deg": data["camera"]["spread_deg"]},
    }
    path, cmd = write_result(cfg, candidates[choice], extra)
    np.set_printoptions(precision=6, suppress=True)
    print(f"\nT_base_cam ({choice}, m):\n{candidates[choice]}\nwritten: {path}\n\n{cmd}")


if __name__ == "__main__":
    main()
