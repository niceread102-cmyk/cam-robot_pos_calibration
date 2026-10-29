"""Object pose seen by the camera (typed in the terminal) -> robot base frame -> move the TCP there.

Input line (camera frame = Photoneo CameraSpace, m / deg):  x y z [roll pitch yaw]      q = quit
  rpy convention: move_obj.euler. TCP target: move_obj.orientation (object * grasp_offset, or tool down with
  move_obj.tcp_rpy; x y z only = tool down). With move_obj.approach > 0 the arm first stops that far back along the
  TCP z axis, then approaches after Enter, then retreats.
With the robot, the arm first goes to robot.home_joints_deg (the lite6_control home pose, as with ee_6d.py) and
always returns there when the script ends: q, error, Ctrl+C, Ctrl+Z, terminal closed (a second Ctrl+C during that
return stops the arm).

    python move_obj.py                 # convert + move
    python move_obj.py --no-robot      # convert only (prints the base-frame pose)
    python move_obj.py move_obj.orientation=down move_obj.approach=0.08
"""
import numpy as np
from scipy.spatial.transform import Rotation

from calib_utils import T_to_pose, cli, inv_T, load_result, pose_to_T
from robot_lite6 import Interrupted, Lite6, RobotError, catch_exit_signals


def add_args(p):
    p.add_argument("--no-robot", action="store_true", help="only convert camera -> base, do not connect / move")


def parse(line: str):
    vals = [float(v) for v in line.replace(",", " ").split()]
    if len(vals) not in (3, 6):
        raise ValueError("x y z  또는  x y z roll pitch yaw  (m, deg)")
    return vals


def tcp_targets(cfg, T_base_obj, has_rpy: bool):
    """(TCP target pose, approach pose or None) in the base frame, [m, deg]."""
    m = cfg["move_obj"]
    if has_rpy and m["orientation"] == "object":
        T_tcp = T_base_obj @ pose_to_T(m["grasp_offset"])
    else:
        T_tcp = pose_to_T([*T_base_obj[:3, 3], *m["tcp_rpy"]])
    approach = float(m.get("approach") or 0.0)
    T_app = None
    if approach > 0:
        T_app = T_tcp.copy()
        T_app[:3, 3] -= approach * T_tcp[:3, 2]  # back off along the TCP z axis
    return T_to_pose(T_tcp), (T_to_pose(T_app) if T_app is not None else None)


def fmt(pose):
    return (f"xyz [{pose[0]:+.4f} {pose[1]:+.4f} {pose[2]:+.4f}] m  "
            f"rpy [{pose[3]:+.2f} {pose[4]:+.2f} {pose[5]:+.2f}] deg")


def main():
    args, cfg = cli(__doc__.splitlines()[0], add_args)
    euler = cfg["move_obj"]["euler"]
    T_base_cam = load_result(cfg)
    catch_exit_signals()
    robot = None if args.no_robot else Lite6(cfg)
    T_tool = pose_to_T(cfg["robot"]["tcp_offset"])
    try:
        if robot:
            input(f"Enter = home {cfg['robot']['home_joints_deg']} deg 로 이동 (Ctrl+C = 종료) ")
            robot.move_home()
        print(f"tool (robot.tcp_offset) {cfg['robot']['tcp_offset']} (m, deg)")
        print(f"camera-frame object pose: x y z [roll pitch yaw]  (m, deg, euler '{euler}')   q = quit")
        while True:
            line = input("\ncam obj > ").strip()
            if line.lower() in ("q", "quit", "exit"):
                break
            if not line:
                continue
            try:
                vals = parse(line)
            except ValueError as e:
                print(f"  입력 오류: {e}")
                continue

            T_cam_obj = np.eye(4)
            T_cam_obj[:3, 3] = vals[:3]
            if len(vals) == 6:
                T_cam_obj[:3, :3] = Rotation.from_euler(euler, vals[3:], degrees=True).as_matrix()
            T_base_obj = T_base_cam @ T_cam_obj
            rpy_in = Rotation.from_matrix(T_base_obj[:3, :3]).as_euler(euler, degrees=True)
            print(f"  base obj  : {fmt(T_to_pose(T_base_obj))}  (xArm rpy)")
            if euler != "xyz":
                print(f"              rpy '{euler}' [{rpy_in[0]:+.2f} {rpy_in[1]:+.2f} {rpy_in[2]:+.2f}] deg")
            target, approach = tcp_targets(cfg, T_base_obj, len(vals) == 6)
            print(f"  TCP target: {fmt(target)}")
            print(f"  flange    : {fmt(T_to_pose(pose_to_T(target) @ inv_T(T_tool)))}  (TCP offset 0)")
            if approach:
                print(f"  approach  : {fmt(approach)}")
            if robot is None:
                continue

            for name, pose in (("approach", approach), ("target", target)):
                why = pose and robot.reachable(pose)
                if why:
                    print(f"  [SKIP] {name} not reachable: {why}")
                    break
            else:
                try:
                    if approach:
                        if input("  Enter = approach 로 이동 | n = 취소 > ").strip().lower() == "n":
                            continue
                        robot.move_tcp(approach)
                    if input("  Enter = 목표(obj)로 이동 | n = 취소 > ").strip().lower() == "n":
                        continue
                    robot.move_tcp(target)
                    print(f"  도착: TCP {fmt(robot.tcp_pose())}")
                    if approach and input("  Enter = approach 로 후퇴 | n = 그대로 > ").strip().lower() != "n":
                        robot.move_tcp(approach)
                except RobotError as e:
                    print(f"  [ERROR] {e}")
    except (KeyboardInterrupt, Interrupted) as e:
        print(f"\n중단 ({type(e).__name__} {e})")
    finally:
        if robot:
            try:
                robot.go_home()
            finally:
                robot.close()


if __name__ == "__main__":
    main()
