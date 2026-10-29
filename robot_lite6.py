"""Lite6 over the xArm Python SDK (pip install xarm-python-sdk). Poses in and out are [x y z (m) roll pitch yaw (deg)];
the SDK's mm is converted here only.

TCP = flange * robot.tcp_offset (the tool mounted now), computed here, not by the controller: with
robot.controller_tcp_offset_zero the controller's TCP offset is set to 0 (it then reports / takes the flange pose, as in
lite6_control). Whatever offset the controller has, tcp_pose() / move_tcp() remove it, so changing the tool only means
changing robot.tcp_offset.

Do not move the arm from another client (xarm_ros2 driver / MoveIt) while a script here is running.
"""
from __future__ import annotations

import signal
import time

import numpy as np

from calib_utils import T_to_pose, inv_T, pose_is_safe, pose_to_T

MM = 1000.0


def _to_sdk(pose):
    p = [float(v) for v in pose]
    return [p[0] * MM, p[1] * MM, p[2] * MM, *p[3:6]]


def _from_sdk(pose):
    p = [float(v) for v in pose[:6]]
    return [p[0] / MM, p[1] / MM, p[2] / MM, *p[3:6]]


class RobotError(RuntimeError):
    pass


class Interrupted(BaseException):
    """Ctrl+Z / kill / terminal closed, raised by catch_exit_signals(): handle it like Ctrl+C."""


def _on_signal(signum, _frame):
    raise Interrupted(signal.Signals(signum).name)


def catch_exit_signals():
    """Ctrl+Z (SIGTSTP), SIGTERM, SIGHUP (terminal closed), SIGQUIT raise Interrupted instead of suspending / killing,
    so a script can still return the arm home."""
    for s in (signal.SIGTSTP, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(s, _on_signal)


class Lite6:
    def __init__(self, cfg: dict, log=print):
        from xarm.wrapper import XArmAPI

        self.cfg = cfg
        self.rcfg = cfg["robot"]
        self.log = log
        self.arm = XArmAPI(self.rcfg["ip"], is_radian=False)
        if not self.arm.connected:
            raise RobotError(f"cannot connect to Lite6 at {self.rcfg['ip']}")
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)
        self._check_error("connect")
        self.setup_tcp_offset()

    # ------------------------------------------------------------ state

    def _check_error(self, what: str):
        if self.arm.has_err_warn and self.arm.error_code != 0:
            raise RobotError(f"{what}: controller error_code={self.arm.error_code} warn_code={self.arm.warn_code}")

    def controller_tcp_offset(self) -> list:
        """TCP offset the controller is using right now [m, deg] (from the report stream)."""
        return _from_sdk(self.arm.tcp_offset)

    def setup_tcp_offset(self):
        if self.rcfg.get("controller_tcp_offset_zero", True):
            code = self.arm.set_tcp_offset([0.0] * 6, wait=True)
            if code != 0:
                raise RobotError(f"set_tcp_offset failed (code {code})")
            self.arm.set_state(0)
            time.sleep(0.5)  # let the report stream pick up the new offset
        self.log(f"controller TCP offset {np.round(self.controller_tcp_offset(), 5).tolist()}, "
                 f"tool (robot.tcp_offset) {self.tool()} (m, deg)")

    def tool(self) -> list:
        """T_flange_tcp of the mounted tool = robot.tcp_offset [m, deg]."""
        return [float(v) for v in self.rcfg["tcp_offset"]]

    def reported_pose(self) -> list:
        """Pose as the controller reports it (flange * the controller's own TCP offset)."""
        code, pose = self.arm.get_position(is_radian=False)
        if code != 0:
            raise RobotError(f"get_position failed (code {code})")
        return _from_sdk(pose)

    def flange_T(self) -> np.ndarray:
        return pose_to_T(self.reported_pose()) @ inv_T(pose_to_T(self.controller_tcp_offset()))

    def tcp_pose(self) -> list:
        """Tool tip pose in the base frame: flange * robot.tcp_offset."""
        return T_to_pose(self.flange_T() @ pose_to_T(self.tool()))

    def _command_pose(self, tcp_pose) -> list:
        """TCP target -> the pose the controller has to be sent (flange target * the controller's TCP offset)."""
        T_flange = pose_to_T(tcp_pose) @ inv_T(pose_to_T(self.tool()))
        return T_to_pose(T_flange @ pose_to_T(self.controller_tcp_offset()))

    def joints_deg(self) -> list:
        code, angles = self.arm.get_servo_angle(is_radian=False)
        if code != 0:
            raise RobotError(f"get_servo_angle failed (code {code})")
        return [float(a) for a in angles[:6]]

    def snapshot(self) -> dict:
        """Everything a sample needs from the robot. tcp_pose = the pose the controller reported, with the
        controller_tcp_offset it was computed with (flange = tcp_pose * inv(controller_tcp_offset)); tool_offset = the
        mounted tool (robot.tcp_offset) at that moment; flange_pose / tip_pose for reading."""
        reported, ctrl = self.reported_pose(), self.controller_tcp_offset()
        T_flange = pose_to_T(reported) @ inv_T(pose_to_T(ctrl))
        return {"tcp_pose": reported, "controller_tcp_offset": ctrl, "tool_offset": self.tool(),
                "flange_pose": T_to_pose(T_flange), "tip_pose": T_to_pose(T_flange @ pose_to_T(self.tool())),
                "joints_deg": self.joints_deg()}

    # ------------------------------------------------------------ motion

    def reachable(self, pose) -> str | None:
        """None if the TCP (tool tip) pose is inside safety.* and has an IK solution, else the reason."""
        why = pose_is_safe(self.cfg, pose)
        if why:
            return why
        code, _ = self.arm.get_inverse_kinematics(_to_sdk(self._command_pose(pose)), input_is_radian=False,
                                                  return_is_radian=False)
        return None if code == 0 else f"no IK solution (code {code})"

    def move_tcp(self, pose):
        """Move the tool tip (flange * robot.tcp_offset) to pose."""
        why = self.reachable(pose)
        if why:
            raise RobotError(f"target {np.round(pose, 4).tolist()} rejected: {why}")
        x, y, z, roll, pitch, yaw = _to_sdk(self._command_pose(pose))
        code = self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                     speed=self.rcfg["speed"] * MM, mvacc=self.rcfg["acc"] * MM,
                                     wait=True, timeout=self.rcfg["move_timeout_s"])
        if code != 0:
            raise RobotError(f"set_position failed (code {code}, error_code {self.arm.error_code})")
        time.sleep(self.rcfg["settle_s"])

    def move_joints(self, joints_deg):
        """Joint-space move to joints_deg (6 values, deg) at robot.joint_speed / robot.joint_acc."""
        code = self.arm.set_servo_angle(angle=[float(a) for a in joints_deg], speed=self.rcfg["joint_speed"],
                                        mvacc=self.rcfg["joint_acc"], wait=True, timeout=self.rcfg["move_timeout_s"],
                                        is_radian=False)
        if code != 0:
            raise RobotError(f"set_servo_angle failed (code {code}, error_code {self.arm.error_code})")
        time.sleep(self.rcfg["settle_s"])

    def move_home(self):
        """Lift straight up to robot.home_lift_z if lower, then joint move to robot.home_joints_deg
        (= lite6_control robot.poses.home, the pose ee_6d.py / sequence_real_action start from)."""
        lift_z = float(self.rcfg["home_lift_z"])
        pose = self.tcp_pose()
        if pose[2] < lift_z:
            try:
                self.move_tcp([pose[0], pose[1], lift_z, *pose[3:]])
            except RobotError as e:
                self.log(f"[HOME] lift skipped: {e}")
        self.move_joints(self.rcfg["home_joints_deg"])

    def go_home(self):
        """Final return when a script ends: stop what is running (an interrupted move, teach mode), then move_home.
        Never raises; a Ctrl+C during the return stops the arm."""
        self.log("\nhome 으로 복귀 중 (Ctrl+C = 정지)")
        try:
            self.halt()
            if self.arm.error_code != 0:
                self.log(f"[HOME] controller error_code {self.arm.error_code}: 자동 복귀 안 함. 로봇 상태를 확인하세요")
                return
            self.move_home()
            self.log("home 도착")
        except (KeyboardInterrupt, Interrupted):
            self.halt()
            self.log("[HOME] 복귀 중단, 로봇 정지")
        except RobotError as e:
            self.log(f"[HOME] 복귀 실패: {e}")

    def halt(self):
        """Stop the motion in progress (also one an interrupted wait left running) and accept new commands again."""
        self.arm.set_state(4)
        time.sleep(0.3)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)

    def teach_mode(self, on: bool):
        """Joint teaching mode (the arm can be moved by hand) on/off."""
        self.arm.set_mode(2 if on else 0)
        self.arm.set_state(0)
        time.sleep(0.5)

    def stop(self):
        self.arm.set_state(4)

    def close(self):
        try:
            if self.arm.mode == 2:
                self.teach_mode(False)
        finally:
            self.arm.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
