"""Capture one Photoneo frame, detect the chessboard and print its pose in the camera frame. No robot needed.

Saves the annotated texture (corners, origin [0,0], +x / +y, the touch.corners circled) so you can see which board
corner is which before touching.

    python test_camera.py
    python test_camera.py board.detector=classic camera.features="{LEDPower: 4095}"
"""
import os

import cv2
import numpy as np

from calib_utils import T_to_pose, cli, new_session_dir
from camera_photoneo import PhotoneoCamera, texture_to_gray8
from chessboard import Chessboard, DetectionError


def main():
    args, cfg = cli(__doc__.splitlines()[0])
    board = Chessboard(cfg)
    out = new_session_dir(cfg, "test_camera")
    with PhotoneoCamera(cfg) as cam:
        texture, points = cam.capture()
    gray = texture_to_gray8(texture, cfg["camera"]["texture_percentile"])
    cv2.imwrite(os.path.join(out, "texture.png"), gray)
    print(f"texture {texture.shape} {texture.dtype}, valid 3D points {np.mean(~np.isnan(points[..., 2])) * 100:.1f} %")

    try:
        res = board.estimate(gray, points)
    except DetectionError as e:
        corners = None
        try:
            corners = board.detect(gray)
        except DetectionError:
            pass
        cv2.imwrite(os.path.join(out, "detection.png"), board.draw(gray, corners))
        print(f"[FAIL] {e}\nimages: {out}")
        return

    cv2.imwrite(os.path.join(out, "detection.png"), board.draw(gray, res["corners_px"], cfg["touch"]["corners"]))
    T = res["T_cam_board"]
    np.set_printoptions(precision=5, suppress=True)
    print(f"corners valid {res['n_valid']}/{len(res['corners_px'])}, board fit RMS {res['fit_rms'] * 1000:.3f} mm")
    print(f"square size measured {res['square_measured'] * 1000:.2f} mm (board.square_size "
          f"{cfg['board']['square_size'] * 1000:.2f} mm)")
    print(f"T_cam_board (m):\n{T}")
    print(f"board origin in camera: {T[:3, 3]} m, distance {np.linalg.norm(T[:3, 3]):.4f} m")
    print(f"pose [x y z rpy]: {np.round(T_to_pose(T), 5).tolist()}")
    print(f"images: {out}")


if __name__ == "__main__":
    main()
