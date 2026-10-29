"""Chessboard detection in the Photoneo texture and board pose from the organised point cloud (no intrinsics).

Board frame: origin = inner corner [0, 0] = the inner corner next to the black outer corner square (the board needs
an even x odd number of squares so the two diagonal outer squares differ in colour), x along increasing column index,
y along increasing row index, z = x cross y pointing into the board (away from the camera). Corner [c, r] is at
(c * square, r * square, 0). Lengths in m.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from calib_utils import mean_T, rigid_fit, rot_angle_deg


class DetectionError(RuntimeError):
    pass


def capture_board(cfg: dict, cam, board: "Chessboard", shots: int, out_dir: str, prefix: str, highlight=(),
                  log=print) -> dict:
    """`shots` captures of a static board, averaged: T_cam_board (mean), corners_cam (per-corner nanmean),
    plus per-shot fit RMS and the spread of the shots. Failed shots are skipped; raises if none succeeds."""
    from camera_photoneo import texture_to_gray8

    results = []
    for k in range(shots):
        texture, points = cam.capture()
        gray = texture_to_gray8(texture, cfg["camera"]["texture_percentile"])
        try:
            res = board.estimate(gray, points)
        except DetectionError as e:
            log(f"  shot {k}: [FAIL] {e}")
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_{k:02d}_fail.png"), gray)
            continue
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{k:02d}.png"), board.draw(gray, res["corners_px"], highlight))
        log(f"  shot {k}: {res['n_valid']} corners, fit RMS {res['fit_rms'] * 1000:.3f} mm, "
            f"square {res['square_measured'] * 1000:.2f} mm")
        results.append(res)
    if not results:
        raise DetectionError(f"board not detected in any of {shots} shots")
    T = mean_T([r["T_cam_board"] for r in results])
    spread_t = max(np.linalg.norm(r["T_cam_board"][:3, 3] - T[:3, 3]) for r in results)
    spread_r = max(rot_angle_deg(T[:3, :3].T @ r["T_cam_board"][:3, :3]) for r in results)
    return {"T_cam_board": T,
            "corners_cam": np.nanmean([r["corners_cam"] for r in results], axis=0),
            "corners_px": results[-1]["corners_px"],
            "shots": [{"T_cam_board": r["T_cam_board"], "fit_rms": r["fit_rms"], "n_valid": r["n_valid"],
                       "square_measured": r["square_measured"]} for r in results],
            "spread_m": float(spread_t), "spread_deg": float(spread_r)}


class Chessboard:
    def __init__(self, cfg: dict):
        b = cfg["board"]
        self.b = b
        self.cols, self.rows = int(b["squares"][0]) - 1, int(b["squares"][1]) - 1   # inner corners
        self.square = float(b["square_size"])
        c, r = np.meshgrid(np.arange(self.cols), np.arange(self.rows))               # OpenCV order: row-major
        self.model = np.column_stack([c.ravel() * self.square, r.ravel() * self.square, np.zeros(c.size)])

    def index(self, col: int, row: int) -> int:
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            raise ValueError(f"corner [{col}, {row}] outside the {self.cols} x {self.rows} inner corners")
        return row * self.cols + col

    def model_point(self, col: int, row: int) -> np.ndarray:
        return self.model[self.index(col, row)]

    # ------------------------------------------------------------ 2D

    def detect(self, gray: np.ndarray) -> np.ndarray:
        """Inner corners (N, 2) in pixels, OpenCV order, or DetectionError."""
        size = (self.cols, self.rows)
        if self.b.get("detector", "sb") == "sb":
            ok, corners = cv2.findChessboardCornersSB(gray, size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        else:
            ok, corners = cv2.findChessboardCorners(
                gray, size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
        if not ok:
            raise DetectionError(f"chessboard {self.cols}x{self.rows} inner corners not found")
        return self._canonical_order(gray, corners.reshape(-1, 2).astype(np.float64))

    def _canonical_order(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """OpenCV may start the corner order at any board corner. Make it unique:
        (1) +x cross +y in the image > 0, i.e. the board z axis points away from the camera (into the board),
        (2) the outer square diagonally outside corner [0,0] is black (vs. the one outside the last corner)."""
        g = corners.reshape(self.rows, self.cols, 2)
        ex, ey = g[0, 1] - g[0, 0], g[1, 0] - g[0, 0]
        if ex[0] * ey[1] - ex[1] * ey[0] < 0:
            g = g[:, ::-1]  # mirror the column order
        first = self._patch_mean(gray, g[0, 0] - 0.5 * (g[1, 1] - g[0, 0]))
        last = self._patch_mean(gray, g[-1, -1] + 0.5 * (g[-1, -1] - g[-2, -2]))
        if first > last:
            g = g[::-1, ::-1]  # 180 deg: start at the other end
        return np.ascontiguousarray(g).reshape(-1, 2)

    @staticmethod
    def _patch_mean(gray: np.ndarray, uv, r: int = 3) -> float:
        u, v = int(round(uv[0])), int(round(uv[1]))
        patch = gray[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
        return float(patch.mean()) if patch.size else 0.0

    # ------------------------------------------------------------ 3D

    def corner_points(self, points: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """3D point (m, camera frame) at each sub-pixel corner: a local linear fit X,Y,Z = a + b*du + c*dv over the
        valid points of the surrounding window, evaluated at the corner. NaN where too few valid points."""
        r = int(self.b["corner_window_px"])
        min_pts = int(self.b["corner_min_points"])
        H, W = points.shape[:2]
        out = np.full((len(corners), 3), np.nan)
        for i, (u, v) in enumerate(corners):
            u0, v0 = int(round(u)), int(round(v))
            v_lo, v_hi, u_lo, u_hi = max(v0 - r, 0), min(v0 + r + 1, H), max(u0 - r, 0), min(u0 + r + 1, W)
            win = points[v_lo:v_hi, u_lo:u_hi].reshape(-1, 3)
            vv, uu = np.mgrid[v_lo:v_hi, u_lo:u_hi]
            ok = ~np.isnan(win[:, 2])
            if ok.sum() < min_pts:
                continue
            A = np.column_stack([np.ones(ok.sum()), uu.ravel()[ok] - u, vv.ravel()[ok] - v])
            coef, *_ = np.linalg.lstsq(A, win[ok], rcond=None)
            out[i] = coef[0]
        return out

    def estimate(self, gray: np.ndarray, points: np.ndarray) -> dict:
        """Board pose in the camera frame. Keys: T_cam_board, corners_px, corners_cam (NaN = invalid), fit_rms,
        n_valid."""
        corners = self.detect(gray)
        cam = self.corner_points(points, corners)
        valid = ~np.isnan(cam[:, 0])
        ratio = valid.mean()
        if ratio < float(self.b["min_valid_ratio"]):
            raise DetectionError(f"only {valid.sum()}/{len(valid)} corners have valid 3D points")
        T, rms = rigid_fit(self.model[valid], cam[valid])
        if rms > float(self.b["max_fit_rms"]):
            raise DetectionError(f"board fit RMS {rms * 1000:.2f} mm > board.max_fit_rms")
        return {"T_cam_board": T, "corners_px": corners, "corners_cam": cam, "fit_rms": rms,
                "n_valid": int(valid.sum()), "square_measured": self.measured_square(cam)}

    def measured_square(self, cam: np.ndarray) -> float:
        """Median distance between neighbouring corners in the point cloud (compare with board.square_size)."""
        g = cam.reshape(self.rows, self.cols, 3)
        d = np.concatenate([np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(),
                            np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()])
        return float(np.nanmedian(d))

    # ------------------------------------------------------------ drawing

    def draw(self, gray: np.ndarray, corners: np.ndarray | None, highlight=()) -> np.ndarray:
        """Annotated BGR image: detected corners, origin [0,0], +x / +y directions and the highlighted corners."""
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if corners is None:
            return img
        cv2.drawChessboardCorners(img, (self.cols, self.rows), corners.reshape(-1, 1, 2).astype(np.float32), True)
        p = lambda c, r: tuple(int(round(x)) for x in corners[self.index(c, r)])  # noqa: E731
        scale = max(1, img.shape[1] // 800)
        cv2.arrowedLine(img, p(0, 0), p(min(2, self.cols - 1), 0), (0, 0, 255), 3 * scale, tipLength=0.2)
        cv2.arrowedLine(img, p(0, 0), p(0, min(2, self.rows - 1)), (0, 255, 0), 3 * scale, tipLength=0.2)
        cv2.putText(img, "[0,0]", p(0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 0, 255), 2 * scale)
        cv2.putText(img, "+x", p(min(2, self.cols - 1), 0), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 255),
                    2 * scale)
        cv2.putText(img, "+y", p(0, min(2, self.rows - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 255, 0),
                    2 * scale)
        for c, r in highlight:
            cv2.circle(img, p(c, r), 12 * scale, (0, 255, 255), 2 * scale)
            cv2.putText(img, f"[{c},{r}]", (p(c, r)[0] + 14 * scale, p(c, r)[1] - 14 * scale),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 255), 2 * scale)
        return img
