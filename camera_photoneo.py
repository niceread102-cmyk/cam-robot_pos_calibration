"""Photoneo capture over GigE-V (harvesters + the Photoneo GenTL producer), software trigger.

Returns the texture image and the organised point cloud (H x W x 3, camera space, m; invalid points = NaN). The
feature names follow photoneo-python-examples/GigE-V/harvesters (pointcloud_with_marker_space.py, components.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MM = 1000.0


def default_producer_path() -> str:
    name = "mvGenTLProducer.cti" if os.name == "nt" else "libmvGenTLProducer.so"
    for d in os.getenv("GENICAM_GENTL64_PATH", "").split(os.pathsep):
        if d and (Path(d) / name).exists():
            return str(Path(d) / name)
    raise FileNotFoundError(f"{name} not found in GENICAM_GENTL64_PATH; set camera.gentl_producer")


def _enable_components(features, names):
    for component in features.ComponentSelector.symbolics:
        features.ComponentSelector.value = component
        features.ComponentEnable.value = component in names


def _enabled_components(features):
    """Enabled components in payload order (increasing ComponentIDValue)."""
    ids = {}
    for c in features.ComponentSelector.symbolics:
        features.ComponentSelector.value = c
        if features.ComponentEnable.value:
            ids[c] = features.ComponentIDValue.value
    return sorted(ids, key=ids.get)


def _set(features, name, value):
    node = features.get_node(name)
    try:
        node.value = value
    except Exception as e:
        choices = getattr(node, "symbolics", None)
        raise RuntimeError(f"cannot set {name}={value!r}: {e}" + (f" (choices: {choices})" if choices else "")) from e


def texture_to_gray8(texture: np.ndarray, percentile=(1.0, 99.5)) -> np.ndarray:
    img = texture.astype(np.float64)
    if img.ndim == 3:
        img = img @ [0.299, 0.587, 0.114]  # RGB -> luma
    lo, hi = np.percentile(img, percentile)
    return (np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


class PhotoneoCamera:
    def __init__(self, cfg: dict, log=print):
        self.ccfg = cfg["camera"]
        self.log = log
        self.h = None
        self.ia = None

    def open(self):
        from harvesters.core import Harvester

        c = self.ccfg
        self.h = Harvester()
        self.h.add_file(c.get("gentl_producer") or default_producer_path(), check_existence=True, check_validity=True)
        self.h.update()
        self.log(f"connecting to Photoneo {c['serial']}")
        try:
            self.ia = self.h.create({"serial_number": str(c["serial"])})
        except Exception as e:
            if "AccessDenied" in type(e).__name__ or "-1005" in str(e):
                raise RuntimeError(
                    f"Photoneo {c['serial']} is already opened by another program (another script of this folder, "
                    "PhoXi Control connected to it, a ROS driver, ...). Close it and retry.") from e
            raise
        f = self.ia.remote_device.node_map
        self.log(f"firmware {f.DeviceFirmwareVersion.value}")

        if c.get("load_default_user_set", True):
            f.UserSetSelector.value = "Default"
            f.UserSetLoad.execute()
        for name, value in (c.get("features") or {}).items():
            _set(f, name, value)

        # Fixed settings: one software-triggered frame = texture + organised point cloud in camera space.
        f.TriggerSelector.value = "FrameStart"
        f.TriggerMode.value = "On"
        f.TriggerSource.value = "Software"
        self.texture_name = c.get("texture_component", "Intensity")
        _enable_components(f, [self.texture_name, "Range"])
        _set(f, "Scan3dOutputMode", "CalibratedABC_Grid")
        _set(f, "CoordinateSpace", "CameraSpace")
        self.components = _enabled_components(f)
        self.ia.start()
        return self

    def capture(self):
        """(texture HxW or HxWx3, points HxWx3 in m with NaN where invalid)."""
        f = self.ia.remote_device.node_map
        f.TriggerSoftware.execute()
        with self.ia.fetch(timeout=float(self.ccfg["fetch_timeout_s"])) as buffer:
            parts = dict(zip(self.components, buffer.payload.components))
            tex, rng = parts[self.texture_name], parts["Range"]
            if tex.data_format == "RGB8":
                texture = np.array(tex.data, copy=True).reshape(tex.height, tex.width, 3)
            else:
                texture = np.array(tex.data, copy=True).reshape(tex.height, tex.width)
            points = np.array(rng.data, dtype=np.float64, copy=True).reshape(rng.height, rng.width, 3) / MM
        if texture.shape[:2] != points.shape[:2]:
            raise RuntimeError(f"{self.texture_name} {texture.shape[:2]} and Range {points.shape[:2]} differ in size; "
                               "use a texture component aligned with the point cloud")
        points[points[..., 2] <= 0] = np.nan  # Photoneo marks invalid points with 0
        return texture, points

    def close(self):
        if self.ia is not None:
            try:
                self.ia.stop()
            finally:
                self.ia.destroy()
                self.ia = None
        if self.h is not None:
            self.h.reset()
            self.h = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
