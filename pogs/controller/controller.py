import os
import sys
import subprocess
from pathlib import Path


class ControllerBase:
    """Abstract base controller interface."""
    def connect(self):
        raise NotImplementedError()

    def disconnect(self):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()


class RealSenseController(ControllerBase):
    """A lightweight wrapper that launches the existing RealSense capture script
    as a subprocess. This is intentionally minimal: it provides `connect`/`disconnect`
    checks and `start`/`stop` to run the interactive capture script in background.
    """

    def __init__(self, scene_name: str = "my_scene", save_path: str = "data/realsense_captures"):
        self.scene_name = scene_name
        self.save_path = save_path
        self.proc = None
        self.connected = False

    def connect(self):
        """Try importing the RealSense library to detect device availability."""
        try:
            import pyrealsense2  # noqa: F401
            self.connected = True
        except Exception:
            self.connected = False
        return self.connected

    def disconnect(self):
        self.stop()
        self.connected = False

    def start(self):
        """Start the `pogs/scripts/realsense_capture.py` script as a subprocess.
        The script is interactive (opens a window and listens for keys). This wrapper
        simply spawns it and returns the subprocess object.
        """
        if self.proc:
            return self.proc

        script_path = Path(__file__).resolve().parents[1] / "scripts" / "realsense_capture.py"
        if not script_path.exists():
            raise FileNotFoundError(f"RealSense capture script not found: {script_path}")

        cmd = [sys.executable, str(script_path), "--scene_name", self.scene_name, "--save_path", self.save_path]
        # Start subprocess; keep stdout/stderr piped so caller can read logs if desired.
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return self.proc

    def stop(self):
        if not self.proc:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            self.proc = None

    # --- Non-interactive streaming API ---
    def start_stream(self, width: int = 1280, height: int = 720, fps: int = 30):
        """Start an in-process RealSense pipeline for programmatic frame access.
        Returns True if the sensor is available and streaming started.
        """
        try:
            import pyrealsense2 as rs
            import numpy as np
        except Exception as e:
            raise RuntimeError("pyrealsense2 not available: " + str(e))

        if getattr(self, "_pipeline", None):
            return True

        self._rs = rs
        self._np = np
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._profile = self._pipeline.start(cfg)

        # depth scale
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        # align depth to color
        self._align = rs.align(rs.stream.color)

        return True

    def get_frame(self, timeout_ms: int = 5000):
        """Return a tuple (color_image, depth_meters) or None on timeout.
        `color_image` is HxWx3 uint8 BGR, `depth_meters` is HxW float32.
        """
        if not getattr(self, "_pipeline", None):
            raise RuntimeError("Stream not started. Call start_stream() first.")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        aligned = self._align.process(frames)
        d = aligned.get_depth_frame()
        c = aligned.get_color_frame()
        if not d or not c:
            return None

        depth_raw = self._np.asanyarray(d.get_data())
        color = self._np.asanyarray(c.get_data())
        depth_m = depth_raw.astype(self._np.float32) * self._depth_scale
        return color, depth_m

    def stop_stream(self):
        if getattr(self, "_pipeline", None):
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._profile = None
        self._align = None
        self._depth_scale = None
