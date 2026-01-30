from typing import Tuple
import numpy as np
from .commands import Command


class SimpleDepthPolicy:
    """A minimal perception->action policy.

    Strategy: compute the mean depth in a small center crop of the depth image
    and generate a small Cartesian pose delta to move the camera/robot forward/back
    so the mean depth approaches `target_depth`.
    This is a placeholder to be replaced with a learned or more sophisticated policy.
    """

    def __init__(self, target_depth: float = 0.5, crop_frac: float = 0.2, z_gain: float = 0.5):
        self.target_depth = target_depth
        self.crop_frac = crop_frac
        self.z_gain = z_gain

    def propose_command(self, color: np.ndarray, depth: np.ndarray) -> Command:
        """Return a `Command` object.

        - `color`: HxWx3 uint8 BGR
        - `depth`: HxW float32 meters
        """
        H, W = depth.shape[:2]
        ch = int(H * self.crop_frac // 2)
        cw = int(W * self.crop_frac // 2)
        cy, cx = H // 2, W // 2
        crop = depth[cy - ch: cy + ch, cx - cw: cx + cw]
        # ignore zeros (missing depth)
        valid = crop[crop > 0]
        if valid.size == 0:
            # no valid depth — do nothing
            return Command(type="pose", pose=np.eye(4).tolist())

        mean_depth = float(np.mean(valid))
        dz = mean_depth - self.target_depth
        # compute a simple delta along camera Z (negative moves camera forward)
        delta_z = -self.z_gain * dz

        pose = np.eye(4)
        pose[2, 3] = float(delta_z)

        return Command(type="pose", pose=pose.tolist())
