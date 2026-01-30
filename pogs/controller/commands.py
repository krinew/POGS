from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class Command:
    """Represents a motor command from the policy.

    type: 'joint' or 'pose'
    - for 'joint': `joints` is a list of floats (radians)
    - for 'pose': `pose` is a 4x4 list-of-lists representing a transformation matrix (camera/world frame)
    """
    type: str
    joints: Optional[List[float]] = None
    pose: Optional[List[List[float]]] = None

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "joints": self.joints, "pose": self.pose})
