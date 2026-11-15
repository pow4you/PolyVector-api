"""Keypoint detection for PolyVector Flow.

``PyramidalKeypointDetector`` needs the optional ``flow`` extra (PyTorch,
scikit-image, matplotlib); importing this package does not.
"""

from .base import KeypointDetector, Keypoints, KeypointType
from .pyramidal import PyramidalKeypointDetector

__all__ = ["KeypointDetector", "Keypoints", "KeypointType", "PyramidalKeypointDetector"]
