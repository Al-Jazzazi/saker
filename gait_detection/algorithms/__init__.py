"""
Gait Detection Algorithms

Available detectors:
- HysteresisDetector: Dual-threshold detector that prevents noise
- ThresholdDetector: Simple single-threshold detector
- WeightedDetector: Weighted sum of sensors per foot
"""

from .hysteresis_detector import GaitDetector as HysteresisDetector
from .threshold_detector import ThresholdDetector
from .weighted_detector import WeightedDetector

__all__ = ['HysteresisDetector', 'ThresholdDetector', 'WeightedDetector']
