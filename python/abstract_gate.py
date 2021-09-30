# A superclass for a 2D gate.
import numpy as np

class AbstractGate():
    """A parent class for gates.
    This class specifies the parameters used for a generic gate.
    """
    def __init__(self):
        # Each side of the gate is a beam that's part of the frame. It has a width (left to right) and length (back to front).
        self.frame_width = 0
        self.frame_length = 0
        self.gate_width = 0
        self.translation = np.array([0,0])
        self.rotation = 0 # Radians.
        # Line starts (0) and line ends (1).
        self.l0 = np.zeros((8,2))
        self.l1 = np.zeros((8,2))