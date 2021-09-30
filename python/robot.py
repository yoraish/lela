# Code for the LELA project. (Learned Lasermotor Navigagtion Policies.
# Class Robot includes intrinsic parameters and settings alongside the robot pose.
import numpy as np
from config import LelaConfig

class Robot():
    def __init__(self, x = np.zeros(2), h = np.zeros(2)):
        config = LelaConfig()
        # Position.
        self.x = x
        # Heading. Keep normalized.
        self.h = h/np.linalg.norm(h)
        self.th = np.arctan2(self.h[1], self.h[0]) % (np.pi*2) # World frame rotation angle in radians.

        # Lidar info.
        self.num_beams = 1285

        # Forward range.
        self.ph_opening = np.pi
        self.ph_step = 2*np.pi / self.num_beams
        self.num_beams_in_range = int(self.ph_opening/(2*np.pi) * self.num_beams)

        # Beam angles relative to **world** frame. These are NOT relative to robot heading, they already take that into account.
        self.beam_phs = [(self.th  - self.ph_opening/2. + self.ph_step * i) % (np.pi*2)
                         for i in range(self.num_beams_in_range)]

        
        self.lidar_max_dist = config.lidar_max_dist # cm

    def set_x_h(self, x, h):
        # Position.
        self.x = x
        # Heading. Keep normalized.
        self.h = h/np.linalg.norm(h)
        self.th = np.arctan2(self.h[1], self.h[0]) % (np.pi*2) # World frame rotation angle in radians.

        # Beam angles relative to **world** frame. These are NOT relative to robot heading, they already take that into account.
        self.beam_phs = [(self.th  - self.ph_opening/2. + self.ph_step * i) % (np.pi*2) for i in range(self.num_beams_in_range)]

    def move_forward(self, dist):
        self.x = self.x +  self.h * dist

