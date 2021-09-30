# A class for a 2D gate.
import numpy as np
import time 

from abstract_gate import AbstractGate

class Gate(AbstractGate):
    """A parent class for gates.
    This class specifies the parameters used for a generic gate.
    """
    def __init__(self, name = 0, width = 20, frame_width = 2, frame_length = 2, translation = np.array([[15,10]]), rotation = -1):
        super().__init__()
        # Each side of the gate is a beam that's part of the frame. It has a width (left to right) and length (back to front).
        self.name = name
        # All units are cm.
        self.frame_width = frame_width
        self.frame_length = frame_length
        # The width of the gate opening, measured between the inside faces of the beams.
        self.gate_width = width
        self.translation = translation
        self.rotation = rotation # Radians. The forward direction is along the x axis per robotics convention.
        self.R = np.array([[np.cos(self.rotation), -np.sin(self.rotation)],
                           [np.sin(self.rotation), np.cos(self.rotation)]])
                    
        self.left_points = np.array([
                             [-self.gate_width/2.0 - self.frame_width, -self.frame_length/2.0],
                             [-self.gate_width/2.0, -self.frame_length/2.0],
                             [-self.gate_width/2.0, self.frame_length/2.0],
                             [-self.gate_width/2.0 - self.frame_width, self.frame_length/2.0]])

        self.right_points = np.array([
                             [self.gate_width/2.0 + self.frame_width, -self.frame_length/2.0],
                             [self.gate_width/2.0, -self.frame_length/2.0],
                             [self.gate_width/2.0, self.frame_length/2.0],
                             [self.gate_width/2.0 + self.frame_width, self.frame_length/2.0]])

        # Rotate points such that x axis is forward.
        self.left_points = np.hstack((self.left_points[:,1].reshape(-1, 1), -self.left_points[:, 0].reshape(-1, 1)))
        self.right_points = np.hstack((self.right_points[:,1].reshape(-1, 1), -self.right_points[:, 0].reshape(-1, 1)))

        # Translate to place.
        self.left_points = self.R.dot(self.left_points.T).T + self.translation
        self.right_points = self.R.dot(self.right_points.T).T + self.translation

        # Array of start points.
        self.us_left = self.left_points.copy() #np.vstack((self.left_points, self.right_points))
        self.us_right = self.right_points.copy() #np.vstack((self.left_points, self.right_points))
        # Array of edge end points.
        self.vs_left = np.roll(self.us_left.copy(), -1, axis=0)
        self.vs_right = np.roll(self.us_right.copy(), -1, axis=0)
        # Combine the starts and combine the ends.
        self.us = np.vstack((self.us_left, self.us_right))
        self.vs = np.vstack((self.vs_left, self.vs_right))

    def ray_hit(self, laser_origin, laser_vector, max_distance):
        '''
        Returns the point on the gate that is hit by a laser beam from `laser_origin`$\in \mathbb{R}^2$ in the direction of `laser_vector` $\in \mathbb{R}^2$ as long as it is closer to the origin than `max_distance`.
        '''
        # Say u0 is the robot position, v0 is the endpoint of the laser ray (max_distance away). All of these vectors are np arrays.
        u0 = laser_origin
        v0 = laser_vector * max_distance# laser_origin + 
        # Say we have u1s that are start segment of a rectangle edge, and v1s that are corresponding end segments.
        hits = []
        for uv1_ix in range(len(self.us)):
            u1 = self.us[uv1_ix, :]
            v1 = self.vs[uv1_ix, :] - u1
            # | v1.x   v1.y |  -> T -> | v1.x   v0.x |
            # | v0.x   v0.y |          | v1.y   v0.y |
            A = np.vstack((v1, v0)).T 
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue
            s, t = A_inv.dot(u0 - u1).reshape((2,)) * np.array([1,-1])
            if s >= 0 and s <=1 and t >= 0 and t <= 1:
                hits.append((np.linalg.norm(v0*t) , u0 + v0*t))
        
        hits.sort() # Sort by the distance to the robot.
        if len(hits) > 0:
            return hits[0][1]
        else:
            return None








if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure()
    # for th in range (3):
    #     g = Gate(width = 30, frame_width = 8, frame_length = 8, rotation=-th/4, translation=np.array([[100*th,100*th]]))
    #     plt.scatter(*g.left_points.T)
    #     plt.scatter(*g.right_points.T)
    # plt.scatter([-30,30,-30,30],[-30,-30,30,30])
    # plt.show()

    # Precompute lidare beam vectors.
    phs = [np.pi*2*th_lidar/1285 for th_lidar in range(1285)]

    # Now draw the same thing but with lines.
    plt.figure(1)

    # Each ray at a time, not each gate at a time.

    gates = []
    for th in range (5):
        g = Gate(width = 30, frame_width = 8, frame_length = 8, rotation=-th/4, translation=np.array([[100*th,100*th]]))
        
        # Add gate to gates collection.
        gates.append(g)
        for ix in range(len(g.us)):
            plt.plot([g.us[ix][0], g.vs[ix][0]], [g.us[ix][1], g.vs[ix][1]], c = 'b') 
    
    robot_x = np.array([70,50])
    lidar_dist = 200
    ax = plt.gca()
    circle = plt.Circle(robot_x, lidar_dist, color='b', fill=False)
    ax.add_patch(circle)
    plt.scatter(*robot_x, marker="*")
    
    # Ph is laser beam heading.
    for ph in phs:
        ray_hits = []
        for g in gates:
            gate_hit_point = g.ray_hit(robot_x, np.array([np.cos(ph), np.sin(ph)]), lidar_dist)
            if gate_hit_point is not None:
                ray_hits.append((np.linalg.norm(gate_hit_point - robot_x), gate_hit_point))
        
        # Sort the hits and only take the closest one to the robot.
        ray_hits.sort()
        if len(ray_hits) != 0:
            ray_hit_point = ray_hits[0][1]
            plt.scatter(*ray_hit_point, c = "r")
    
    plt.show()