# Code for the LELA project. (Learned Lasermotor Navigagtion Policies).
# This module keeps track of all the gates in the playground and the robot. This is the "World". The playground is charged with providing lidar ray-tracing observations to the robot.
# This module also visualizes the map.
'''
# Notes:
* Gate directions point towards the right peg, so need to be adjusted by -1.5pi. This is a design choice that should be changed to be consistent with the robot pose setup.
* The gate position is [[x],[y]] -- a size 2x1 array-- it is not consistent with 

'''

import numpy as np
import matplotlib.pyplot as plt
import time
import PIL
import os
import random
import string

import torchvision.transforms.functional as transform
import torch

from cubic_spline_with_derivative import sampleCubicSplinesWithDerivative
from robot import Robot
from gate import Gate
from config import LelaConfig
from network import AerialImagesVAE, Dronet

class Playground():
    def __init__(self, robot, gates):
        """[summary]

        Args:
            gates (list): list of Gate objects.
            robot (Robot): the Robot object.
        """
        self.gates = gates
        self.r = robot
        self.config = LelaConfig()
        

    def get_scan(self):
        # For each ray in the robot's lidar, include the closest hit among all gates.
        # Return the scan and also the closest gate index.
        # The returned scan is of points in the world frame
        scan = []
        hit_gate_names = set()
        # `beam_ph` is laser beam heading.
        for beam_ph in self.r.beam_phs:
            ray_hits = []
            for g in self.gates:
                gate_hit_point = g.ray_hit(self.r.x, np.array([np.cos(beam_ph), np.sin(beam_ph)]), self.r.lidar_max_dist)
                if gate_hit_point is not None:
                    ray_hits.append((np.linalg.norm(gate_hit_point - self.r.x), gate_hit_point))
                    hit_gate_names.add(g.name)
            # Sort the hits and only take the closest one to the robot.
            ray_hits.sort()
            if len(ray_hits) != 0:
                ray_hit_point = ray_hits[0][1]
                scan.append(ray_hit_point)
        return np.array(scan), min(hit_gate_names)
    
    def get_scan_tensor(self):
        # Returns a tensor of the scan in the robot frame.
        # Build a rotation matrix for the robot pose. Use the inverse of that matrix to shift the scan to be axis aligned.
        # Create an empty scan.
        scan = np.zeros((2*self.r.lidar_max_dist+1, 2*self.r.lidar_max_dist+1))
        
        # Get scan hit points in the world frame.
        scan_points, g_name = self.get_scan()
        # Shift the points to the center.
        scan_points -= self.r.x

        # Rotate points to correct for robot rotation.
        # NOTE(yoraish): The rotation matrix is negated to rotate to the "wrong" side. Later, we will flip the x-axis of the resulting scan image. This takes care of the fact numpy handles arrays as row-column and not x,y.
        R_minus = -np.array([[np.cos(self.r.th), -np.sin(self.r.th)],
                        [np.sin(self.r.th), np.cos(self.r.th)]])
        # R_inv = np.linalg.inv(R)
        R_minus_inv = R_minus.T
        # Rotate points.
        scan_points = R_minus_inv.dot(scan_points.T).T

        # Voxelize to the 1x1cm level. Also Shift the point (0,0) to the center of the image, so add lidar_max_dist to both x and y values.
        scan_points = np.unique(scan_points.astype(np.int), axis=0) + self.r.lidar_max_dist
        # Invert x axis.
        scan_points = scan_points* np.array([[-1,1]]) + 2*np.array([[self.r.lidar_max_dist, 0]])

        # Shift the point (0,0) to the center of the image, so add lidar_max_dist to both x and y values.
        for sp in scan_points:
            scan[sp[1]][sp[0]] = 255

        # To image.
        im = PIL.Image.fromarray(scan.astype(np.uint8))
        # Change the size of the scan to be the specified.
        im = im.resize([config.img_size ,config.img_size],PIL.Image.ANTIALIAS)

        # To tensor and return.
        return transform.to_tensor(im), g_name


        
    def vis(self, draw_robot = True, draw_gates = True, draw_scan = True, additional = [], img = None, text= "", save_path = ""):
        plt.figure(0, figsize=(16, 8))

        if img is not None:
            plt.subplot(1,2,1)

        # If robot, draw a star for robot position and arrow for robot heading.
        if draw_robot:
            plt.scatter(*self.r.x, marker="*")
            h_vis = np.array(self.r.h)*10
            plt.arrow(x= self.r.x[0], dx=h_vis[0], y= self.r.x[1], dy = h_vis[1], width = 1)

            # Show opening angle.
            plt.plot([self.r.x[0], 
                        self.r.x[0] + self.r.lidar_max_dist*np.cos(self.r.beam_phs[0])],
                        [self.r.x[1], 
                        self.r.x[1] + self.r.lidar_max_dist*np.sin(self.r.beam_phs[0])],
                        c ='b')
            plt.plot([self.r.x[0], 
                        self.r.x[0] + self.r.lidar_max_dist*np.cos(self.r.beam_phs[-1])],
                        [self.r.x[1], 
                        self.r.x[1] + self.r.lidar_max_dist*np.sin(self.r.beam_phs[-1])],
                        c ='b')

            # Draw all rays.
            # for i in range(self.r.num_beams_in_range):
            #     plt.plot([self.r.x[0], 
            #     self.r.x[0] + self.r.lidar_max_dist*np.cos(self.r.beam_phs[i])],
            #     [self.r.x[1], 
            #     self.r.x[1] + self.r.lidar_max_dist*np.sin(self.r.beam_phs[i])],
            #     c ='b')

            ax = plt.gca()
            circle = plt.Circle(self.r.x, self.r.lidar_max_dist, color='b', fill=False)
            ax.add_patch(circle)

        # If gates, draw gates with lines.
        if draw_gates:
            for g in self.gates:
                for ix in range(len(g.us)):
                    plt.plot([g.us[ix][0], g.vs[ix][0]], [g.us[ix][1], g.vs[ix][1]], c = 'b') 
                    
                # Also draw the gate axis.
                a = g.translation
                ab_dir = g.R.dot(np.array([[1,0]]).T).T 
                b = g.translation +10*ab_dir
                orth_R =  np.array([[0, -1], [1, 0]])

                ac_dir = orth_R.dot(ab_dir.T).T
                c = a + 10*ac_dir

                # plt.plot(*zip(a[0], b[0]), c = "r")
                plt.arrow(a[0][0], a[0][1], 5*ab_dir[0][0], 5*ab_dir[0][1], width = 1, color = "r")
                plt.arrow(a[0][0], a[0][1], 5*ac_dir[0][0], 5*ac_dir[0][1], width = 1, color = "g")

        # If scan, generate and draw scan.
        if draw_scan:
            scan, g_name = self.get_scan()
            for ray_hit_point in scan:       
                plt.scatter(*ray_hit_point, c = "r")
                plt.title("Closest gate is " +str(g_name))

        plt.axis('equal')


        if additional:
            plt.scatter(additional['x'], additional['y'], alpha = 0.4)

        if img is not None:
            plt.subplot(1,2,2)
            plt.imshow(img)

        if text:
            plt.subplot(1,2,2)
            plt.text(20,20, text, c = "w")

        if save_path:
            plt.savefig(save_path)
            plt.cla()
            plt.clf()
        else:
            print("show")
            plt.show()

    def get_spline_samples(self):
        xs = [self.r.x[0]]
        ys = [self.r.x[1]]
        points = [self.r.x]
        tangents = [np.array([np.cos(self.r.th), np.sin(self.r.th)])]
        dydx = [self.r.th]
        for g in self.gates:
            points.append(g.translation.reshape(2))
            tangents.append(np.array([np.cos(g.rotation), np.sin(g.rotation)]))

        spline_samples = sampleCubicSplinesWithDerivative(points, tangents, 4)
        return spline_samples

if __name__ == "__main__":
    config = LelaConfig()
    # Playground test for model.

    # Get model.
    #-----------
    device = 'cpu'
    model_class = config.model_class
    model_path = config.model_path[model_class]

    if model_class == "VAE_IMG":
        if os.path.exists(model_path):
            model = AerialImagesVAE(LelaConfig.n_z)
            model.load_state_dict(torch.load(model_path))
        else:
            print("WARNING: Weights not found for model ", model_class)
            model = AerialImagesVAE(config.n_z).to(device)

    elif model_class == "DRONET":
        if os.path.exists(model_path):
            model = Dronet(3)
            model.load_state_dict(torch.load(model_path))
        else:
            print("WARNING: Weights not found for model ", model_class)
            model = Dronet(3).to(device)

    elif model_class == "VAE_IMG_CMD":
        if os.path.exists(model_path):
            raise NotImplementedError
            model = AerialImagesVAE(LelaConfig.n_z)
            model.load_state_dict(torch.load(model_path))
        else:
            model = AerialImagesVAE(config.n_z).to(device)
            print("WARNING: Weights not found for model ", model_class)

    model.eval()

    # Create playground with robot and gates.
    # For some number of steps (or until robot our of gates/map).
    #   1. Get scan.
    #   2. Ask for inference.
    #   3. Move robot in direction of gate.

    # Set up a robot.
    #----------------
    r = Robot(x = np.array([1.0, 2.0]), h = np.array([0, 0.3]))

    # Put down some gates.
    #---------------------
    gates = [Gate(name = 0, width = 30, frame_width = 4, frame_length = 4, rotation= np.pi/2, translation = np.array([[1 ,  30 + np.random.rand()]]))]
    for i in range(config.num_gates_in_run - 1):
        # Previous gate information.
        prev_name = gates[-1].name
        prev_rot = gates[-1].rotation
        prev_trans = gates[-1].translation
        g = Gate(name = prev_name + 1, width = 30, frame_width = 4, frame_length = 4, 
                rotation= prev_rot + (np.random.rand()-0.5)* config.gate_rot_rand_scaler, 
                translation= prev_trans + np.array([np.cos(prev_rot), np.sin(prev_rot)])*(config.gate_trans_min_dist + np.random.rand()*config.gate_trans_rand_scaler))
        # Add gate to gates collection.
        gates.append(g)

    # Set up a playground.
    #---------------------
    pg = Playground(r, gates)

    # Draw a spline between the gates for visualization.
    # --------------------------------------------------
    spline_samples = pg.get_spline_samples()

    vis_spline_x = []
    vis_spline_y = []
    for s in spline_samples:
        vis_spline_x.append(s[0])
        vis_spline_y.append(s[1])
    

    N = 4
    run_randname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
    os.mkdir("animation/"+run_randname)

    for step in range(100):
        # Get a scan for the current robot pose.
        # --------------------------------------
        scan_tensor, g_name = pg.get_scan_tensor()
        g_near = pg.gates[g_name]
        # Add a "batch" to this data.
        data = scan_tensor.unsqueeze(0)

        # Inference for gate pose.
        #-------------------------
        if model_class == "VAE_IMG":
            recon_batch, mu, logvar, z = model(data).detach().numpy()

        elif model_class == "DRONET":
            z = model(data).detach().numpy()
            pred_gate_x, pred_gate_y, pred_gate_th = z[0]

        elif model_class == "VAE_IMG_CMD":
            pass
        

        # Ground truth data.
        #-------------------
        gt_g_near = pg.gates[g_name]
        # Rotation of gate along its axis.
        gt_g_rel_th = (gt_g_near.rotation - pg.r.th) % (2*np.pi)

        # Angle between robot and gate center point.
        gt_g_near_centered_r =  gt_g_near.translation.T - np.expand_dims(pg.r.x, axis = 0).T
        R = np.array([[np.cos(pg.r.th), -np.sin(pg.r.th)],
                    [np.sin(pg.r.th), np.cos(pg.r.th)]])

        gt_g_rel_pos = (R.T).dot(gt_g_near_centered_r).T


        # Move robot.
        #------------
        # NOTE(yoraish): Currently the control is naive. Go to the centroid of the gate. No significance to the orientation of the gate. For example, if a gate is directly perpendicular to the robot, it'll drive directly into its side.
        # Rotatino matrix to rotate the robot heading h = [x_h, y_h].
        pred_th_to_gate = np.arctan2(pred_gate_y, pred_gate_x)
        R_to_gate = np.array([[np.cos(pred_th_to_gate), -np.sin(pred_th_to_gate)],
                            [np.sin(pred_th_to_gate),  np.cos(pred_th_to_gate)]])

        # Rotate the robot heading.
        pg.r.h = R_to_gate.dot(pg.r.h)

        # Move forward a little bit.
        pg.r.move_forward(2)

        print("==GATE== GT   gate position:", gt_g_rel_pos)
        print("==GATE== PRED gate position:", pred_gate_x, pred_gate_y)
        print("==GATE== GT   gate angle:", gt_g_rel_th)
        print("==GATE== PRED gate angle:", pred_th_to_gate)
        # print("==CMD== GT   :", gt_g_rel_th)
        # print("==CMD== PRED :", pred_th_to_gate)
        print("\n")

        pg.vis(draw_robot = True, draw_gates= True, draw_scan=False, additional = {'x':vis_spline_x, 'y':vis_spline_y}, save_path=os.path.join("/Users/yoraish/code/lela/python/animation/",run_randname, model_class + "_" + str(step) + ".png"))


        # # Heading array (2,).
        # h = spline_samples[s_ix +1] - spline_samples[s_ix]
        # # Robot position.
        # x = spline_samples[s_ix]
        # # Update robot.
        # # pg.r.set_x_h(x,h)

        




    # # gate_x and gate_y and gate_th for dataset.
    # gate_x = g_rel_pos[0][0]
    # gate_y = g_rel_pos[0][1]
    # gate_th = g_rel_th

    # # yaw_command for dataset. This is robot heading r.th minus spline heading `h`. Between -pi and pi.
    # h_spline = spline_samples[s_ix + config.cmd_lookahead_ix] - spline_samples[s_ix]
    # spline_th = np.arctan2(h_spline[1], h_spline[0])  
    # yaw_cmd = (spline_th - pg.r.th)  % (2*np.pi)
    # if yaw_cmd > np.pi:
    #     yaw_cmd -= 2*np.pi

