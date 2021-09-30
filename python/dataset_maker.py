            
import numpy as np
import matplotlib.pyplot as plt
import time
import PIL
import json
import os
import random
import shutil

from cubic_spline_with_derivative import sampleCubicSplinesWithDerivative
from robot import Robot
from gate import Gate
from config import LelaConfig
from playground import Playground

def create_dataset(num_samples):
    # Config instance.
    config = LelaConfig()

    # Check how many samples already exist in dataset and add to those.
    sample_files = [f for f in os.listdir(config.unbalanced_data_path) if os.path.isfile(os.path.join(config.unbalanced_data_path, f))]
    count_start = 0
    if len(sample_files) > 4:
        if sample_files:
            count_start = sorted([int(f[:f.find(".")]) for f in sample_files if f[:f.find(".")] != ''])[-1] + 1

    sample_count = 0 + count_start

    while True:

        # Set up a robot.
        r = Robot(x = np.array([1,2]), h = np.array([0, 0.3]))

        # Put down some gates.
        # Set the first gate to be in front of the robot ish, then erase it later.
        gates = [Gate(name = 0, width = 30, frame_width = 4, frame_length = 4, rotation= np.pi/2, translation=np.array([[1 ,  30 + np.random.rand()]]))]


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
        pg = Playground(r, gates)


        # Draw a Hermite spline between the points. We need (x,y) gate-center points, and dy/dx which is the slope of the line at the gate.
        xs = [pg.r.x[0]]
        ys = [pg.r.x[1]]
        points = [pg.r.x]
        tangents = [np.array([np.cos(pg.r.th), np.sin(pg.r.th)])]
        dydx = [pg.r.th]
        for g in pg.gates:
            points.append(g.translation.reshape(2))
            tangents.append(np.array([np.cos(g.rotation), np.sin(g.rotation)]))
            # xs.append(g.translation[0][0])
            # ys.append(g.translation[0][1])
            # Set all slopes to 0.
            # dydx.append(g.rotation - 1.5*np.pi) # ****************************

        # spline = scipy.interpolate.CubicHermiteSpline(xs, ys, dydx, )
        spline_samples = sampleCubicSplinesWithDerivative(points, tangents, 4)
        vis_spline_x = []
        vis_spline_y = []

        for s in spline_samples:
            vis_spline_x.append(s[0])
            vis_spline_y.append(s[1])
        

        # pg.vis(draw_robot = True, draw_gates= True, draw_scan=True, additional = {'x':vis_spline_x, 'y':vis_spline_y})
        

        for s_ix in range(len(spline_samples)-1 - config.cmd_lookahead_ix):
            # NOTE(yoraish): this is where we can vary the position and find labels (heading is random and label is the heading correction to look ahead a certain number of samples on the spline).
            # Heading array (2,).
            h = spline_samples[s_ix +1] - spline_samples[s_ix]
            # Robot position.
            x = spline_samples[s_ix]
            # Update robot.
            pg.r.set_x_h(x,h)

            # Build a rotation matrix for the robot pose. Use the inverse of that matrix to shift the scan to be axis aligned.
            # Create an empty scan.
            scan = np.zeros((2*pg.r.lidar_max_dist+1, 2*pg.r.lidar_max_dist+1))
            
            # Get scan hit points (from the lidar).
            scan_points, g_name = pg.get_scan()
            # Shift the points to the center.
            scan_points -= pg.r.x

            # Rotate points to correct for robot rotation.
            # NOTE(yoraish): The rotation matrix is negated to rotate to the "wrong" side. Later, we will flip the x-axis of the resulting scan image. This takes care of the fact numpy handles arrays as row-column and not x,y.
            R_minus = -np.array([[np.cos(pg.r.th), -np.sin(pg.r.th)],
                            [np.sin(pg.r.th), np.cos(pg.r.th)]])
            # R_inv = np.linalg.inv(R)
            R_minus_inv = R_minus.T
            # Rotate points.
            scan_points = R_minus_inv.dot(scan_points.T).T

            # Voxelize to the 1x1cm level. Also Shift the point (0,0) to the center of the image, so add lidar_max_dist to both x and y values.
            scan_points = np.unique(scan_points.astype(np.int), axis=0) + pg.r.lidar_max_dist
            # Invert x axis.
            scan_points = scan_points* np.array([[-1,1]]) + 2*np.array([[pg.r.lidar_max_dist, 0]])

            # Shift the point (0,0) to the center of the image, so add lidar_max_dist to both x and y values.
            for sp in scan_points:
                scan[sp[1]][sp[0]] = 255

            # Dataset creation.
            # Scan image.
            # Json with (a) relative gate pose and (b) expected rotation (yaw in radians with 0 pointing forward).

            # First, the relative position of the gate to the robot. In form dx, dy, theta. Where theta is the gate angle wrt the robot heading. So straight ahead is 0 radians and slightly turned left is small positive value. Slightly turned right is large positive value (close to 2pi).
            # To get this relative position inverse rotation the gate pose with the robot rotation, shift to the origin by subtracting the robot position (x), and subtract the robot rotation from the gate rotation (mod 2pi).
            g_near = pg.gates[g_name]

            # Rotation of gate along its axis.
            g_rel_th = (g_near.rotation - pg.r.th) % (2*np.pi)

            # Angle between robot and gate center point.
            g_near_centered_r =  g_near.translation.T - np.expand_dims(pg.r.x, axis = 0).T
            R = np.array([[np.cos(pg.r.th), -np.sin(pg.r.th)],
                        [np.sin(pg.r.th), np.cos(pg.r.th)]])

            g_rel_pos = (R.T).dot(g_near_centered_r).T



            # gate_x and gate_y and gate_th for dataset.
            gate_x = g_rel_pos[0][0]
            gate_y = g_rel_pos[0][1]
            gate_th = g_rel_th

            # yaw_command for dataset. This is robot heading r.th minus spline heading `h`. Between -pi and pi.
            h_spline = spline_samples[s_ix + config.cmd_lookahead_ix] - spline_samples[s_ix]
            spline_th = np.arctan2(h_spline[1], h_spline[0])  
            yaw_cmd = (spline_th - pg.r.th)  % (2*np.pi)
            if yaw_cmd > np.pi:
                yaw_cmd -= 2*np.pi


            # Drop samples where gate is said to be behind robot (negative x component) or just very small.
            if gate_x < config.dataset_gate_x_min:
                continue
            
            # Otherwise, save the data.
            dataset_entry = {"gate_x": gate_x, "gate_y":gate_y, "gate_th": gate_th, "yaw_cmd": yaw_cmd}

            # Sample path.
            json_path = os.path.join(config.unbalanced_data_path, str(sample_count) + ".json")
            img_path = os.path.join(config.unbalanced_data_path, str(sample_count) + ".png")
            
            im = PIL.Image.fromarray(scan.astype(np.uint8))
            # Change the size of the scan to be the specified.
            im = im.resize([config.img_size ,config.img_size],PIL.Image.ANTIALIAS)

            json_string = json.dumps(dataset_entry)
            with open(json_path, "w") as j:
                j.write(json_string)
            im.save(img_path)

            # Save figure with text.
            figure_text = """Sample {}.\nGate rel. angle: {:10.4f}\nGate rel. pos:   {}\nYaw command:(rads) {}\n            (degs){}""".format(sample_count, g_rel_th, g_rel_pos, yaw_cmd, yaw_cmd/(2*np.pi)*360)
            
            # Save animation figure?
            pg.vis(draw_robot = True, draw_gates= True, draw_scan=True, img = scan, text = figure_text, save_path = os.path.join("/Users/yoraish/code/lela/python/animation",str(sample_count)+".png"))

            # Check if done.
            if sample_count > num_samples + count_start:
                return

            # Increment sample counter.
            print(sample_count)
            sample_count += 1

def balance_dataset():
    # Keep list of all names and thetas of samples in train_data.
    config = LelaConfig()
    data_path = config.unbalanced_data_path
    sample_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) ]
    sample_ixs = list(set([int(f[:f.find(".")]) for f in sample_files if f[0] != '.']))
    
    # Read all labels.
    labels = []
    label_fnames = []
    for sample_ix in sample_ixs:
        label_path = os.path.join(data_path, str(sample_ix) + ".json")
        with open(label_path, "r") as j:
            label = json.load(j)
            labels.append(label)
            label_fnames.append(label_path)
    
    n_bins = 50

    # Shuffle labels and fnames.
    a = list(zip(labels, label_fnames))
    random.shuffle(a)
    labels, label_fnames = zip(*a)

    for key in ["yaw_cmd", "gate_th", "gate_x", "gate_y"]:
        key_data = []

        # Go through the data and read all the values.
        for label in labels:
            key_data.append(label[key])
        print(len(key_data)," -> ", end="")


        # With all the information, create histogram with n bins. 
        bin_to_sample_count, bins = np.histogram(key_data, bins = 20)
        # Map sample ix to the bin bumber it belongs to.
        sample_to_bin = np.digitize(key_data, bins[:-1])-1
        # Map sample ix to the probability of seing their bin in the dataset = (sample-count in its bin) / all samples.
        func_bin_to_freq = lambda x: bin_to_sample_count[x]
        sample_to_prob = func_bin_to_freq(sample_to_bin) / len(key_data)
        # Flip probabilities to get inverted sampling (sample more of underrepresented bins).
        sample_to_inv_prob = 1 - sample_to_prob


        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.hist(key_data, bins = bins)
        ax.set_title("Unbalanced dataset key {}.".format(key))
        ax.set_xlabel(key)
        ax.set_ylabel("Count")   

        median_bin_count = np.median([s for s in bin_to_sample_count if s < np.max(bin_to_sample_count) and s > np.min(bin_to_sample_count) ])*2

        chosen_labels = []
        chosen_label_fnames = []
        bin_to_num_chosen = {b:0 for b in range(len(bin_to_sample_count))}
        for ix in range(len(key_data)):
            sample_bin = sample_to_bin[ix]
            if bin_to_num_chosen[sample_bin] < median_bin_count:
                chosen_labels.append(labels[ix])
                chosen_label_fnames.append(label_fnames[ix])
                bin_to_num_chosen[sample_bin] += 1

        # chosen_labels = []
        # for ix in range(len(key_data)):
        #     rand_num = np.random.rand() 
        #     if rand_num < sample_to_inv_prob[ix]:
        #         chosen_labels.append(labels[ix])

        chosen_key_data = [label[key] for label in chosen_labels]
        labels = chosen_labels
        label_fnames = chosen_label_fnames
        print(len(labels))

        ax2 = fig.add_subplot(132)
        ax2.set_title("Balanced dataset key {}.".format(key))
        ax2.hist(chosen_key_data, bins = bins)
        ax2.set_xlabel(key)
        ax2.set_ylabel("Count")  
        ax2.plot([0, bins[-1]], [median_bin_count,median_bin_count])



        ax3 = fig.add_subplot(133)
        ax3.set_title("Distribution of selection probabilities.")
        ax3.hist(sample_to_inv_prob, bins = 20)
        ax3.set_xlabel("probability")
        ax3.set_ylabel("Count")    
        

        plt.show()


    # Move all chosen labels to the train and test folders.
    num_test = len(labels)*LelaConfig.train_dataset_fraction
    
    for i, label_fname in enumerate(label_fnames):
        img_fname = label_fname[:label_fname.find(".")] + ".png"
        if i < num_test:
            shutil.copy(label_fname, LelaConfig.test_data_path)
            shutil.copy(img_fname, LelaConfig.test_data_path)

        else:
            shutil.copy(label_fname, LelaConfig.train_data_path)
            shutil.copy(img_fname, LelaConfig.train_data_path)


        

if __name__ == "__main__":
    balance_dataset()
    # create_dataset(30000)