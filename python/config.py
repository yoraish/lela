from enum import Enum

"""
Configurations and hyperparameters for the LELA project.
"""
class LelaConfig():

    ################
    # ROBOT
    ################
    # Max lidar lookahead in cm.
    lidar_max_dist = 60

    # Gate Placement.

    # Rotation of next gate can go between -value/2 and value/2.
    gate_rot_rand_scaler = 1.7

    # Translation of next gate (along previous gate's direction) can go between [gate_trans_min_dist, gate_trans_min_dist + gate_trans_rand_scaler]
    gate_trans_min_dist = 40
    gate_trans_rand_scaler = 10

    # Gate sizes.

    #############
    # Dataset.
    #############

    # How near can dataset entries can be to the robot along x axis. In cm.
    dataset_gate_x_min = 2
    # How many gates in a run.
    num_gates_in_run = 4
    # How many spline samples ahead should the controller seek to drive to.
    cmd_lookahead_ix = 5
    # Dataset path.
    unbalanced_data_path = "/Users/yoraish/code/lela/python/unbalanced_data"
    train_data_path = "/Users/yoraish/code/lela/python/train_data"
    test_data_path  = "/Users/yoraish/code/lela/python/test_data"

    # Test train ratio.
    train_dataset_fraction = 0.1

    # Image sizes.
    img_size = lidar_max_dist*2 +1


    ##############
    # Network.
    ##############

    # Design.
    # Latent vector size.
    n_z = 10

    # Training.
    epochs = 4000
    batch_size = 1
    lr = 1e-4
    num_workers = 10

    # Saving the mode.
    model_path = {"DRONET": "DRONET.pt",
                    "VAE_IMG": "VAE_IMG.pt",
                    "VAE_IMG_CMD":"VAE_IMG_CMD.pt"}

    # Model class.
    class ModelClass(Enum):
        DRONET = 0
        VAE_IMG = 1
        VAE_IMG_CMD = 2

    model_class = ModelClass.DRONET.name


    def __init__(self) -> None:
        pass