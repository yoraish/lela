import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from config import LelaConfig
print("Loaded imports for Dataset.")

class LelaDataset(Dataset):
    # Input image is a scan image.
    # Label is a twofold:
    #   (a) A tensor of gate pose [x, y, th].
    #   (b) The yaw command tensor [yaw_cmd]

    def __init__(self, data_path):
        self.config = LelaConfig()
        sample_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        self.num_samples = len(set([int(f[:f.find(".")]) for f in sample_files if f[0] != '.']))
        self.first_ix = sorted([int(f[:f.find(".")]) for f in sample_files if f[0] != '.'])[0]
        self.sample_ixs = list(set([int(f[:f.find(".")]) for f in sample_files if f[0] != '.']))
        self.img_dir = data_path


    def __len__(self):
        return self.num_samples

    def __getitem__(self, ix):
        # Shift to accomodate non-zero start folders.
        sample_ix = self.sample_ixs[ix]
        img_path = os.path.join(self.img_dir, str(sample_ix) + ".png")
        image = read_image(img_path)
        image = image/torch.sum(image)
        label_path = os.path.join(self.img_dir, str(sample_ix) + ".json")
        with open(label_path, "r") as j:
            label = json.load(j)

            label_gate_pose = torch.tensor([label["gate_x"], label["gate_y"], label["gate_th"]])
            label_yaw_cmd   = torch.tensor([label["yaw_cmd"]])

        return image, label_gate_pose, label_yaw_cmd



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    config = LelaConfig()
    train_dataset = LelaDataset(config.train_data_path)
    test_dataset = LelaDataset(config.test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)

    # Display image and label.
    train_features, train_gate_pose_labels, train_cmd_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_gate_pose_labels.size()}")

    for i in range(len(train_features)):
        img = train_features[i].squeeze()
        label = train_gate_pose_labels[i]
        plt.imshow(img, cmap="gray")
        plt.show()

