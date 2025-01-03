
import torch
import os
DATASET_PATH = os.path.join("/content/drive/MyDrive/UnetCloudSegmentation/Swinyseg/dataset", "train")

Real_Images_Path = os.path.join(DATASET_PATH, "images")
GT_Path = os.path.join(DATASET_PATH, "masks")
Split_Dataset = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEMORY = True if DEVICE == "cuda" else False 


n_channels = 1
n_classes = 1
n_levels = 3

l_rate = 0.001
n_epochs = 50
BatchSize = 128
img_cols = 300
img_rows = 300
THRESHOLD = 0.5
out_path = "output"
save_model_path = os.path.join(out_path, "unet_tgs_salt.pth")
save_graph_path = os.path.sep.join([out_path, "plot.png"])
test_imgs_path = os.path.sep.join([out_path, "test_imgs_path.txt"])
