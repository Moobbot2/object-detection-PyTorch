import torch
import os

BATCH_SIZE = 16  # increase/ decrease according to GPU memory
RESIZE_TD = [1280, 1280]  # resize the image for training and transform
NUM_EPOCHS = 500  # number epochs to train for

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", DEVICE)

ANNOTS_DIR = "./dataset/annots/label_train"
IMAGES_DIR = "./dataset/images/pano_train"
SPLIT_RATIO = 0.3
# classes: 0 index is reserved for background
CLASSES = ["background", "3", "4", "5"]
NUM_CLASSES = 4

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = "./outputs_new"

# Type Image
IMAGE_TYPE = (".jpg", ".png", ".bmp")

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

SAVE_PLOTS_EPOCH = 5  # Save loss plots after these many epochs
SAVE_MODEL_EPOCH = 5  # Save model after these many epochs