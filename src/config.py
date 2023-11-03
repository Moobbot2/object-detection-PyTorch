import torch
import os

BATCH_SIZE = 32 #increase/ decrease according to GPU memory
RESIZE_TD = [640,640] # resize the image for training and trainsforms
NUM_EPOCHS = 100 #number epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE:', DEVICE)

ANNOTS_DIR ='./dataset/annots'
IMAGES_DIR ='./dataset/images'
SPLIT_RATIO = 0.2
#classes: 0 index is reserved for background
CLASSES = ['background','cat','dog']
NUM_CLASSES = 3

# whether to visualize images after crearing the data loaders
VISUMLIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = './outputs_new'

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

SAVE_PLOTS_EPOCH = 5 # Save loss plots after these many epochs
SAVE_MODEL_EPOCH = 5 # Save model after these many epochs