import os
import glob

from config import TRAIN_DIR  # Assuming TRAIN_DIR is defined in config.py


if not os.path.isdir('./train'):
    print('/train', 'This folder has not been found. Enter correct path.')
else:
    print("Folder path is correct!")
    
