import torch
import cv2
import numpy as np
import os
import glob as glob
import matplotlib.pyplot as plt

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TD, BATCH_SIZE, ANNOTS_DIR, IMAGES_DIR
from torch.utils.data import Dataset, DataLoader
from ultis import collate_fn, get_train_transform, get_valid_transform


from sklearn.model_selection import train_test_split

# if __name__ == "__main__":
#    image_paths = glob.glob(f"{IMAGES_DIR}/*.jpg")
#    annots_paths = glob.glob(f"{IMAGES_DIR}/*.jpg")
#    train_inds, val_inds = train_test_split(range(len(image_paths)), test_size=0.1)
#    print(image_paths)
   # for idx in val_inds:
      # print(idx)
   #  print( os.path.basename(image_paths[idx]), '--', os.path.basename(annots_paths[idx]))
    


if __name__ == "__main__":
   images_dir = './dataset/images'
   image_paths = [os.path.join(images_dir, fname) 
                  for fname in os.listdir(images_dir) 
                  if fname.endswith(('.jpg', '.png'))] 

   train_inds, val_inds = train_test_split(image_paths, test_size=0.1)
   # all_imag_name = [image_path.split('\\')[-1] 
   #                      for image_path in image_paths]
   all_image_names = [os.path.splitext(os.path.basename(image_path))[0] 
                  for image_path in image_paths]
   print(all_image_names)
   
   index = np.random.randint(0, len(all_image_names))
   image_name = all_image_names[index]
   image_path = os.path.join(images_dir, image_name)
   print(image_path)

   xml_directory = './dataset/annots'
   annots_full_paths = [os.path.join(xml_directory, name + '.xml') 
                             for name in all_image_names]
   print(annots_full_paths)




    