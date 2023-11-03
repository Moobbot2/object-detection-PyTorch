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
    


# if __name__ == "__main__":
#    # images_dir = './dataset/images'
#    # image_paths = [os.path.join(images_dir, fname) 
#    #                for fname in os.listdir(images_dir) 
#    #                if fname.endswith(('.jpg', '.png'))] 

#    # train_inds, val_inds = train_test_split(image_paths, test_size=0.1)
#    # # all_imag_name = [image_path.split('\\')[-1] 
#    # #                      for image_path in image_paths]
#    # all_image_names = [os.path.splitext(os.path.basename(image_path))[0] 
#    #                for image_path in image_paths]
#    # print(all_image_names)
   
#    # index = np.random.randint(0, len(all_image_names))
#    # image_name = all_image_names[index]
#    # image_path = os.path.join(images_dir, image_name)
#    # print(image_path)

#    # xml_directory = './dataset/annots'
#    # annots_full_paths = [os.path.join(xml_directory, name + '.xml') 
#    #                           for name in all_image_names]
#    # print(annots_full_paths)
#    from pathlib import Path

def print_directory_tree(path, output_file, indent='-'):
    if os.path.isfile(path):
        output_file.write(indent + os.path.basename(path) + '\n')
    elif os.path.isdir(path):
        output_file.write(indent + f'[{os.path.basename(path)}]\n')
        for item in os.listdir(path):
            print_directory_tree(os.path.join(path, item), output_file, indent + '--')

# Lấy đường dẫn của thư mục hiện tại
current_directory = os.getcwd()

# Mở một file .txt để ghi cây thư mục
output_filename = 'directory_tree.txt'
with open(output_filename, 'w') as output_file:
    # Gọi hàm in cây thư mục bắt đầu từ thư mục hiện tại và ghi vào file
    print_directory_tree(current_directory, output_file)

print(f'Cây thư mục đã được lưu vào file {output_filename}')

