import torch
import cv2
import numpy as np
import os
import glob as glob
import matplotlib.pyplot as plt
from xml.etree import ElementTree as et

from config import (CLASSES, RESIZE_TD, IMAGES_DIR,
                    ANNOTS_DIR, BATCH_SIZE, SPLIT_RATIO, IMAGE_TYPE)
from torch.utils.data import Dataset, DataLoader
from ultis import collate_fn, get_train_transform, get_valid_transform
from sklearn.model_selection import train_test_split


# The dataset class
class CustDataset(Dataset):
    def __init__(self, list_images_path, img_dir, annots_dir, width, height, classes, transforms=None):
        self.transforms = transforms
        self.list_images_path = list_images_path
        self.img_dir = img_dir
        self.annots_dir = annots_dir
        self.width = width
        self.height = height
        self.classes = classes
        self.all_same_names = self.prepare_all_same_names()

    def prepare_all_same_names(self):
        print("--- Check intersection/incompatible---")
        # Get image file names and remove extensions
        image_name_files = [
            os.path.splitext(os.path.basename(image_file))[0]
            for image_file in self.list_images_path
        ]

        # Get the base names of annotation files
        annot_files = [
            os.path.basename(fname)
            for fname in os.listdir(self.annots_dir)
            if fname.endswith(".xml")
        ]
        annot_name_files = [
            os.path.splitext(annot_file)[0] for annot_file in annot_files
        ]

        # Find common names between image and annotation files
        all_same_names = list(
            set(image_name_files).intersection(annot_name_files))

        incompatible_files = [
            image_name
            for image_name in image_name_files
            if image_name not in annot_name_files
        ]
        if not incompatible_files:
            print("All are compatible")
        else:
            for file_name in incompatible_files:
                print(f"File '{file_name}' is incompatible.")
        print("----------")
        return all_same_names

    def __getitem__(self, index):
        # Capture the image name and the full image path
        image_name = self.all_same_names[index]
        image_path = self.find_image_path(image_name)

        # read the image
        image = cv2.imread(image_path)
        print(image_name, ":", image.shape)

        # Check if the image is grayscale (single channel)
        if image.shape[2] == 1:
            # Convert single-channel grayscale to 3-channel grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Convert BGR to RGB color format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # Capture the corresponding XML file for getting the annotations
        annot_filename = image_name + ".xml"
        annot_file_path = os.path.join(self.annots_dir, annot_filename)

        # Get bounding boxes và labels
        boxes, labels = self.extract_annotations(annot_file_path, image.shape)

        # Create target
        target = self.create_target(boxes, labels, index)

        # Apply the label images if has
        if self.transforms:
            sample = self.transforms(
                image=image_resized, bboxes=target["boxes"], labels=labels)
            image_resized = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return image_resized, target

    def find_image_path(self, image_name):
        for extension in IMAGE_TYPE:
            image_path = os.path.join(self.img_dir, image_name + extension)
            if image_path in self.list_images_path:
                return image_path

    def extract_annotations(self, annot_file_path, image_shape):
        boxes, labels = [], []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        image_width, image_height = image_shape[1], image_shape[0]

        for member in root.findall("object"):
            labels.append(self.classes.index(member.find("name").text))
            coords = ["xmin", "xmax", "ymin", "ymax"]
            # Get the coordinate values of the bounding box
            xmin, xmax, ymin, ymax = [
                int(member.find("bndbox").find(coord).text) for coord in coords
            ]
            # xmin, xmax, ymin, ymax = [int(member.find('bndbox').find(coord).text) for coord in ['xmin', 'xmax', 'ymin', 'ymax']]

            # Adjust the bounding box coordinates to the new size
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        return boxes, labels

    def create_target(self, boxes, labels, index):
        try:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "area": area,
                "iscrowd": iscrowd,
            }
            image_id = torch.tensor([index])
            target["image_id"] = image_id

            return target
        except:
            print("image ", torch.tensor([index]))

    def __len__(self):
        return len(self.all_same_names)


# ---------------------------PREPARE DATA-----------------------------------------#

images_dir = IMAGES_DIR
all_image_paths = [os.path.join(images_dir, fname)
                   for fname in os.listdir(images_dir)
                   if fname.endswith(IMAGE_TYPE)]

print("Count Image:", len(all_image_paths))

TRAIN_IMG, TEST_IMG = train_test_split(all_image_paths, test_size=SPLIT_RATIO)

# prepare the final datasets and data loaders
train_dataset = CustDataset(TRAIN_IMG, IMAGES_DIR, ANNOTS_DIR,
                            RESIZE_TD[0], RESIZE_TD[1], CLASSES, get_train_transform())
valid_dataset = CustDataset(TEST_IMG, IMAGES_DIR, ANNOTS_DIR,
                            RESIZE_TD[0], RESIZE_TD[1], CLASSES, get_valid_transform())

print("train_dataset:", train_dataset)
print("valid_dataset:", valid_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

# execute dataset.py using python command form terminal to visualize sample images
# USAGE: python datasets.py
if __name__ == "__main__":
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustDataset(TRAIN_IMG, IMAGES_DIR, ANNOTS_DIR,
                          RESIZE_TD[0], RESIZE_TD[1], CLASSES)
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample

    def visualize_sample(image, target, save_path=None):
        try:
            image_copy = image.copy()
            box = target["boxes"][0]
            label = CLASSES[target["labels"][0]]
            cv2.rectangle(image_copy,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 255, 0), 2,
                          )  # Removed extra parentheses around the color argument

            cv2.putText(image_copy, label,
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

            if save_path:
                # Optionally, save the annotated image to a file
                image.save(save_path)

        except Exception as e:
            print(f"Error while visualizing the image: {e}")

    NUM_SAMPLES_TO_VISUALIZE = 3
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        idx = np.random.randint(0, len(dataset), size=1)
        image, target = dataset[idx[0]]
        visualize_sample(image, target)
        print("image.shape:", image.shape)
        # print(image.shape)