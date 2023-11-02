import torch
import cv2
import numpy as np
import os
import glob as glob
from PIL import Image, ImageDraw

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TD, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from ultis import collate_fn, get_train_transform, get_valid_transform

# the dataset class


class CatDogDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.width = width
        self.height = height
        self.classes = classes

        # get all image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")  # .png

        self.all_images = [image_path.split(
            '\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, index):
        # capture the image name and the full image path
        image_name = self.all_images[index]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        print(image.shape)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4]+'.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to "classes" list to get...
            # ... the label index and append to "labels" list
            labels.append(self.classes.index(member.find('name').text))

            # xmin - left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)

            # xmax - right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)

            # ymin - left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)

            # ymax - right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the bounding boxes according to the ...
            # ... desired 'width', 'height'
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # label tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final "target" dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        image_id = torch.tensor([index])
        target['image_id'] = image_id

        # apply the label images
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


# prepare the final datasets and data loaders
train_dataset = CatDogDataset(
    TRAIN_DIR, RESIZE_TD[0], RESIZE_TD[1], CLASSES, get_train_transform())
valid_dataset = CatDogDataset(
    VALID_DIR, RESIZE_TD[0], RESIZE_TD[1], CLASSES, get_valid_transform())

print('train_dataset:', train_dataset)
print('valid_dataset:', valid_dataset)

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
    dataset = CatDogDataset(
        TRAIN_DIR,
        RESIZE_TD[0], RESIZE_TD[1], CLASSES
    )
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample
    def visualize_sample(image, target, save_path=None):
        image = Image.fromarray(image)  # Convert NumPy array to PIL Image
        draw = ImageDraw.Draw(image)

        for box, label in zip(target['boxes'], target['labels']):
            # Convert box coordinates to integers
            box = [int(coord) for coord in box]
            label_text = CLASSES[label]

            # Draw a rectangle around the object
            draw.rectangle(box, outline=(0, 255, 0), width=2)

            # Add a label near the object
            draw.text((box[0], box[1] - 15), label_text, fill=(0, 0, 255))

        image.show()  # Display the image

        if save_path:
            image.save(save_path)  # Optionally, save the annotated image to a file

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)
        print('image.shape:',image.shape)
