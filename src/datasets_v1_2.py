import torch
import cv2
import numpy as np
import os
import glob as glob
import matplotlib.pyplot as plt

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TD, IMAGES_DIR, ANNOTS_DIR, BATCH_SIZE, SPLIT_RATIO
from torch.utils.data import Dataset, DataLoader
from ultis import collate_fn, get_train_transform, get_valid_transform
from sklearn.model_selection import train_test_split

# the dataset class


class CatDogDataset(Dataset):
    def __init__(self, img_dir, annots_dir, width, height, classes, transforms=None):
        self.transforms = transforms
        self.img_dir = img_dir
        self.annots_dir = annots_dir
        self.width = width
        self.height = height
        self.classes = classes

        # Get image file names and remove extensions
        image_files = [os.path.basename(fname) for fname in os.listdir(self.img_dir) if fname.endswith(('.jpg', '.png'))]
        # image_files = [os.path.splitext(fname)[0] for fname in os.listdir(self.img_dir) if fname.endswith('.jpg', '.png')]
        image_name_files = [os.path.splitext(image_file)[0] for image_file in image_files]
        self.all_names_img = sorted(image_name_files)

        # Get annotation file names and remove extensions
        annot_files = [os.path.basename(fname) for fname in os.listdir(self.img_dir) if fname.endswith(('.xml'))]
        # annot_files = [os.path.splitext(fname)[0] for fname in os.listdir(self.annots_dir) if fname.endswith('.xml')]
        annot_name_files = [os.path.splitext(annot_file)[0] for annot_file in annot_files]
        self.all_names_annot = sorted(annot_name_files)

        # Find common names between image and annotation files
        self.all_same_names = sorted(set(self.all_names_img).intersection(self.all_names_annot))


    def __getitem__(self, index):
        # capture the image name and the full image path
        image_name = self.all_same_names[index]
        image_path = os.path.join(self.img_dir, f"{image_name}.jpg" if f"{image_name}.jpg" in image_files else f"{image_name}.png")

        # read the image
        image = cv2.imread(image_path)
        print(image.shape)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name +'.xml'
        annot_file_path = os.path.join(self.annots_dir, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(self.annots_full_paths[index])
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
        return len(self.img_dir)

#---------------------------PREPARE DATA-----------------------------------------#

images_dir = IMAGES_DIR
all_image_paths = [os.path.join(images_dir, fname) 
                for fname in os.listdir(images_dir) 
                if fname.endswith(('.jpg', '.png'))]
TRAIN_IMG, TEST_IMG = train_test_split(all_image_paths, test_size=SPLIT_RATIO)

# prepare the final datasets and data loaders
train_dataset = CatDogDataset(TRAIN_IMG, ANNOTS_DIR, 
                              RESIZE_TD[0], RESIZE_TD[1], CLASSES, 
                              get_train_transform())
valid_dataset = CatDogDataset(TEST_IMG, ANNOTS_DIR, 
                              RESIZE_TD[0], RESIZE_TD[1], CLASSES, 
                              get_valid_transform())

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
    dataset = CatDogDataset(TRAIN_IMG, ANNOTS_DIR, RESIZE_TD[0], RESIZE_TD[1], CLASSES)
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample

    def visualize_sample(image, target, save_path=None):
        try:
            image_copy = image.copy()
            box = target['boxes'][0]
            label = CLASSES[target['labels'][0]]
            cv2.rectangle(image_copy,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 255, 0), 2)  # Removed extra parentheses around the color argument

            cv2.putText(image_copy, label, (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

            if save_path:
                # Optionally, save the annotated image to a file
                image.save(save_path)

        except Exception as e:
            print(f"Error while visualizing the image: {e}")

    NUM_SAMPLES_TO_VISUALIZE = 10
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        idx = np.random.randint(0, len(dataset), size=1)
        image, target = dataset[idx[0]]
        visualize_sample(image, target)
        print('image.shape:', image.shape)
