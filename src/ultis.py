import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES as classes

# this class keeps track of the training and validation loss values...
# ... and helps to get average for each epoch as well


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total/self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):

    # To handle the data loading as different number
    # of object and to handle varying size tensors as well
    return tuple(zip(*batch))

# define the training tranforms


def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# define the validation tranforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def show_transformed_image(train_loader):
    if len(train_loader) > 0:
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]

            image = images[0].permute(1, 2, 0).cpu().numpy()
            image_copy = np.copy(image)

            for box in targets[0]['boxes'].cpu().numpy().astype(np.int32):
                cv2.rectangle(
                    image_copy, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            # Display the transformed image using Matplotlib
            plt.imshow(image_copy)
            plt.axis('off')
            plt.show()
