import numpy as np
import os
import cv2
import torch
import glob as glob
import matplotlib.pyplot as plt
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, ElementTree

from config import NUM_CLASSES, CLASSES
from model import create_model

# Set the computation device (CPU or GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the model and trained weights
model = create_model(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('./outputs_new/model_50.pth', map_location=device))
model.eval()

# Directory where all the test images are located
DIR_TEST = './dataset/test_data'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

# Classes: 0 index is reserved for background
CLASSES = CLASSES

# Define the detection threshold - any detection with a score below this will be discarded
detection_threshold = 0.8

# Directory to store the output XML files
xml_dir = './dataset/test_predictions/xml/'
os.makedirs(xml_dir, exist_ok=True)

# Function to save the image and its corresponding XML file
def save_image_with_xml(image, image_name, draw_boxes, pred_classes):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Save the image to the test_predictions folder
    pil_image.save(f'./dataset/test_predictions/{image_name}.jpg')

    # Create an XML file in Pascal VOC format
    annotation = Element('annotation')

    # Add basic image information
    folder = SubElement(annotation, 'folder')
    folder.text = 'test_predictions'
    filename = SubElement(annotation, 'filename')
    filename.text = f'{image_name}.jpg'

    # Add object information for each detected box
    for j, box in enumerate(draw_boxes):
        obj = SubElement(annotation, 'object')
        name = SubElement(obj, 'name')
        name.text = pred_classes[j]
        bndbox = SubElement(obj, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(box[0])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(box[1])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(box[2])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(box[3])

    # Save the XML file
    xml_file = os.path.join(xml_dir, f'{image_name}.xml')
    tree = ElementTree(annotation)
    tree.write(xml_file)
    print(f'The bounding box of the {image_name} image has been saved successfully')

# Process each test image
for i in range(len(test_images)):
    # Get the image file name for saving the output later
    image_name = os.path.basename(test_images[i])
    image_name = image_name.split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    
    # Convert color from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Normalize pixel range to [0, 1]
    image /= 255.0
    # Change color channels to the front
    image = np.transpose(image, (2, 0, 1)).astype(float)
    # Convert to a tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # Add a batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    # Move all detections to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    # Continue only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        
        # Filter out boxes based on the detection threshold
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        
        # Get the predicted class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # Draw bounding boxes and write class names on them
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 255, 0), 2)
            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)
        
        # Save the image and generate the XML file
        pil_image = save_image_with_xml(orig_image, image_name, draw_boxes, pred_classes)

        plt.imshow(orig_image)
        plt.axis('off')
        plt.show()
    print(f'Image {i+1} done ...')
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
