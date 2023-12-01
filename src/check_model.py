import os
import cv2
from xml.etree import ElementTree as ET
import sys
import time
from datetime import datetime
import csv
from PIL import Image

from config import IMAGE_TYPE

gt_xml_directory = './dataset/annots/label_test'
pred_xml_directory = './dataset/test_predictions/xml'
image_directory = './dataset/images/pano_test'
output_directory = './dataset/check_data'
csv_output_path = 'output_results.csv'  # Specify the CSV file path
os.makedirs(output_directory, exist_ok=True)

def find_image_path(img_dir, image_name):
    for extension in IMAGE_TYPE:
        image_path = os.path.join(img_dir, f"{image_name}{extension}")
        if os.path.exists(image_path):
            return image_path
    return None  # Return None if the image is not found


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)

    return iou


def draw_boxes_on_image(image, gt_boxes, pred_boxes):
    for j, box in enumerate(gt_boxes['boxes']):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image, gt_boxes['names'][j],
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    lineType=cv2.LINE_AA)

    for j, box in enumerate(pred_boxes['boxes']):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(image, pred_boxes['names'][j],
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                    lineType=cv2.LINE_AA)

    return image



def process_xml_file(xml_file_path):
    boxes = {'boxes': [], 'names': []}
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        box_element = obj.find('bndbox')
        box = [int(box_element.find(coord).text) if box_element.find(coord) is not None else 0 for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
        boxes['boxes'].append(box)
        name_element = obj.find('name')
        name = name_element.text if name_element is not None else ''
        boxes['names'].append(name)
    return boxes

def log_message(message):
    print(f"{datetime.now()} - {message}")

def process_image(image_path, gt_boxes, pred_boxes, output_directory, xml_filename, csv_writer = ''):
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        drawn_image = draw_boxes_on_image(image, gt_boxes, pred_boxes)
        if drawn_image is not None:
            output_path = os.path.join(output_directory, f"{xml_filename.split('.')[0]}.png")
            pil_image = Image.fromarray(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
            pil_image.save(output_path)

            # Calculate IoU and write to CSV
            for i, gt_box in enumerate(gt_boxes['boxes']):
                for j, pred_box in enumerate(pred_boxes['boxes']):
                    iou = calculate_iou(gt_box, pred_box)
                    if gt_boxes['names'][i] == pred_boxes['names'][j]:
                        csv_writer.writerow(
                            [xml_filename, gt_boxes['names'][i], gt_box, pred_box, iou])
    else:
        print(f"{datetime.now()} - Image not found: {image_path}")


def process_image_and_xml(gt_xml_directory, pred_xml_directory, image_directory, output_directory, csv_output_path):
    with open(csv_output_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write CSV header
        csv_writer.writerow(['Filename', 'Name', 'Box_label', 'Box_model', 'IoU'])

        # Process XML files
        for xml_filename in os.listdir(gt_xml_directory):
            if xml_filename.endswith('.xml'):
                gt_xml_file_path = os.path.join(gt_xml_directory, xml_filename)
                pred_xml_file_path = os.path.join(pred_xml_directory, xml_filename)

                if not os.path.exists(gt_xml_file_path) or not os.path.exists(pred_xml_file_path):
                    log_message(f"Ground truth or predicted XML file not found for: {xml_filename}")
                    continue

                gt_boxes = process_xml_file(gt_xml_file_path)
                pred_boxes = process_xml_file(pred_xml_file_path)

                image_name, _ = os.path.splitext(os.path.basename(gt_xml_file_path))
                image_path = find_image_path(image_directory, image_name)

                process_image(image_path, gt_boxes, pred_boxes, output_directory, xml_filename, csv_writer)
                # process_image(image_path, gt_boxes, pred_boxes, output_directory, xml_filename)

                # Calculate IoU and print information
                iou_threshold = 0.1
                for i, gt_box in enumerate(gt_boxes['boxes']):
                    for j, pred_box in enumerate(pred_boxes['boxes']):
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > iou_threshold and gt_boxes['names'][i] == pred_boxes['names'][j]:
                            log_message(f"Match found!\nGround Truth XML: {xml_filename}, Bounding Box: {gt_box}, Name: {gt_boxes['names'][i]}\nPredicted XML: {xml_filename}, Bounding Box: {pred_box}, Name: {pred_boxes['names'][j]}\nIoU: {iou}")


# Create a log file with timestamps
log_file_path = f'log/history_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
# with open(log_file_path, 'w') as log_file:
    # Redirect stdout to the log file
    # sys.stdout = log_file
process_image_and_xml(gt_xml_directory, pred_xml_directory, image_directory, output_directory, csv_output_path)

# Restore the original stdout
# sys.stdout = sys.__stdout__

log_message('Done! Log saved to ' + log_file_path)
