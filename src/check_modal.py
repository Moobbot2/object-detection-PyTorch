import os
import cv2
from xml.etree import ElementTree as ET


def calculate_iou(box1, box2):
    # box: (xmin, ymin, xmax, ymax)

    # Tính toán diện tích chung
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Tính toán diện tích của từng bounding box
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Tính toán IoU
    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)

    return iou


def draw_boxes_on_image(image_path, gt_boxes, pred_boxes):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    for box in gt_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    for box in pred_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    return image

# Your draw_boxes_on_image function

gt_xml_directory = './dataset/annots/label_test'
pred_xml_directory = './dataset/test_predictions/xml'
image_directory = './dataset/images/pano_test'
output_directory = './dataset/check_data'
os.makedirs(output_directory, exist_ok=True)

# Initialize gt_boxes and pred_boxes outside the loop
gt_boxes = {}
pred_boxes = {}

# Process XML files
for xml_filename in os.listdir(gt_xml_directory):
    if xml_filename.endswith('.xml'):
        gt_xml_file_path = os.path.join(gt_xml_directory, xml_filename)
        pred_xml_file_path = os.path.join(pred_xml_directory, xml_filename)

        if not os.path.exists(gt_xml_file_path) or not os.path.exists(pred_xml_file_path):
            print(f"Ground truth or predicted XML file not found for: {xml_filename}")
            continue

        gt_boxes[xml_filename] = []
        tree_gt = ET.parse(gt_xml_file_path)
        root_gt = tree_gt.getroot()
        for obj in root_gt.findall('object/bndbox'):
            box = [int(obj.find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
            gt_boxes[xml_filename].append(box)

        pred_boxes[xml_filename] = []
        tree_pred = ET.parse(pred_xml_file_path)
        root_pred = tree_pred.getroot()
        for obj in root_pred.findall('object/bndbox'):
            box = [int(obj.find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
            pred_boxes[xml_filename].append(box)

        image_name = root_gt.find('filename').text
        image_path = os.path.join(image_directory, image_name)

        if os.path.exists(image_path):
            drawn_image = draw_boxes_on_image(image_path, gt_boxes[xml_filename], pred_boxes[xml_filename])
            if drawn_image is not None:
                output_path = os.path.join(output_directory, f'{image_name}')
                cv2.imwrite(output_path, drawn_image)
        else:
            print(f"Image not found: {image_path}")

# Calculate IoU and print information
iou_threshold = 0.1
for gt_filename, gt_bounding_boxes in gt_boxes.items():
    for pred_filename, pred_bounding_boxes in pred_boxes.items():
        if gt_filename == pred_filename:
            for gt_box in gt_bounding_boxes:
                for pred_box in pred_bounding_boxes:
                    iou = calculate_iou(gt_box, pred_box)
                    if iou > iou_threshold:
                        print(
                            f"Match found!\nGround Truth XML: {gt_filename}, Bounding Box: {gt_box}\nPredicted XML: {pred_filename}, Bounding Box: {pred_box}\nIoU: {iou}\n")
