from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET


def process_xml_file(xml_file_path):
    boxes = {"boxes": [], "names": []}
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for obj in root.findall("object/bndbox"):
        box = [
            int(obj.find(coord).text) if obj.find(coord) is not None else 0
            for coord in ["xmin", "ymin", "xmax", "ymax"]
        ]
        boxes["boxes"].append(box)
        name_element = obj.find("name")
        name = name_element.text if name_element is not None else ""
        boxes["names"].append(name)
    return boxes


def evaluate_model(xml_test_directory, xml_pred_directory):
    true_labels = []
    pred_labels = []

    for xml_filename in os.listdir(xml_test_directory):
        if xml_filename.endswith(".xml"):
            xml_test_file_path = os.path.join(xml_test_directory, xml_filename)
            xml_pred_file_path = os.path.join(xml_pred_directory, xml_filename)

            if not os.path.exists(xml_test_file_path):
                print(f"Ground truth XML file not found for: {xml_filename}")
                continue

            gt_boxes = process_xml_file(xml_test_file_path)

            if os.path.exists(xml_pred_file_path):
                pred_boxes = process_xml_file(xml_pred_file_path)
                # Extend pred_labels with unique predicted labels
                pred_labels.extend(set(pred_boxes["names"]))
            else:
                print(f"Predicted XML file not found for: {xml_filename}")

            # Extend true_labels with unique ground truth labels
            true_labels.extend(set(gt_boxes["names"]))

    if not true_labels or not pred_labels:
        print("No samples for evaluation.")
        return

    # Continue with metric calculations
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    roc_auc = roc_auc_score(true_labels, pred_labels, average="weighted")
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=set(true_labels),
        yticklabels=set(true_labels),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


xml_test_directory = "./dataset/annots/label_test"
xml_pred_directory = "./dataset/annots/model_500_xml_pred"
# Gọi hàm với các thư mục của bạn
evaluate_model(xml_test_directory, xml_pred_directory)
