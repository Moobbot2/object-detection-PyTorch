import os
import xml.etree.ElementTree as ET
import shutil

# Directory containing Pascal VOC XML files
xml_dir = './dataset/annots/label_train'

# Directory containing img files
image_dir = './dataset/images/pano_train'

# Thư mục đích để di chuyển tệp tin
destination_directory = './dataset/err_img'
if not os.path.isdir(destination_directory):
    os.makedirs(destination_directory)

# List to store all the class names
class_names = []
# List file no class name
list_file_no_class_name = []
# Giá trị tìm trong phần tử <name>
target_values = ['2', 'dog', '43', 'P4']

# Iterate through all XML files in the directory
for filename in os.listdir(xml_dir):
    if filename.endswith('.xml'):
        xml_file = os.path.join(xml_dir, filename)

        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find all <object> elements and extract the <name> tags
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is not None:
                class_name = name.text
                class_names.append(class_name)
            # if name.text == 'dog':
            if name.text in target_values:
                # Tạo đường dẫn mới để di chuyển tệp tin
                new_xml_file_path = os.path.join(destination_directory, filename)
                # Di chuyển tệp tin
                try:
                    shutil.move(xml_file, new_xml_file_path)
                    print(f"Đã di chuyển tệp tin '{os.path.basename(xml_file)}' vào thư mục '{destination_directory}'.")
                    image_link = os.path.join(image_dir, filename.split('.')[0]+'.jpg')
                    print(image_link)
                    new_img_file_path = os.path.join(destination_directory, filename.split('.')[0]+'.jpg')
                    shutil.move(image_link, new_img_file_path)
                    print(f"Đã di chuyển tệp tin '{image_link}' vào thư mục '{destination_directory}'.")
                except:
                    continue
            if name is None:
                file_no_class_name = filename
                list_file_no_class_name.append(class_name)

# Remove duplicates and sort the class names
class_names = list(set(class_names))
class_names.sort()

# Print the list of class names
for class_name in class_names:
    print(class_name)

# Print the list of file no class name
for file_no_class_name in list_file_no_class_name:
    print(class_name)
