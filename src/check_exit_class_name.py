import os
from xml.etree import ElementTree as ET

# Thư mục chứa các tệp tin XML
xml_directory = './dataset/annots/label_train'

# Thư mục đích để lưu trữ các tệp XML không có phần tử <name>
destination_directory = './dataset/err_img_name'
if not os.path.isdir(destination_directory):
    os.makedirs(destination_directory)

# Lặp qua tất cả các tệp tin XML trong thư mục
for xml_filename in os.listdir(xml_directory):
    if xml_filename.endswith('.xml'):
        xml_file_path = os.path.join(xml_directory, xml_filename)
        
        # Đọc và phân tích tệp tin XML
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Tìm phần tử <name>
        name_element = root.find('object/name')
        
        # Kiểm tra xem phần tử <name> có tồn tại hay không
        if name_element is None:
            print(xml_file_path)
            # Di chuyển tệp tin vào thư mục đích
            new_xml_file_path = os.path.join(destination_directory, xml_filename)
            os.rename(xml_file_path, new_xml_file_path)
            print(f"Đã di chuyển tệp tin '{xml_filename}' vào thư mục '{destination_directory}'")
