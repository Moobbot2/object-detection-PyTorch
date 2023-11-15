import os
import cv2
from xml.etree import ElementTree as ET
from PIL import Image

# Thư mục chứa các tệp tin XML
xml_directory = './dataset/annots/label_test'

# Thư mục chứa ảnh
image_directory = './outputs_new/test_predictions_model_170_bs16_500epoch_7_3/images'

# Thư mục để lưu trữ ảnh đã được vẽ
output_directory = './dataset/check_data'
os.makedirs(output_directory, exist_ok=True)

# Lặp qua tất cả các tệp tin XML trong thư mục
for xml_filename in os.listdir(xml_directory):
    if xml_filename.endswith('.xml'):
        xml_file_path = os.path.join(xml_directory, xml_filename)
        
        # Đọc và phân tích tệp tin XML
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Lấy tên ảnh từ tệp tin XML
        image_name = root.find('filename').text
        image_path = os.path.join(image_directory, image_name)
        
        # Kiểm tra xem ảnh có được đọc thành công không
        if os.path.exists(image_path):
            # Đọc ảnh
            image = cv2.imread(image_path)
            
            # Kiểm tra xem ảnh có tồn tại hay không
            if image is not None:
                # Lấy thông tin bounding box từ tệp tin XML
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    
                    # Vẽ bounding box lên ảnh
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (70, 90, 164), 2)
                    cv2.putText(image, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, lineType=cv2.LINE_AA)
                
                # Lưu ảnh đã được vẽ vào thư mục đích
                output_path = os.path.join(output_directory, f'{image_name}')
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pil_image.save(output_path)
            else:
                print(f"Không thể đọc ảnh: {image_path}")
        else:
            print(f"Ảnh không tồn tại: {image_path}")

print('Xong!')
