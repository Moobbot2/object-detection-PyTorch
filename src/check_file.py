import os

# Thư mục chứa các tệp tin XML
xml_directory = './dataset/annots/label_test'

# Thư mục chứa các tệp tin ảnh
img_directory = './dataset/images/pano_test'

destination_directory = './dataset/err_img'

# Lấy danh sách tên tệp tin trong thư mục xml_directory
xml_files = set([os.path.splitext(file)[0] for file in os.listdir(xml_directory)])

# Lấy danh sách tên tệp tin trong thư mục img_directory
img_files = set([os.path.splitext(file)[0] for file in os.listdir(img_directory)])

# Tìm các tên tệp tin có trong xml_files nhưng không có trong img_files
missing_in_img = xml_files - img_files

# Tìm các tên tệp tin có trong img_files nhưng không có trong xml_files
missing_in_xml = img_files - xml_files

# In ra các tên tệp tin có trong thư mục label_train nhưng không có trong thư mục pano_train
print("Các tên tệp tin có trong label_train nhưng không có trong pano_train:")
for filename in missing_in_img:
    print(filename)

# In ra các tên tệp tin có trong thư mục pano_train nhưng không có trong thư mục label_train
print("Các tên tệp tin có trong pano_train nhưng không có trong label_train:")
for filename in missing_in_xml:
    print(filename)
    # Di chuyển tệp tin vào thư mục đích
    old_img_file_path = os.path.join(img_directory, filename + '.jpg')
    new_img_file_path = os.path.join(destination_directory, filename + '.jpg')
    os.rename(old_img_file_path, new_img_file_path)
    print(f"Đã di chuyển tệp tin '{filename}' vào thư mục '{destination_directory}'")
