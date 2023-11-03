# import OS module
import os
import shutil

# Get the list of all files and directories
# path = "E://Giap_storage//test_pano_02_06_23//khoanh_vung_vqc/"
path = "F:\DuLieuRang\object-detection-PyTorch\dataset\images\PANO_ALL"
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

# prints all files
filenames=[]
for i in range (len(dir_list)):
    # print(os.path.splitext(dir_list[i])[0])
    filenames.append(os.path.splitext(dir_list[i])[0])
print ('size ',len(dir_list))
print(filenames)
# print (filenames)
# source_path = "E://Giap_storage//test_pano_02_06_23//PANO_imgs/"
# target_path = "E://Giap_storage//test_pano_02_06_23//match_pano_imgs/"
# soure_dir_list = os.listdir(source_path)
# for j in range (len(soure_dir_list)):
#     current_file =os.path.splitext(soure_dir_list[j])[0]
#     print(current_file)
#     exit = 0
#     for k in range (len(filenames)):
#         if current_file.__eq__(filenames[k]):
#             exit=1
#             break
#     if exit==1:
#         # copy
#         print('copy')
#         shutil.copy(source_path + soure_dir_list[j], target_path + soure_dir_list[j])

# target_path = "E://Giap_storage//test_pano_02_06_23//match_pano_imgs/"
# target_dir_list = os.listdir(target_path)
# tagetfilenames=[]
# for i in range (len(target_dir_list)):
#     print(os.path.splitext(target_dir_list[i])[0])
#     tagetfilenames.append(os.path.splitext(target_dir_list[i])[0])
# for j in range (len(filenames)):
#     exit = 0
#     for k in range (len(target_dir_list)):
#         if filenames[j].__eq__(tagetfilenames[k]):
#             exit=1
#             break
#     if exit==0:
#         print('not exist'
#         )
#         print(filenames[j])
#         os.remove(path+dir_list[j])