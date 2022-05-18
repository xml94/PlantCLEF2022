"""
Visualize a dataset for classification
input: a directory
    class_1: img_1, img_2, ...
    class_2: img_1, img_2, ...
    ...
output: a directory
    class_1_img, class_2_img, ...
"""
import os
import shutil
import numpy as np

src_dir = '/home/oem/Mingle/datasets/leaf_disease/Rice_leaf_2022'
tgt_dir = src_dir + '_vis'
if os.path.exists(tgt_dir):
    shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir)
else:
    os.makedirs(tgt_dir)

for root, dirs, files in os.walk(src_dir):
    for dir_name in dirs:
        abs_dir = os.path.join(root, dir_name)
        file_list = os.listdir(abs_dir)
        if len(file_list) > 0:
            random = np.random.randint(len(file_list))
            src_file_name = os.path.join(abs_dir, file_list[random])
            tgt_file_name = os.path.join(tgt_dir, dir_name + '.jpg')
            shutil.copyfile(src_file_name, tgt_file_name)
