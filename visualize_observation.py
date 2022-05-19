import shutil
import os.path as osp
import os


#############################
# train dataset
#############################
label = '2683214'
src_dir = f"/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022/train/{label}"
tgt_dir_base = "/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022/visualize"
tgt_dir = f"/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022/visualize/{label}"
if not os.path.exists(tgt_dir_base):
    os.makedirs(tgt_dir_base)
# shutil.copytree(src_dir, tgt_dir)


#############################
# test dataset
#############################
obs = '5763'
img_names = [11930, 11931, 11932, 11933]
src_dir = f"/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022_test/val/images"
tgt_dir_base = "/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022/visualize"
tgt_dir = f"/home/oem/Mingle/datasets/leaf_disease/ClefPlant2022/visualize/obs_{obs}"
if not os.path.exists(tgt_dir_base):
    os.makedirs(tgt_dir_base)
if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

for name in img_names:
    abs_src_name = osp.join(src_dir, str(name) + '.jpg')
    abs_tgt_name = osp.join(tgt_dir, str(name) + '.jpg')
    shutil.copyfile(abs_src_name, abs_tgt_name)