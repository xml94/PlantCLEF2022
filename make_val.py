import os
import shutil
import os.path as osp


# split 10% as validation dataset
a = 1
b = 9
c = 0

src_dir = "/home/oem/Mingle/datasets/PlantCLEF2022/train"
tgt_dir = "/home/oem/Mingle/datasets/PlantCLEF2022_train"

# train, val, test
ratio = [0.1 * a, 0.1 * b, 0.1 * c]

dir_names = sorted(os.listdir(src_dir))
for dir in dir_names:
    num = 0
    abs_dir = osp.join(src_dir, dir)
    total = len(os.listdir(abs_dir))

    os.makedirs(osp.join(tgt_dir, 'train', dir), exist_ok=True)
    os.makedirs(osp.join(tgt_dir, 'val', dir), exist_ok=True)
    os.makedirs(osp.join(tgt_dir, 'test', dir), exist_ok=True)

    for file in os.listdir(abs_dir):
        abs_src_file = osp.join(abs_dir, file)
        if num < total * ratio[0]:
            abs_tgt_file = osp.join(tgt_dir, 'val', dir, file)
            shutil.copyfile(abs_src_file, abs_tgt_file)
        elif num < total * (ratio[0] + ratio[1]):
            abs_tgt_file = osp.join(tgt_dir, 'train', dir, file)
        else:
            abs_tgt_file = osp.join(tgt_dir, 'test', dir, file)
        num += 1
        # shutil.copyfile(abs_src_file, abs_tgt_file)
