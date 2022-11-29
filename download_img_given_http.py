import os
import os.path as osp
import subprocess
import pandas as pd


def runcmd(cmd, verbose=False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


file_name = "/home/oem/Mingle/datasets/china_island_plant_database/3217A418DF98695E.txt"
save_file_dir = "/home/oem/Mingle/datasets/china_island_plant_database/raw"
dict_save_path = "/home/oem/Mingle/datasets/china_island_plant_database/raw/name_dict.csv"
# This is our shell command, executed in subprocess.

label_name = 0
img_name = 0
class_anchor_name = "random"
src_name, tgt_name = [], []
with open(file_name, 'r+') as file:
    contents = file.readlines()
    for file_name in contents:
        name, webpath = file_name.split()
        class_name, file_name = name.split("_", 1)
        if class_anchor_name == "random":
            class_anchor_name = class_name
            src_name.append(class_name)
            tgt_name.append(str(label_name))
        if class_anchor_name != class_name:
            label_name += 1
            class_anchor_name = class_name
            src_name.append(class_name + '\n')
            tgt_name.append(str(label_name) + '\n')
        # print(file_name)
        abs_dir_name = osp.join(save_file_dir, str(label_name))
        os.makedirs(abs_dir_name, exist_ok=True)
        # abs_img_name = osp.join(abs_dir_name, str(img_name) + '.' + file_name.split('.')[1])
        abs_img_name = osp.join(abs_dir_name, file_name)
        print(abs_img_name)

        runcmd(f"wget {webpath} -O {abs_img_name}", verbose=False)

        img_name += 1

        # if label_name > 3:
        #     break

        dict = {"target_name": tgt_name, "src_name": src_name}
        df = pd.DataFrame(dict)
        df.to_csv(dict_save_path, index=False)

        # with open(dict_save_path, 'w+') as dict_name:
        #     dict_name.writelines(src_name + tgt_name)