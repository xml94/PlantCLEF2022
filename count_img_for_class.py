import os.path as osp
import os
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams["savefig.bbox"] = 'tight'
matplotlib.rcParams['savefig.dpi'] = 1200


num = 100
num_100 = 0
num_larger_than_100 = 0

given_num = 50
num_larger_than = 0

num_all = list(range(100))

num_all_img = 0

train_dir = "/home/oem/Mingle/datasets/PlantCLEF2022/PlantCLEF2022_trust/train"
class_lists = os.listdir(train_dir)
for class_name in class_lists:
    abs_class_dir = osp.join(train_dir, class_name)
    img_num = len(os.listdir(abs_class_dir))
    num_all_img += img_num
    if img_num < 101:
        num_all[img_num - 1] += 1
    else:
        num_larger_than_100 += 1
        print(class_name)
    if img_num == 100:
        num_100 += 1
    if img_num > given_num:
        num_larger_than += 1


print(f"There are {num_all_img} images in total for {len(class_lists)} classes.")
print(f"{num_larger_than_100} classes have more than 100 images.")
print(f"{num_100} classes have 100 images.")
print(f"{num_larger_than} classes have more than {given_num} images.")
plt.plot(num_all, 'b-')
plt.xlabel("Number of images for one class")
plt.ylabel("Number of classes")

plt.savefig("distribution_num_img.png")
