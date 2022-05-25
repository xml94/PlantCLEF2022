# first step: download original train and test dataset
# Look like this
#   PlantCLEF2022/train
#   PlantCLEF2022/test


# second step: make validation dataset
# give correct directory in make_val.py
python make_val.py

# third step: make the final training dataset
# looks like this:
#ClefPlant2022_train/
#├── train
#└── val
# In this directory, train is original training dataset (all)
# val is from the second step, 10% of the original training dataset
cp -r /home/oem/Mingle/datasets/PlantCLEF2022/train /home/oem/Mingle/datasets/PlantCLEF2022_test


# fourth step: make the final testing dataset
# looks like this:
#ClefPlant2022_test/
#├── train
#└── val
# here, train is actually the validation dataset made in second step
# val is actually the real testing dataset
cp -r /home/oem/Mingle/datasets/PlantCLEF2022_train/val /home/oem/Mingle/datasets/PlantCLEF2022_test
mv /home/oem/Mingle/datasets/PlantCLEF2022_test/val /home/oem/Mingle/datasets/PlantCLEF2022_test/train
cp -r /home/oem/Mingle/datasets/PlantCLEF2022/test /home/oem/Mingle/datasets/PlantCLEF2022_val
mv /home/oem/Mingle/datasets/PlantCLEF2022_test/test /home/oem/Mingle/datasets/PlantCLEF2022_test/val