import shutil

import pandas as pd
import os.path as osp
import os
import csv


epoch=100
meta_file = "/home/oem/Mingle/datasets/FungiCLEF2022/FungiCLEF2022_test_metadata.csv"
pred_dir = f"clef_fungi_results_epoch{epoch}"
out_name = f"fungi_submission_epoch{epoch}.csv"
if osp.exists(out_name):
    os.remove(out_name)



# pred = pd.read_csv(pred_file, index_col=False, header=None)
meta = pd.read_csv(meta_file, header=0)
# print(meta.columns.to_list())
# print(meta.values[0][1])
# print(len(meta.values))
# print(meta.values[1][0])
# print(len(meta.values[0]))
# 6
# print(len(meta.values.T))
history_obs_id = '3305985310'
history_score = 0
value = meta.values
content = "ObservationId,ClassId\n"
with open(out_name, 'a+') as file:
    file.writelines(content)
total_content= []
threshold = 0.08
for i in range(len(value)):
    obs_id = value[i][1]
    if str(obs_id) != str(history_obs_id) and i > 0:
        total_content.append(content)
        with open(out_name, 'a+') as file:
            write = csv.writer(file, delimiter=',')
            write.writerow(content)

        csv_id = str(value[i][-1]).replace('.JPG', '.csv')
        csv_id = osp.join(pred_dir, csv_id)
        # print(f"checking obs_id {obs_id}")
        # print(history_obs_id)
        # print(f"checking csv_id {csv_id}")
        # print(csv_id)
        class_id = pd.read_csv(csv_id, header=None, sep=";").values.T[0][0]

        # print(len(class_score_rank))
        # print(len(class_score_rank))
        history_score = pd.read_csv(csv_id, header=None, sep=";").values[0, 1]
        # print(f"checking {history_score}")

        if history_score > threshold:
            content = [int(obs_id), int(class_id)]
        else:
            content = [int(obs_id), -1]
        history_obs_id = int(obs_id)
    else:
        csv_id = str(value[i][-1]).replace('.JPG', '.csv')
        csv_id = osp.join(pred_dir, csv_id)
        score = pd.read_csv(csv_id, header=None, sep=";").values[0, 1]
        if score > history_score or i == 0:
            history_score = score
            class_id = pd.read_csv(csv_id, header=None, sep=";").values.T[0][0]
            if score > threshold:
                content = [int(obs_id), int(class_id)]
            else:
                content = [int(obs_id), -1]

    if i == len(value) - 1:
        with open(out_name, 'a+') as file:
            write = csv.writer(file, delimiter=',')
            write.writerow(content)

# with open(out_name, 'w+') as file:
#     file.writelines(str(total_content))