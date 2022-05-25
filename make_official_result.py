import pandas as pd
import os.path as osp
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='sorted',
                    choices=['single_low', 'single_high', 'sorted', 'random'])
# how to combine observations:
# single: chose the biggest or lowest top-1 score and its rank from single image
    # low: chose the lowest top-1 score
    # high: chose the highest top-1 score
# sorted: sort by all ranks and get the top-30 for all images
# mode = 'single_img'
parser.add_argument('--epoch', type=int, required=True,
                    help='which epoch you want to check')
parser.add_argument('--max_num', type=int, default=30,
                    help='how many ranks you want to keep')
parser = parser.parse_args()

mode = parser.mode
max_num = parser.max_num
epoch = parser.epoch

meta_file = 'PlantCLEF2022_test_metadata.csv'
pred_dir = f"results/clef_plant_results_epoch{epoch}"
out_name = f"results/clef_plant_submission_epoch{epoch}_{mode}_top{max_num}.csv"
if osp.exists(out_name):
    os.remove(out_name)

# read the meta data to combine observations
meta = pd.read_csv(meta_file, header=0, sep=';')


history_obs_id = 1
history_score = 0
value = meta.values
content = '1; 2927096; 0.90, 1'
if 'single' in mode:
    for i in range(len(value)):
        obs_id = value[i][0]
        csv_id = value[i][1].replace('.jpg', '.csv')
        csv_id = osp.join(pred_dir, csv_id)
        if int(obs_id) != history_obs_id and i > 0:
            with open(out_name, 'a+') as file:
                file.writelines(content)

            history_obs_id = int(obs_id)
            history_score = pd.read_csv(csv_id, header=0).values[0, 1]
            single_csv = pd.read_csv(csv_id, header=0).values.T
            label = single_csv[0]
            score = single_csv[1]
            rank = single_csv[2]
            content = []
            for num in range(max_num):
                single = str(obs_id) + ';' + str(int(label[num])) + ";" + \
                         str(score[num]) + ";" + str(num + 1) + "\n"
                content.append(single)
        else:
            top1_score = pd.read_csv(csv_id, header=0).values[0, 1]
            if 'high' in mode:
                if top1_score > history_score or i == 0:
                    history_score = top1_score
                    single_csv = pd.read_csv(csv_id, header=0).values.T
                    label = single_csv[0]
                    score = single_csv[1]
                    rank = single_csv[2]
                    content = []
                    for num in range(max_num):
                        single = str(obs_id) + ';' + str(int(label[num])) + ";" + \
                                 str(score[num]) + ";" + str(num + 1) + "\n"
                        content.append(single)
            else:
                assert 'low' in mode
                if top1_score < history_score or i == 0:
                    history_score = top1_score
                    single_csv = pd.read_csv(csv_id, header=0).values.T
                    label = single_csv[0]
                    score = single_csv[1]
                    rank = single_csv[2]
                    content = []
                    for num in range(max_num):
                        single = str(obs_id) + ';' + str(int(label[num])) + ";" + \
                                 str(score[num]) + ";" + str(num + 1) + "\n"
                        content.append(single)

        if i == len(value) - 1:
            with open(out_name, 'a+') as file:
                file.writelines(content)
elif mode == 'sorted':
    rank = np.array(range(1, 31))
    for i in range(len(value)):
        obs_id = value[i][0]
        csv_id = value[i][1].replace('.jpg', '.csv')
        csv_id = osp.join(pred_dir, csv_id)

        if int(obs_id) != history_obs_id and i > 0:
            # look at top-30 scores for the accumulated scores and labels
            # remove the duplicate label first and keep the largest score
            obs_score = np.array(obs_score).flatten()
            max_idx = (-obs_score).argsort()
            sorted_obs_score = obs_score[max_idx]
            sorted_obs_label = np.array(obs_label).flatten()[max_idx]

            unique_obs_label, unique_idx = np.unique(sorted_obs_label, return_index=True)
            unique_obs_score = sorted_obs_score[unique_idx]
            unique_max_idx = (-unique_obs_score).argsort()[:max_num]

            final_score = unique_obs_score[unique_max_idx]
            final_label = unique_obs_label[unique_max_idx]
            content = []
            for num in range(max_num):
                single = str(history_obs_id) + ';' + str(int(final_label[num])) \
                         + ";" + str(final_score[num]) + ";" + str(num + 1) + "\n"
                content.append(single)
            with open(out_name, 'a+') as file:
                file.writelines(content)

            # redefine for new obs_id
            history_obs_id = int(obs_id)
            single_csv = pd.read_csv(csv_id, header=0).values.T
            obs_label = single_csv[0].flatten()
            obs_score = single_csv[1].flatten()
        else:
            # accumulate the scores
            single_csv = pd.read_csv(csv_id, header=0).values.T
            label = single_csv[0].flatten()
            score = single_csv[1].flatten()
            if i == 0:
                obs_label = label
                obs_score = score
            else:
                obs_label = np.concatenate((obs_label, label))
                obs_score = np.concatenate((obs_score, score))

        if i == len(value) - 1:
            # look at top-30 scores for the accumulated scores and labels
            # remove the duplicate label first and keep the largest score
            obs_score = np.array(obs_score).flatten()
            max_idx = (-obs_score).argsort()
            sorted_obs_score = obs_score[max_idx]
            sorted_obs_label = np.array(obs_label).flatten()[max_idx]

            unique_obs_label, unique_idx = np.unique(sorted_obs_label, return_index=True)
            unique_obs_score = sorted_obs_score[unique_idx]
            unique_max_idx = (-unique_obs_score).argsort()[:max_num]

            final_score = unique_obs_score[unique_max_idx]
            final_label = unique_obs_label[unique_max_idx]
            content = []
            for num in range(max_num):
                single = str(history_obs_id) + ';' + str(int(final_label[num])) \
                         + ";" + str(final_score[num]) + ";" + str(num + 1) + "\n"
                content.append(single)
            with open(out_name, 'a+') as file:
                file.writelines(content)

else:
    assert mode == 'random'
    rank = np.array(range(1, 31))
    valid_num_obs = 0
    for i in range(len(value)):
        obs_id = value[i][0]
        csv_id = value[i][1].replace('.jpg', '.csv')
        csv_id = osp.join(pred_dir, csv_id)

        if int(obs_id) != history_obs_id and i > 0:
            # just chose a random image and its prediction and rank for an observation
            # assert valid_num_obs > 0, print(f"please check the situation around {csv_id}")
            # assert valid_num_obs > 0, print(f"Obs_id: {obs_id}")
            # assert valid_num_obs > 0, print(f"History_obs_id: {history_obs_id}")
            # assert valid_num_obs > 0, print(f"i: {i}")
            rand = np.random.randint(valid_num_obs)
            final_score = obs_score.flatten()[rand * 30: rand * 30 + max_num]
            final_label = obs_label.flatten()[rand * 30: rand * 30 + max_num]

            content = []
            for num in range(max_num):
                single = str(history_obs_id) + ';' + str(int(final_label[num])) \
                         + ";" + str(final_score[num]) + ";" + str(num + 1) + "\n"
                content.append(single)
            with open(out_name, 'a+') as file:
                file.writelines(content)

            # redefine for new obs_id
            history_obs_id = int(obs_id)
            single_csv = pd.read_csv(csv_id, header=0).values.T
            obs_label = single_csv[0].flatten()
            obs_score = single_csv[1].flatten()

            valid_num_obs = 1
        else:
            # accumulate the scores
            valid_num_obs += 1
            single_csv = pd.read_csv(csv_id, header=0).values.T
            label = single_csv[0].flatten()
            score = single_csv[1].flatten()
            if i == 0:
                obs_label = label
                obs_score = score
            else:
                obs_label = np.concatenate((obs_label, label))
                obs_score = np.concatenate((obs_score, score))

        if i == len(value) - 1:
            # just chose a random image and its prediction and rank for an observation
            rand = np.random.randint(valid_num_obs)
            final_score = obs_score.flatten()[rand * 30: rand * 30 + max_num]
            final_label = obs_label.flatten()[rand * 30: rand * 30 + max_num]
            content = []
            for num in range(max_num):
                single = str(history_obs_id) + ';' + str(int(final_label[num])) \
                         + ";" + str(final_score[num]) + ";" + str(num + 1) + "\n"
                content.append(single)
            with open(out_name, 'a+') as file:
                file.writelines(content)