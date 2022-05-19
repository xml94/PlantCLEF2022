def save_csv(obs_id, label, score, max_num, out_name):
    content = []
    for num in range(max_num):
        single = str(obs_id) + ';' + str(int(label[num])) \
                 + ";" + str(score[num]) + ";" + str(num + 1) + "\n"
        content.append(single)
    with open(out_name, 'a+') as file:
        file.writelines(content)