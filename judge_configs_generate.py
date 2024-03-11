"""Generate judge configs `judgeConfig-i-j.json`"""
import os
import json
import copy

# target path
folder = 'dataset/'

with open('judgeConfig_v2.json', 'r', encoding='utf-8') as f:
    origin = json.load(f)

for i in range(1, 11):
    temp = []
    temp0 = []
    for filename in os.listdir(folder+'OneRound-{:d}/Test_set'.format(i)):
        # 拼接完整的文件路径
        if filename[:7] == 'players':
            idx = int(filename[8:].replace('.json', ''))
            print(idx)
            temp1 = copy.deepcopy(origin[idx])
            temp1["logFile"] = 'OneRound-{:d}/Test_set/'.format(i)+temp1["logFile"]
            temp1["resultFile"] = 'OneRound-{:d}/Test_set/'.format(i)+temp1["resultFile"]
            temp.append(temp1)
        with open('judgeConfig-1-{:d}.json'.format(i), 'w', encoding='utf-8') as f:
            json.dump(temp, f, ensure_ascii=False, indent=4)
    for filename in os.listdir(folder+'TwoRound-{:d}/Test_set'.format(i)):
        # 拼接完整的文件路径
        if filename[:7] == 'players':
            idx = int(filename[8:].replace('.json', ''))
            print(idx)
            temp1 = copy.deepcopy(origin[idx])
            temp1["logFile"] = 'TwoRound-{:d}/Test_set/'.format(i)+temp1["logFile"]
            temp1["resultFile"] = 'TwoRound-{:d}/Test_set/'.format(i)+temp1["resultFile"]
            temp0.append(temp1)
        with open('judgeConfig-2-{:d}.json'.format(i), 'w', encoding='utf-8') as f:
            json.dump(temp0, f, ensure_ascii=False, indent=4)
