"""models evaluation functions"""
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score


def evaluate(model_id, mode='expert'):
    """
    :param model_id: experiment index
    # '0'-judge_zero.log-return accuracy, wa-recall, wa-f1
    # 'CoT'-judge_CoT.log-CoT-return accuracy, wa-recall, wa-f1
    # 'expert'-judge_expert.log-expert model-return accuracy, wa-recall, wa-f1
    # 'expert_CoT'-judge_expert_CoT.log-expert+COT-return accuracy1, wa-recall1, wa-f11, accuracy2, wa-recall2, wa-f12
    :return: acc, recall, f1
    """
    y_true = []
    y_pred1 = []
    y_pred2 = []
    if mode == '0':
        with open(f'dataset/judge_zero_1_{model_id}.log', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data:
                if isinstance(t[0], int) and isinstance(t[1], int):
                    if t[1] > -1:
                        y_true.append(t[0])
                        y_pred1.append(t[1])
        return accuracy_score(y_true, y_pred1), \
               recall_score(y_true, y_pred1, average='weighted'), \
               f1_score(y_true, y_pred1, average='weighted')
    elif mode == 'CoT':
        with open(f'dataset/judge_CoT_1_{model_id}.log', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data:
                if isinstance(t[0], int) and isinstance(t[1], int):
                    if t[1] > -1:
                        y_true.append(t[0])
                        y_pred1.append(t[1])
        print(y_true, '\n', y_pred1)
        return accuracy_score(y_true, y_pred1), \
               recall_score(y_true, y_pred1, average='weighted'), \
               f1_score(y_true, y_pred1, average='weighted')
    elif mode == 'expert':
        with open(f'dataset/judge_expert_1_{model_id}.log', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data:
                if isinstance(t[0], int) and isinstance(t[1], int):
                    if t[1] > -1:
                        y_true.append(t[0])
                        y_pred1.append(t[1])
        return accuracy_score(y_true, y_pred1), \
               recall_score(y_true, y_pred1, average='weighted'), \
               f1_score(y_true, y_pred1, average='weighted')
    elif mode == 'expert_CoT':
        with open(f'dataset/judge_expert_CoT_1_{model_id}.log', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data:
                if isinstance(t[0], int) and isinstance(t[1], int) and isinstance(t[2], int):
                    if t[1] > -1 and t[2] > -1:
                        y_true.append(t[0])
                        y_pred1.append(t[1])
                        y_pred2.append(t[2])
        return accuracy_score(y_true, y_pred1), \
               recall_score(y_true, y_pred1, average='weighted'), \
               f1_score(y_true, y_pred1, average='weighted'), \
               accuracy_score(y_true, y_pred2), \
               recall_score(y_true, y_pred2, average='weighted'), \
               f1_score(y_true, y_pred2, average='weighted')
    elif mode == '0_2':
        with open(f'dataset/judge_zero_2_{model_id}.log', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data:
                if isinstance(t[0], int) and isinstance(t[1], int):
                    if t[1] > -1:
                        y_true.append(t[0])
                        y_pred1.append(t[1])
        return accuracy_score(y_true, y_pred1), \
               recall_score(y_true, y_pred1, average='weighted'), \
               f1_score(y_true, y_pred1, average='weighted')
    elif mode == 'CoT_2':
        with open(f'dataset/judge_CoT_2_{model_id}.log', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data:
                if isinstance(t[0], int) and isinstance(t[1], int):
                    if t[1] > -1:
                        y_true.append(t[0])
                        y_pred1.append(t[1])
        print(y_true, '\n', y_pred1)
        return accuracy_score(y_true, y_pred1), \
               recall_score(y_true, y_pred1, average='weighted'), \
               f1_score(y_true, y_pred1, average='weighted')
    elif mode == 'expert_2':
        with open(f'dataset/judge_expert_2_{model_id}.log', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data:
                if isinstance(t[0], int) and isinstance(t[1], int):
                    if t[1] > -1:
                        y_true.append(t[0])
                        y_pred1.append(t[1])
        return accuracy_score(y_true, y_pred1), \
               recall_score(y_true, y_pred1, average='weighted'), \
               f1_score(y_true, y_pred1, average='weighted')
    elif mode == 'expert_CoT_2':
        with open(f'dataset/judge_expert_CoT_2_{model_id}.log', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for t in data:
                if isinstance(t[0], int) and isinstance(t[1], int) and isinstance(t[2], int):
                    if t[1] > -1 and t[2] > -1:
                        y_true.append(t[0])
                        y_pred1.append(t[1])
                        y_pred2.append(t[2])
        return accuracy_score(y_true, y_pred1), \
               recall_score(y_true, y_pred1, average='weighted'), \
               f1_score(y_true, y_pred1, average='weighted'), \
               accuracy_score(y_true, y_pred2), \
               recall_score(y_true, y_pred2, average='weighted'), \
               f1_score(y_true, y_pred2, average='weighted')


def main():
    # one round
    conf1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in conf1:
        # results = evaluate(i, mode='0')
        # print('0', i, results)
        # print(results[1], ',', results[2], end=',')
        results = evaluate(i, mode='expert')
        print('expert', i, results)
        print(results[1], ',', results[2], end=',')
        results = evaluate(i, mode='expert_CoT')
        print('expert_CoT', i, results)
        print(results[1], ',', results[2], end=',')
        print(results[4], ',', results[5])

    # two rounds
    conf2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in conf2:
        # results = evaluate(i, mode='0_2')
        # print('0_2', i, results)
        # print(results[1], ',', results[2], end=',')
        results = evaluate(i, mode='expert_2')
        print('expert_2', i, results)
        print(results[1], ',', results[2], end=',')
        results = evaluate(i, mode='expert_CoT_2')
        # print('expert_C0T_2', i, results)
        print(results[1], ',', results[2], end=',')
        print(results[4], ',', results[5])


if __name__ == '__main__':
    main()
