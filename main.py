"""main function of simulator"""
import json
import sys
import os

from agents.players import PlayersQianfanChinese
from agents import global_variables
from engine_qianfan_chinese import Engine


def main():
    print(sys.argv)
    if not os.path.exists('./agents/logs'):
        os.mkdir('./agents/logs')
    if not os.path.exists('./model/bestmodel'):
        os.mkdir('./model/bestmodel')
    top_p = float(sys.argv[1])
    temperature = float(sys.argv[2])
    with open('gameConfig.json', 'r', encoding='utf-8') as f:
        players_sheets = json.load(f)
    error_list = []
    for i in range(len(players_sheets)):
        players_sheet = players_sheets[i]
        suffix = f'_{i}.log'
        for j in [1, 2, 3, 4]:
            players_sheet[j][2] += suffix
        players_test = [
            PlayersQianfanChinese(players_sheet[1][0], players_sheet[1][1], players_sheet[0], players_sheet[1][2], players_sheet[1][3], top_p=top_p, temperature=temperature),
            PlayersQianfanChinese(players_sheet[2][0], players_sheet[2][1], players_sheet[0], players_sheet[2][2], players_sheet[2][3], top_p=top_p, temperature=temperature),
            PlayersQianfanChinese(players_sheet[3][0], players_sheet[3][1], players_sheet[0], players_sheet[3][2], players_sheet[3][3], top_p=top_p, temperature=temperature),
            PlayersQianfanChinese(players_sheet[4][0], players_sheet[4][1], players_sheet[0], players_sheet[4][2], players_sheet[4][3], top_p=top_p, temperature=temperature)
        ]
        print('--------------\n', i, str(players_sheet), '--------------\n')
        with open(f'agents/logs/players_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(players_sheet, f, indent=4, ensure_ascii=False)
        engine_test = Engine(players_test)
        engine_test.game_start()
        GG = False
        while not GG:
            GG = engine_test.run_round()
            if isinstance(GG, int) and GG == -1:
                error_list.append(i)
                break
            print('Current Usage:', global_variables.Usage)
            with open(f'agents/logs/uttr_{i}.json', 'w', encoding='utf-8') as f:
                json.dump(engine_test.utterance_record, f, indent=4, ensure_ascii=False)
            with open(f'agents/logs/tendency_{i}.json', 'w', encoding='utf-8') as f:
                json.dump(engine_test.tendency_data, f, indent=4, ensure_ascii=False)
            with open(f'agents/logs/vote_{i}.json', 'w', encoding='utf-8') as f:
                json.dump(engine_test.vote_data, f, indent=4, ensure_ascii=False)
    print(error_list)
    return 0


if __name__ == '__main__':
    main()
