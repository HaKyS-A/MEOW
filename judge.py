"""run judge agent"""
import json
from agents.judge_chat_basic_qianfan_chinese import JudgeChatBasicQianfanChinese
from GetExpert import call_expert


def judge_0(conf, log_file='dataset/judge_zero.log'):
    """one round judge by judge agent with one prompt"""
    results = []
    for c in conf:
        players = c['players']
        hint = c['hints']
        agent = c['agent']
        context_sheet = 'dataset/{:s}'.format(c['logFile'])
        result_file = 'dataset/{:s}'.format(c['resultFile'])

        # call judge agent
        test = JudgeChatBasicQianfanChinese(hint, players, result_file, context_sheet, top_p=0.85, temperature=0.9)
        output = test.chat_basic(test.pop_fn_judge_0, test.check_fn_judge_0)
        print(output)

        try:
            i = players.index(output[0]['卧底'])  # i - index of doubted player
            results.append((agent, i))
        except Exception as e:
            print(e)
            results.append((agent, output))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def judge_0_2(conf, log_file='dataset/judge_zero_2.log'):
    """two rounds judge by judge agent with one prompt"""
    results = []
    for c in conf:
        players = c['players']
        hint = c['hints']
        agent = c['agent']
        context_sheet = 'dataset/{:s}'.format(c['logFile'])
        result_file = 'dataset/{:s}'.format(c['resultFile'])

        # call judge agent
        test = JudgeChatBasicQianfanChinese(hint, players, result_file, context_sheet, top_p=0.85, temperature=0.9)
        output = test.chat_basic(test.pop_fn_judge_0_2, test.check_fn_judge_0_2)
        print(output)

        try:
            i = players.index(output[0]['卧底'])  # i - index of doubted player
            results.append((agent, i))
        except Exception as e:
            print(e)
            results.append((agent, output))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def judge_CoT(conf, log_file='dataset/judge_CoT.log'):
    """one round judge by judge agent with CoT prompts"""
    results = []
    for c in conf:
        players = c['players']
        hint = c['hints']
        agent = c['agent']
        context_sheet = 'dataset/{:s}'.format(c['logFile'])
        result_file = 'dataset/{:s}'.format(c['resultFile'])

        # call judge agent
        test = JudgeChatBasicQianfanChinese(hint, players, result_file, context_sheet, top_p=0.85, temperature=0.9)
        output = test.chat_basic(test.pop_fn_judge_no_expert, test.check_fn_judge_no_expert)
        print(output)

        try:
            i = players.index(output[0]['卧底'])  # i - index of doubted player
            results.append((agent, i))
        except Exception as e:
            print(e)
            results.append((agent, output))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def judge_CoT_2(conf, log_file='dataset/judge_CoT_2.log'):
    """one round judge by judge agent with CoT prompts"""
    results = []
    for c in conf:
        players = c['players']
        hint = c['hints']
        agent = c['agent']
        context_sheet = 'dataset/{:s}'.format(c['logFile'])
        result_file = 'dataset/{:s}'.format(c['resultFile'])

        # call judge agent
        test = JudgeChatBasicQianfanChinese(hint, players, result_file, context_sheet, top_p=0.85, temperature=0.9)
        output = test.chat_basic(test.pop_fn_judge_no_expert_2, test.check_fn_judge_no_expert_2)
        print(output)

        try:
            i = players.index(output[0]['卧底'])  # i - index of doubted player
            results.append((agent, i))
        except Exception as e:
            print(e)
            results.append((agent, output))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def judge_expert(conf, model_id, log_file='dataset/judge_expert.log'):
    """
    one round judge by expert model
    # model_id refers to cross-validation id 1-10
    """
    results = []
    for c in conf:
        players = c['players']
        agent = c['agent']

        # load interaction graph
        result_file = 'dataset/{:s}'.format(c['resultFile'])
        tendency_file = result_file.replace('uttr', 'tendency')
        vote_file = result_file.replace('uttr', 'vote')
        with open(tendency_file, 'r', encoding='utf-8') as f:
            tendency = json.load(f)
        with open(vote_file, 'r', encoding='utf-8') as f:
            vote = json.load(f)

        # call expert model
        output = call_expert(1, tendency, vote, model_id)  # return player name
        print(output)

        try:
            i = players.index(output)
            results.append((agent, i))
            with open(result_file.replace('uttr', 'expert'), 'w', encoding='utf-8') as f:
                json.dump([{'agent': output}], f)    # [{round1}] round-{"agents": <name>}
        except Exception as e:
            print(e)
            results.append((agent, output))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def judge_expert_2(conf, model_id, log_file='dataset/judge_expert_2.log'):
    """
    two rounds judge by expert model
    # model_id refers to cross-validation id 1-10
    """
    results = []
    for c in conf:
        players = c['players']
        agent = c['agent']

        # load interaction graph
        result_file = 'dataset/{:s}'.format(c['resultFile'])
        tendency_file = result_file.replace('uttr', 'tendency')
        vote_file = result_file.replace('uttr', 'vote')
        with open(tendency_file, 'r', encoding='utf-8') as f:
            tendency = json.load(f)
        with open(vote_file, 'r', encoding='utf-8') as f:
            vote = json.load(f)

        # call expert model
        output = call_expert(2, tendency, vote, model_id)  # return player name
        print(output)
        try:
            i = players.index(output)
            results.append((agent, i))
            with open(result_file.replace('uttr', 'expert'), 'w', encoding='utf-8') as f:
                json.dump([{'agent': output}], f)    # [{round2}] round-{"agents": <name>}
        except Exception as e:
            print(e)
            results.append((agent, output))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def judge_expert_CoT(conf, log_file='dataset/judge_expert_CoT.log'):
    """
    one round judge by judge agent with CoT prompts and expert observation
    # model_id refers to cross-validation id 1-10
    """
    results = []
    for c in conf:
        players = c['players']
        hint = c['hints']
        agent = c['agent']
        context_sheet = 'dataset/{:s}'.format(c['logFile'])
        result_file = 'dataset/{:s}'.format(c['resultFile'])
        expert_file = result_file.replace('uttr', 'expert')

        # call judge agent
        test = JudgeChatBasicQianfanChinese(hint, players, result_file, context_sheet, top_p=0.85, temperature=0.9)
        with open(expert_file, 'r', encoding='utf-8') as f:
            expert_output_agent = json.load(f)[0]['agent']
        output = test.chat_basic(test.pop_fn_judge_with_expert, test.check_fn_judge_with_expert,
                                 agent=expert_output_agent)
        print(output)  # output - [{'卧底': CoT}, {'卧底': CoT+EO}]

        try:
            i, j = players.index(output[0]['卧底']), players.index(output[1]['卧底'])
            # i, j - index of doubted player of Cot, CoT+EO
            results.append((agent, i, j))
        except Exception as e:
            print(e)
            results.append((agent, output))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def judge_expert_CoT_2(conf, log_file='dataset/judge_expert_CoT_2.log'):
    """
    two rounds judge by judge agent with CoT prompts and expert observation
    # model_id refers to cross-validation id 1-10
    """
    results = []
    for c in conf:
        players = c['players']
        hint = c['hints']
        agent = c['agent']
        context_sheet = 'dataset/{:s}'.format(c['logFile'])
        result_file = 'dataset/{:s}'.format(c['resultFile'])
        expert_file = result_file.replace('uttr', 'expert')

        # call judge agent
        test = JudgeChatBasicQianfanChinese(hint, players, result_file, context_sheet, top_p=0.85, temperature=0.9)
        with open(expert_file, 'r', encoding='utf-8') as f:
            expert_output_agent = json.load(f)[0]['agent']
        output = test.chat_basic(
            test.pop_fn_judge_with_expert_2,
            test.check_fn_judge_with_expert_2,
            agent=expert_output_agent
        )
        print(output)  # output - [{'卧底': CoT}, {'卧底': CoT+EO}]
        try:
            i, j = players.index(output[0]['卧底']), players.index(output[1]['卧底'])
            # i, j - index of doubted player of Cot, CoT+EO
            results.append((agent, i, j))
        except Exception as e:
            print(e)
            results.append((agent, output))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main():
    conf1 = [1, 5, 6, 7, 8]
    for i in conf1:
        with open('judgeConfig-1-{:d}.json'.format(i), 'r', encoding='utf-8') as f:
            conf = json.load(f)
        judge_0(conf, log_file='dataset/judge_zero_1_{:d}.log'.format(i))
        # judge_CoT(conf)
        judge_expert(conf, i, log_file='dataset/judge_expert_1_{:d}.log'.format(i))
        judge_expert_CoT(conf, log_file='dataset/judge_expert_CoT_1_{:d}.log'.format(i))
    conf2 = [1, 3, 4, 5, 6]
    for i in conf2:
        with open('judgeConfig-2-{:d}.json'.format(i), 'r', encoding='utf-8') as f:
            conf = json.load(f)
        judge_0_2(conf, log_file='dataset/judge_zero_2_{:d}.log'.format(i))
        # judge_CoT_2(conf)
        judge_expert_2(conf, i, log_file='dataset/judge_expert_2_{:d}.log'.format(i))
        judge_expert_CoT_2(conf, log_file='dataset/judge_expert_CoT_2_{:d}.log'.format(i))


if __name__ == '__main__':
    main()
