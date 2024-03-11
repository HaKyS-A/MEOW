"""the prompts loader and checker of the judge agent"""
import re
import json

import numpy as np
from agents.chat_basic_qianfan_chinese import ChatBasicQianfanChinese


# eos list
def get_valid_str(s: str, eos: list[str] = None, new_eos: str=None):
    """
    :param s: string to eb processed
    :param eos: list of defined EOS; if None, no process is conducted
    :param new_eos: new EOS for substitute
    :return: valid string
    """
    if eos is None:
        return s
    s_idx = np.array([len(s)] * len(eos))

    # find the the first appearance of each EOS
    for i, value in enumerate(eos):
        temp = s.find(value)
        if temp > 0:
            s_idx[i] = temp
    min_idx = np.min(s_idx)

    return s[:min_idx].rstrip() + new_eos  # rstrip first


# PlayerChatBasic class
class JudgeChatBasicQianfanChinese(ChatBasicQianfanChinese):
    """
    Judge agent class
    every action phase has a series of prompts,
    and corresponding pop_fn and check_fn to load the prompts, check response, and extract data
    """

    def __init__(
            self,
            hint:str,
            remain_players: list,
            log_file: str,
            context_sheet: str,
            context: dict = None,
            top_p=0.95,
            temperature=0.8
    ):
        # hint - hints about two words of the current game, usually their common points
        super().__init__(context=context, top_p=top_p, temperature=temperature, model='ERNIE-Bot')
        self.hint = hint
        self.context_sheet = open(context_sheet, 'w', encoding='utf-8')
        self.remain_players = remain_players
        self.num_of_remain_players = len(remain_players)
        self.log_file = log_file

    def __del__(self):
        del self.context
        del self.context_sheet
        del self.remain_players
        del self.num_of_remain_players

    def refresh_self(self):
        """refresh member variables to keep consistency"""
        self.num_of_remain_players = len(self.remain_players)

    def append_context(self, context):
        """
        update log file - self.context_sheet
        update self.context for prompting
        :contex: newly generated context
        """
        self.context_sheet.write(context['role'])
        self.context_sheet.write(': ')
        self.context_sheet.write(context['content'])
        self.context_sheet.write('\n')
        self.context.append(context)

    def refresh_remain_players(self, remain_players):
        """refresh self.remain_players to keep consistency"""
        self.remain_players = remain_players

    def pop_fn_judge_0(self, n):
        """pop_fn of judge w/o CoT or Expert"""
        prompt = None  # [(prompt .max_new_tokens)]
        with open('agents/templates/judge_0_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log = json.load(f)

                with open(self.log_file.replace('uttr', 'vote'), 'r', encoding='utf-8') as f:
                    vote = json.load(f)

                # round 1
                ## uttr.json
                log_str = '\n```\n'
                for i in log[0][:8]:
                    log_str += i
                    log_str += '\n'
                log_str += '```\n'
                ## vote.json
                vote_str = '\n```\n'
                for key, value in vote[0].items():
                    vote_str += f'{key}选择投出{value}。'
                    vote_str += '\n'
                vote_str += '```\n'

                prompt = (
                    ps[n][:-1].format(
                        players=json.dumps(self.remain_players),
                        num_of_players=self.num_of_remain_players,
                        num_of_op=self.num_of_remain_players - 1,
                        log_json=log_str,
                        vote=vote_str,
                        hint=self.hint
                    ),  # remove last \n
                    50  # max new tokens, not used
                )
                del ps
            elif n == 1:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            else:
                return '', 0
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_judge_0(self, response, n):
        """
        check_fn of judge w/o CoT or Expert
        :return: valid response - (True, data); invalid response - (False, None)
        """
        new_content = None

        # in chat version, responses are directly used.
        if n == 0:
            new_content = get_valid_str(response)
        elif n == 1:
            new_content = get_valid_str(response)

        if new_content is None:
            return False, None

        # update context and extract JSON data
        if n == 0:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 1:
            # extract JSON data {"卧底": str}
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # remove \n before re.search
                if '卧底' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['卧底'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        else:
            return False, None

    def pop_fn_judge_no_expert(self, n):
        """pop_fn of judge CoT only"""
        prompt = None  # [(prompt .max_new_tokens)]
        with open('agents/templates/judge_no_expert_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log = json.load(f)

                with open(self.log_file.replace('uttr', 'vote'), 'r', encoding='utf-8') as f:
                    vote = json.load(f)

                # round 1
                ## uttr.json
                log_str = '\n```\n'
                for i in log[0][:8]:
                    log_str += i
                    log_str += '\n'
                log_str += '```\n'
                ## vote.json
                vote_str = '\n```\n'
                for key, value in vote[0].items():
                    vote_str += f'{key}选择投出{value}。'
                    vote_str += '\n'
                vote_str += '```\n'

                prompt = (
                    ps[n][:-1].format(
                        players=json.dumps(self.remain_players),
                        num_of_players=self.num_of_remain_players,
                        num_of_op=self.num_of_remain_players - 1,
                        log_json=log_str,
                        vote=vote_str,
                        hint=self.hint
                    ),  # remove last \n
                    50  # max new tokens, not used
                )
                del ps
            elif n == 1:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 2:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 3:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            else:
                return '', 0
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_judge_no_expert(self, response, n):
        """
        check_fn of judge CoT only
        :return: valid response - (True, data); invalid response - (False, None)
        """
        new_content = None

        # in chat version, responses are directly used.
        if n == 0:
            new_content = get_valid_str(response)
        elif n == 1:
            new_content = get_valid_str(response)
        elif n == 2:
            new_content = get_valid_str(response)
        elif n == 3:
            new_content = get_valid_str(response)

        if new_content is None:
            return False, None

        # update context and extract JSON data
        if n == 0:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 1:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 2:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 3:
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # remove \n before re.search
                if '卧底' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['卧底'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        else:
            return False, None

    def pop_fn_judge_with_expert(self, n, **kwargs):
        """
        pop_fn of judge with CoT and Expert observation
        :param kwargs: kwargs - agent=name:str
        """
        prompt = None  # [(prompt .max_new_tokens)]
        with open('agents/templates/judge_with_expert_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log = json.load(f)

                with open(self.log_file.replace('uttr', 'vote'), 'r', encoding='utf-8') as f:
                    vote = json.load(f)

                # round 1
                ## uttr.json
                log_str = '\n```\n'
                for i in log[0][:8]:
                    log_str += i
                    log_str += '\n'
                log_str += '```\n'
                ## vote.json
                vote_str = '\n```\n'
                for key, value in vote[0].items():
                    vote_str += f'{key}选择投出{value}。'
                    vote_str += '\n'
                vote_str += '```\n'

                prompt = (
                    ps[n][:-1].format(
                        players=json.dumps(self.remain_players),
                        num_of_players=self.num_of_remain_players,
                        num_of_op=self.num_of_remain_players - 1,
                        log_json=log_str,
                        vote=vote_str,
                        hint=self.hint
                    ),  # remove last \n
                    50  # max new tokens, not used
                )
                del ps
            elif n == 1:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 2:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 3:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 4:
                prompt = (
                    ps[n][:-1].format(
                        agent=kwargs['agent']
                    ),
                    250  # max new tokens, not used
                )
            elif n == 5:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            else:
                return '', 0
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_judge_with_expert(self, response, n):
        """
        check_fn of judge with CoT and Expert observation
        :return: valid response - (True, data); invalid response - (False, None)
        """
        new_content = None

        # in chat version, responses are directly used.
        if n == 0:
            new_content = get_valid_str(response)
        elif n == 1:
            new_content = get_valid_str(response)
        elif n == 2:
            new_content = get_valid_str(response)
        elif n == 3:
            new_content = get_valid_str(response)
        elif n == 4:
            new_content = get_valid_str(response)
        elif n == 5:
            new_content = get_valid_str(response)


        if new_content is None:
            return False, None

        # update context and extract JSON data
        if n == 0:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 1:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 2:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 3:
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # remove \n before re.search
                if '卧底' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['卧底'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        elif n == 4:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 5:
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # remove \n before re.search
                if '卧底' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['卧底'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        else:
            return False, None

    def pop_fn_judge_0_2(self, n):
        """pop_fn of 2-rounds judge w/o CoT or Expert"""
        prompt = None  # [(prompt .max_new_tokens)]
        with open('agents/templates/judge_0_2_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log = json.load(f)

                with open(self.log_file.replace('uttr', 'vote'), 'r', encoding='utf-8') as f:
                    vote = json.load(f)

                # round 1
                ## uttr.json
                log_str = '\n```\n'
                for i in log[0][:9]:
                    log_str += i
                    log_str += '\n'
                log_str += '```\n'
                ## vote.json
                vote_str = '\n```\n'
                for key, value in vote[0].items():
                    vote_str += f'{key}选择投出{value}。'
                    vote_str += '\n'
                vote_str += '```\n'
                # round 2
                ## uttr.json
                log_str_2 = '\n```\n'
                for i in log[1][:6]:
                    log_str += i
                    log_str += '\n'
                log_str_2 += '```\n'
                ## vote.json
                vote_str_2 = '\n```\n'
                for key, value in vote[1].items():
                    vote_str += f'{key}选择投出{value}。'
                    vote_str += '\n'
                vote_str_2 += '```\n'

                prompt = (
                    ps[n][:-1].format(
                        players=json.dumps(self.remain_players),
                        num_of_players=self.num_of_remain_players,
                        num_of_op=self.num_of_remain_players - 1,
                        log_json=log_str,
                        vote=vote_str,
                        log_json_2=log_str_2,
                        vote_2=vote_str_2,
                        hint=self.hint
                    ),  # remove last \n
                    50  # max new tokens, not used
                )
                del ps
            elif n == 1:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            else:
                return '', 0
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_judge_0_2(self, response, n):
        """
        check_fn of 2-rounds judge w/o CoT or Expert observation
        :return: valid response - (True, data); invalid response - (False, None)
        """
        new_content = None

        # in chat version, responses are directly used.
        if n == 0:
            new_content = get_valid_str(response)
        elif n == 1:
            new_content = get_valid_str(response)

        if new_content is None:
            return False, None

        # update context and extract JSON data
        if n == 0:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 1:
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # remove \n before re.search
                if '卧底' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['卧底'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        else:
            return False, None

    def pop_fn_judge_no_expert_2(self, n):
        """pop]_fn of 2-rounds judge with CoT only"""
        prompt = None  # [(prompt .max_new_tokens)]
        with open('agents/templates/judge_no_expert_2_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log = json.load(f)

                with open(self.log_file.replace('uttr', 'vote'), 'r', encoding='utf-8') as f:
                    vote = json.load(f)

                    # round 1
                ## uttr.json
                log_str = '\n```\n'
                for i in log[0][:9]:
                    log_str += i
                    log_str += '\n'
                log_str += '```\n'
                ## vote.json
                vote_str = '\n```\n'
                for key, value in vote[0].items():
                    vote_str += f'{key}选择投出{value}。'
                    vote_str += '\n'
                vote_str += '```\n'
                # round 2
                ## uttr.json
                log_str_2 = '\n```\n'
                for i in log[1][:6]:
                    log_str_2 += i
                    log_str_2 += '\n'
                log_str_2 += '```\n'
                ## vote.json
                vote_str_2 = '\n```\n'
                for key, value in vote[1].items():
                    vote_str_2 += f'{key}选择投出{value}。'
                    vote_str_2 += '\n'
                vote_str_2 += '```\n'

                prompt = (
                    ps[n][:-1].format(
                        players=json.dumps(self.remain_players),
                        num_of_players=self.num_of_remain_players,
                        num_of_op=self.num_of_remain_players - 1,
                        log_json=log_str,
                        vote=vote_str,
                        log_json_2=log_str_2,
                        vote_2=vote_str_2,
                        hint=self.hint
                    ),  # remove last \n
                    50  # max new tokens, not used
                )
                del ps
            elif n == 1:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 2:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 3:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            else:
                return '', 0
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_judge_no_expert_2(self, response, n):
        """
        check_fn of 2-rounds judge with CoT only
        :return: valid response - (True, data); invalid response - (False, None)
        """
        new_content = None

        # in chat version, responses are directly used.
        if n == 0:
            new_content = get_valid_str(response)
        elif n == 1:
            new_content = get_valid_str(response)
        elif n == 2:
            new_content = get_valid_str(response)
        elif n == 3:
            new_content = get_valid_str(response)

        if new_content is None:
            return False, None

        # update context and extract JSON data
        if n == 0:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 1:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 2:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 3:
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # remove \n before re.search
                if '卧底' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['卧底'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        else:
            return False, None

    def pop_fn_judge_with_expert_2(self, n, **kwargs):
        """pop_fn of 2-rounds judge with CoT and Expert observation"""
        # kwargs - agent=name:str
        prompt = None  # [(prompt .max_new_tokens)]
        with open('agents/templates/judge_with_expert_2_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log = json.load(f)

                with open(self.log_file.replace('uttr', 'vote'), 'r', encoding='utf-8') as f:
                    vote = json.load(f)

                    # round 1
                ## uttr.json
                log_str = '\n```\n'
                for i in log[0][:9]:
                    log_str += i
                    log_str += '\n'
                log_str += '```\n'
                ## vote.json
                vote_str = '\n```\n'
                for key, value in vote[0].items():
                    vote_str += f'{key}选择投出{value}。'
                    vote_str += '\n'
                vote_str += '```\n'
                # round 2
                ## uttr.json
                log_str_2 = '\n```\n'
                for i in log[1][:6]:
                    log_str_2 += i
                    log_str_2 += '\n'
                log_str_2 += '```\n'
                ## vote.json
                vote_str_2 = '\n```\n'
                for key, value in vote[1].items():
                    vote_str_2 += f'{key}选择投出{value}。'
                    vote_str_2 += '\n'
                vote_str_2 += '```\n'

                prompt = (
                    ps[n][:-1].format(
                        players=json.dumps(self.remain_players),
                        num_of_players=self.num_of_remain_players,
                        num_of_op=self.num_of_remain_players - 1,
                        log_json=log_str,
                        vote=vote_str,
                        log_json_2=log_str_2,
                        vote_2=vote_str_2,
                        hint=self.hint
                    ),  # remove last \n
                    50  # max new tokens, not used
                )
                del ps
            elif n == 1:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 2:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 3:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 4:
                prompt = (
                    ps[n][:-1].format(
                        agent=kwargs['agent']
                    ),
                    250  # max new tokens, not used
                )
            elif n == 5:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            else:
                return '', 0
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_judge_with_expert_2(self, response, n):
        """
        check_fn of 2-rounds judge with CoT and Expert observation
        :return: valid response - (True, data); invalid response - (False, None)
        """
        new_content = None

        # in chat version, responses are directly used.
        if n == 0:
            new_content = get_valid_str(response)
        elif n == 1:
            new_content = get_valid_str(response)
        elif n == 2:
            new_content = get_valid_str(response)
        elif n == 3:
            new_content = get_valid_str(response)
        elif n == 4:
            new_content = get_valid_str(response)
        elif n == 5:
            new_content = get_valid_str(response)

        if new_content is None:
            return False, None

        # update context and extract JSON data
        if n == 0:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 1:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 2:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 3:
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # remove \n before re.search
                if '卧底' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['卧底'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        elif n == 4:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None
        elif n == 5:
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # remove \n before re.search
                if '卧底' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['卧底'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        else:
            return False, None
