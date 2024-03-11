"""the prompts loader and checker of the player agent"""
import re
import json
import numpy as np

from agents.chat_basic_qianfan_chinese import ChatBasicQianfanChinese

OrdinalNumbers = ['一', '二', '三', '四', '五', '六']


# eos list
def get_valid_str(s: str, eos: list[str] = None, new_eos: str = None):
    """
    :param s: string to eb processed
    :param eos: list of defined EOS; if None, no process is conducted
    :param new_eos: new EOS for substitute
    :return: valid string
    """
    if eos is None:
        return s
    s_idx = np.array([len(s)] * len(eos))

    # find the first appearance of each EOS
    for i, value in enumerate(eos):
        temp = s.find(value)
        if temp > 0:
            s_idx[i] = temp
    min_idx = np.min(s_idx)

    return s[:min_idx].rstrip() + new_eos  # rstrip first


# PlayerChatBasic class
class PlayerChatBasicQianfanChinese(ChatBasicQianfanChinese):
    """
    Player agent class
    every action phase has a series of prompts,
    and corresponding pop_fn and check_fn to load the prompts, check response, and extract data
    """

    def __init__(
            self,
            name: str,
            word: str,
            remain_players: list,
            context_sheet: str,
            context=None,
            top_p=0.75,
            temperature=0.9
    ):
        super().__init__(context=context, top_p=top_p, temperature=temperature)
        self.name = name
        self.word = word
        self.context_sheet = open(context_sheet, 'w', encoding='utf-8')
        self.remain_players = remain_players
        self.idx_in_game = remain_players.index(name)
        self.num_of_remain_players = len(remain_players)

    def __del__(self):
        del self.name
        del self.context
        del self.context_sheet
        del self.remain_players
        del self.idx_in_game
        del self.num_of_remain_players

    def refresh_self(self):
        """refresh member variables to keep consistency"""
        self.num_of_remain_players = len(self.remain_players)
        self.idx_in_game = self.remain_players.index(self.name)

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
        self.idx_in_game = self.remain_players.index(self.name)

    def pop_fn_game_start(self, n):
        """pop_fn of the game start phase"""
        prompt = None  # [(prompt, max_new_tokens)]
        with open('agents/templates/game_start_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                prompt = (
                    ps[n][:-1].format(
                        players=json.dumps(self.remain_players),
                        num_of_players=self.num_of_remain_players,
                        num_of_op=self.num_of_remain_players - 1,
                        name=self.name,
                        statement_number=self.idx_in_game + 1,
                        my_word=self.word
                    ),  # remove last \n
                    50  # max new tokens, not used
                )
                del ps
            else:
                return '', 0
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_game_start(self, response, n):
        """
        check_fn of the vote phase
        :return: valid response - (True, data); invalid response - (False, None)
        """
        new_content = None

        # in chat version, responses are directly used.
        if n == 0:
            new_content = get_valid_str(response)

        if new_content is None:
            return False, None

        # update context and extract JSON data
        if n == 0:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None

    def pop_fn_word_description(self, n):
        """
        pop_fn of the word description phase
        """
        prompt = None  # [(prompt, max_new_tokens)]
        with open('agents/templates/word_describe_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                prompt = (
                    ps[n][:-1].format(
                        my_word=self.word
                    ),  # remove last \n
                    250  # max new tokens, not used
                )
            elif n == 1:
                prompt = (
                    ps[n][:-1],
                    250  # max new tokens, not used
                )
            elif n == 2:
                prompt = (
                    ps[n][:-1].format(
                        ordinal_number=self.idx_in_game + 1,  # starts from 1 instead of 0
                        my_word=self.word
                    ),
                    250  # max new tokens, not used
                )
            elif n == 3:
                prompt = (
                    ps[n][:-1],
                    20  # max new tokens, not used
                )
            else:
                return '', 0
            del ps
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_word_description(self, response, n):
        """
        check_fn of the word description phase#
        :return: valid response - (True, data); invalid response - (False, None)
        output - {"description": <str>}}
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
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # \n -> _ for re_search
                if 'description' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['description'], str):
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

    def pop_fn_identity_guess(self, n):
        """
        pop_fn of the identity guess phase
        # guessing identity before word description
        """
        prompt = None  # [(prompt, max_new_tokens)]
        with open('agents/templates/identity_guess_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                prompt = (
                    ps[n][:-1],
                    200  # max new tokens, not used
                )
            else:
                return '', 0
            del ps
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_identity_guess(self, response, n):
        """
        check_fn of the identity guess phase
        :return: valid response - (True, data); invalid response - (False, None)
        """
        # guessing identity before word description
        new_content = None

        # in chat version, responses are directly used.
        if n == 0:
            new_content = get_valid_str(response)

        if new_content is None:
            return False, None

        # update context and extract JSON data
        if n == 0:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None

    def pop_fn_description_analysis(self, n, **kwargs):
        """
        pop_fn of the description analysis phase
        # analyzing last player's word description
        # kwargs - {'target': <player name>, 'description': <target description>}
        """
        prompt = None  # [(prompt, max_new_tokens)]
        with open('agents/templates/description_analysis_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                target = kwargs['target']
                target_ordinal_number = self.remain_players.index(target) + 1   # 从1开始
                description = kwargs['description']
                prompt = (
                    ps[n][:-1].format(
                        my_word=self.word,
                        target=target,
                        target_ordinal_number=target_ordinal_number,
                        target_description=description
                    ),
                    200  # max new tokens, not used
                )
            elif n == 1:
                target = kwargs['target']
                prompt = (
                    ps[n][:-1].format(
                        target=target
                    ),
                    50  # max new tokens, not used
                )
            else:
                return '', 0
            del ps
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_description_analysis(self, response, n):
        """
        check_fn of the description analysis phase
        :return: valid response - (True, data); invalid response - (False, None)
        output - None
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
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, None

    def pop_fn_statement_and_discussion(self, n):
        """
        pop_fn of the statement and discussion phase
        # analyzing other players' statement first
        """
        prompt = None  # [(prompt, max_new_tokens)]
        with open('agents/templates/statement_and_discussion_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                prompt = (
                    ps[n][:-1].format(
                        name=self.name,
                        my_word=self.word
                    ),
                    250  # max new tokens, not used
                )
            elif n == 1:
                remain_players_except_self = self.remain_players[:]
                del remain_players_except_self[self.idx_in_game]
                prompt = (
                    ps[n][:-1].format(
                        remain_players=json.dumps(remain_players_except_self)
                    ),
                    120  # max new tokens, not used
                )
            elif n == 2:
                remain_players_except_self = self.remain_players[:]
                del remain_players_except_self[self.idx_in_game]
                prompt = (
                    ps[n][:-1].format(
                        remain_players=json.dumps(remain_players_except_self)
                    ),
                    120  # max new tokens, not used
                )
            elif n == 3:
                remain_players_except_self = self.remain_players[:]
                del remain_players_except_self[self.idx_in_game]
                prompt = (
                    ps[n][:-1].format(
                        remain_players=json.dumps(remain_players_except_self),
                        my_word=self.word
                    ),
                    250  # max new tokens, not used
                )
            else:
                return '', 0
            del ps
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_statement_and_discussion(self, response, n):
        """
        check_fn of the statement and discussion phase
        :return: valid response - (True, data); invalid response - (False, None)
        # output - tendency:dict, discussion:str
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
            pattern = re.compile(r'\{(.)*\}')
            try:
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # 需要将 \n 变为_
                if 'for' not in data.keys() or 'against' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None
        elif n == 3:
            self.append_context({'role': 'assistant', 'content': new_content[:]})
            return True, new_content

    def pop_fn_vote(self, n):
        """pop_fn of the vote phase"""
        prompt = None  # [(prompt, max_new_tokens)]
        with open('agents/templates/vote_qianfan_chinese.txt', 'r', encoding='utf-8') as f:
            ps = f.read().split('-----*****-----\n')
            if n == 0:
                prompt = (
                    ps[n][:-1],
                    50  # max new tokens, not used
                )
            elif n == 1:
                prompt = (
                    ps[n][:-1],
                    50  # max new tokens, not used
                )
            elif n == 2:
                remain_players_except_self = self.remain_players[:]
                del remain_players_except_self[self.idx_in_game]
                prompt = (
                    ps[n][:-1].format(
                        remain_players=json.dumps(remain_players_except_self)
                    ),
                    50  # max new tokens, not used
                )
            elif n == 3:
                prompt = (
                    ps[n][:-1],
                    50  # max new tokens, not used
                )
            else:
                return '', 0
            del ps
        self.append_context({'role': 'user', 'content': prompt[0][:]})
        return prompt

    def check_fn_vote(self, response, n):
        """
        check_fn of the vote phase
        :return: valid response - (True, data); invalid response - (False, None)
        # output - {"vote": <name>}
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
                data = json.loads(pattern.search(new_content.replace('\n', ' ')).group(0))  # \n -> _ for re_search
                if 'vote' not in data.keys():
                    print(new_content, '\n--------key error--------')
                    return False, None
                elif not isinstance(data['vote'], str):
                    print(new_content, '\n--------value error--------')
                    return False, None
                self.append_context({'role': 'assistant', 'content': new_content[:]})
                return True, data
            except Exception as e:
                print(e)
                print(new_content, '\n--------format error--------')
                return False, None


if __name__ == '__main__':
    players = ['Alice', 'Bob', 'Carol', 'Daniel', 'Edward', 'Frank']
    Alice_test = PlayerChatBasicQianfanChinese('Alice', '鸡', players, 'logs/Alice_test.log')
    Bob_test = PlayerChatBasicQianfanChinese('Bob', '鸡', players, 'logs/Bob_test.log')
    Carol_test = PlayerChatBasicQianfanChinese('Carol', '鸡', players, 'logs/Carol_test.log')
    Daniel_test = PlayerChatBasicQianfanChinese('Daniel', '鸭', players, 'logs/Daniel_test.log')
    Edward_test = PlayerChatBasicQianfanChinese('Edward', '鸡', players, 'logs/Edward_test.log')
    Frank_test = PlayerChatBasicQianfanChinese('Frank', '鸡', players, 'logs/Frank_test.log')
    Alice_test.chat_basic(Alice_test.pop_fn_game_start, Alice_test.check_fn_game_start)
