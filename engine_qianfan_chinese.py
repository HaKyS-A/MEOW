"""simulator class and methods"""
import json
import copy
import pandas as pd

from agents.players import PlayersQianfanChinese
from agents import global_variables


class Engine:
    """
    Game Engine class
    connect game phases, and collect, transport, and store game data and transactions
    record and skip errors in simulation
    """

    def __init__(self, players):
        if not isinstance(players[0], PlayersQianfanChinese):
            raise TypeError("Players must be PlayersQianfanChinese type!")
        self.utterance_record = []  # list(list(player, utterance)) idx=round_number-1
        self.players = players
        self.remain_players = players.copy()
        self.current_round = 0
        self.vote_data = []  # list(dict(player, vote)) idx=round_number-1
        self.tendency_data = []  # list(dict(player, tendency)) idx=round_number-1

    def broadcast(self, receivers: list, message: str, broadcaster=None):
        """
        broadcast open message to receivers for certain phases and information sync
        # receivers - list(Players)
        """
        if broadcaster:
            # broadcaster don't receive the message
            for receiver in receivers:
                if receiver.name != broadcaster.name:
                    receiver.chat.append_context({'role': 'user', 'content': message})
                    receiver.chat.append_context({'role': 'assistant', 'content': '好的，我明白了。'})
        else:
            for receiver in receivers:
                receiver.chat.append_context({'role': 'user', 'content': message})
                receiver.chat.append_context({'role': 'assistant', 'content': '好的，我明白了。'})

    def clear_context(self):
        """clear useless context of every player after every round"""
        for player in self.remain_players:
            length = len(player.chat.context)
            for i in range(length):
                temp = player.chat.context[length-1-i]
                if temp['role'] == 'user':
                    d = False   # whether to delete
                    if temp['content'][:7] == '让我们回顾一下':
                        # statement & discussion [0]
                        # vote [1]
                        d = True
                    elif temp['content'][:6] == '在这一轮，第':
                        # description analysis [0]
                        d = True
                    elif temp['content'][:7] == '根据当前信息，':
                        # description analysis [1]
                        d = True
                    elif temp['content'][:7] == '描述环节，请你':
                        # word description [0]
                        d = True
                    elif temp['content'][:8] == '要赢得这场游戏，':
                        # word description [1]
                        d = True
                    elif temp['content'][:6] == '现在，你是第':
                        # word description [2]
                        d = True
                    if d:
                        del player.chat.context[length-i]
                        del player.chat.context[length-1-i]

    def game_start(self):
        """game start process"""
        self.current_round = 1
        for player in self.players:
            output = player.chat.chat_basic(player.chat.pop_fn_game_start, player.chat.check_fn_game_start)
            if isinstance(output, int) and output == -1:
                # error occurred in this process
                return -1
        return 0

    def run_round(self):
        """processes of each round"""
        uttr = []
        vote = {}
        tendency = {}
        self.broadcast(
            self.players,
            f'(第{self.current_round}轮的词语描述阶段开始。)'
        )
        state = self.player_description_session(uttr)
        if state == -1:
            # error occurred in this process
            return -1
        self.broadcast(
            self.players,
            f'(第{self.current_round}轮的发言讨论阶段开始。)'
        )
        state = self.player_discussion_session(uttr, tendency)
        if state == -1:
            # error occurred in this process
            return -1
        self.broadcast(
            self.players,
            f'(第{self.current_round}轮的投票淘汰阶段开始。)'
        )
        state = self.player_vote_session(uttr, vote)
        if state == -1:
            # error occurred in this process
            return -1

        # update data and logs
        self.utterance_record.append(uttr)
        self.tendency_data.append(tendency)
        self.vote_data.append(vote)
        self.current_round += 1
        self.clear_context()
        return state

    def player_description_session(self, uttr: list):
        """
        player description session
        :param uttr: uttr record list
        :return: -1 - error: 0 - success
        """
        for player in self.remain_players:
            if len(uttr) != 0 or len(self.utterance_record) != 0:
                # not round 1 player 1. guess identities before making description
                output0 = player.chat.chat_basic(player.chat.pop_fn_identity_guess, player.chat.check_fn_identity_guess)
                if isinstance(output0, int) and output0 == -1:
                    # error occurred in this process
                    return -1

            # confirm strategy and make description
            output = player.chat.chat_basic(player.chat.pop_fn_word_description, player.chat.check_fn_word_description)
            if isinstance(output, int) and output == -1:
                # error occurred in this process
                return -1
            description = output[0]['description']
            kwargs = {'target': player.name, 'description': description}
            mes = f'{player.name} 在这一轮对其收到的词语的描述是“{description}”'
            uttr.append(mes)

            # broadcast description to other players for description analysis
            self.broadcast(self.remain_players, mes, broadcaster=player)
            for player0 in self.remain_players:
                if player0.name != player.name:
                    # description analysis
                    output1 = player0.chat.chat_basic(
                        player0.chat.pop_fn_description_analysis,
                        player0.chat.check_fn_description_analysis,
                        **kwargs
                    )
                    if isinstance(output1, int) and output1 == -1:
                        # error occurred in this process
                        return -1
        return 0

    def player_discussion_session(self, uttr, tendency):
        """discussion session"""
        for player in self.remain_players:
            output = player.chat.chat_basic(
                player.chat.pop_fn_statement_and_discussion,
                player.chat.check_fn_statement_and_discussion
            )
            if isinstance(output, int) and output == -1:
                # error occurred in this process
                return -1

            mes = f'{player.name} 在这一轮的发言是“{output[1]}”'
            uttr.append(mes)
            tendency[player.name] = copy.deepcopy(output[0])
            # broadcast utterance in discussion to other players
            self.broadcast(self.remain_players, mes, broadcaster=player)
        return 0

    def player_vote_session(self, uttr: list, vote: list):
        """
        vote session.
        calculate votes and check whether game is over after vote.
        # players vote privately in turn
        :param uttr: utterance record list
        :param vote: vote record list {"player name": <vote>}
        :return: 0-game continues 1-the folk wins 2-the spies wins
        """

        vote_box = []
        mes_box = []  # vote -> message to be broadcast after checking

        for player in self.remain_players:
            output = player.chat.chat_basic(
                player.chat.pop_fn_vote,
                player.chat.check_fn_vote
            )   # output - [vote:{'vote': <player name>}]
            if isinstance(output, int) and output == -1:
                # error occurred in this process
                return -1
            vote = output[0]['vote']
            mes = f'({player.name} 选择投出 {vote}。)\n'
            mes_box.append(mes)
            vote[player.name] = vote
            vote_box.append(vote)

        # calculate votes.
        # If several players get the same number of votes, who received his first vote earlier will be eliminated.
        # This is for the balance of players disadvantage of describing earlier.
        mes0 = ''
        for m in mes_box:
            mes0 += m
        vote_result = str(pd.Series(vote_box).value_counts().idxmax())
        if vote_result not in [i.name for i in self.remain_players]:
            print("--------votes error--------")
            return -1
        mes0 += f'(玩家 {vote_result} 被淘汰)'
        uttr.append(mes0)
        self.broadcast(self.remain_players, mes0)

        # broadcast vote results and check game results
        ## update remained players
        is_UA, new_mes = self.remain_player_update(vote_result)
        uttr.append(new_mes)
        self.broadcast(self.remain_players, new_mes)
        if is_UA:
            return 1
        elif len(self.remain_players) == 2:
            print('剩余玩家数量为2，卧底获胜。')
            return 2
        else:
            return 0

    def remain_player_update(self, eliminated: str):
        """
        update remained plyers
        :param eliminated: player to be eliminated in the game
        :return: whether the eliminated is the undercover agent-<True or False>,  str
        """
        mes = ''
        length = len(self.remain_players)
        for i in range(length):
            if self.remain_players[length-1-i].name == eliminated:
                self.remain_players[length-1-i].alive = 0
                if self.remain_players[length-1-i].identity == 1:
                    # the undercover agent is eliminated
                    mes += '(他的身份是卧底，游戏结束。)\n'
                    return True, mes
                else:
                    mes += '(他的身份是平民，游戏继续。)\n'
                del self.remain_players[length-1-i]
                break
        else:
            raise ValueError(f'{eliminated} - player not found')
        names = [x.name for x in self.remain_players]
        for player in self.remain_players:
            player.chat.remain_players.clear()
            player.chat.remain_players.extend(names)
            player.chat.refresh_self()
            print(player.name, player.chat.remain_players)
        mes += f'(系统)(剩余的玩家是{names}。)现在{eliminated}已经被淘汰，在接下来的发言和思考中，请忽略他的存在。\n'
        return False, mes


if __name__ == '__main__':
    playersSheet = [
        ['Alice', 'Bob', 'Carol', 'Daniel'],
        ['Alice', '绿茶', 'agents/logs/Alice_test_qianfan_chinese.log', 0],
        ['Bob', '红茶', 'agents/logs/Bob_test_qianfan_chinese.log', 1],
        ['Carol', '绿茶', 'agents/logs/Carol_test_qianfan_chinese.log', 0],
        ['Daniel', '绿茶', 'agents/logs/Daniel_test_qianfan_chinese.log', 0]
    ]
    Players_test = [
        PlayersQianfanChinese(playersSheet[1][0], playersSheet[1][1], playersSheet[0], playersSheet[1][2], playersSheet[1][3]),
        PlayersQianfanChinese(playersSheet[2][0], playersSheet[2][1], playersSheet[0], playersSheet[2][2], playersSheet[2][3]),
        PlayersQianfanChinese(playersSheet[3][0], playersSheet[3][1], playersSheet[0], playersSheet[3][2], playersSheet[3][3]),
        PlayersQianfanChinese(playersSheet[4][0], playersSheet[4][1], playersSheet[0], playersSheet[4][2], playersSheet[4][3])
    ]
    with open('agents/logs/players.json', 'w', encoding='utf-8') as f:
        json.dump(playersSheet, f, indent=4, ensure_ascii=False)
    Engine_test = Engine(Players_test)
    Engine_test.game_start()
    GG = False
    while not GG:
        GG = Engine_test.run_round()
        print('Current Usage:', global_variables.Usage)
        with open('agents/logs/uttr.json', 'w', encoding='utf-8') as f:
            json.dump(Engine_test.utterance_record, f, indent=4, ensure_ascii=False)
        with open('agents/logs/tendency.json', 'w', encoding='utf-8') as f:
            json.dump(Engine_test.tendency_data, f, indent=4)
        with open('agents/logs/vote.json', 'w', encoding='utf-8') as f:
            json.dump(Engine_test.vote_data, f, indent=4)
