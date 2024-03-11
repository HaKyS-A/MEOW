"""player module for simulator"""
from agents.player_chat_basic_qianfan_chinese import PlayerChatBasicQianfanChinese


class PlayersQianfanChinese:
    """
    LLM Role-play interface fo player agent for simulator
    add 2 states:
    alive - still in the game
    identity - identity in the game
    """

    def __init__(self, name, word, remain_players, context_sheet: str, identity, context=None, top_p=0.75, temperature=0.9):
        self.chat = PlayerChatBasicQianfanChinese(name, word, remain_players, context_sheet, context, top_p=top_p, temperature=temperature)
        self.name = name
        self.alive = True
        self.identity = identity    # 0-ordinary people; 1-undercover agent
