# MEOW

## Overview

This project is the experiment code of our work "[Large Language Models Need Consultants for Reasoning: Becoming an Expert in a Complex Human System Through Behavior Simulation" ](https://arxiv.org/abs/2403.18230)

**Abstract**:
Large language models (LLMs), in conjunction with various reasoning reinforcement methodologies, have demonstrated remarkable capabilities comparable to humans in fields such as mathematics, law, coding, common sense, and world knowledge. In this paper, we delve into the reasoning abilities of LLMs within complex human systems. We propose a novel reasoning framework, termed "Mosaic Expert Observation Wall" (MEOW) exploiting generative-agents-based simulation technique. In the MEOW framework, simulated data are utilized to train an expert model concentrating "experience" about a specific task in each independent time of simulation. It is the accumulated "experience" through the simulation that makes for an expert on a task in a complex human system. We conduct the experiments within a communication game that mirrors real-world security scenarios. The results indicate that our proposed methodology can cooperate with existing methodologies to enhance the reasoning abilities of LLMs in complex human systems.

In our experiment, we utilize generative-agents-based simulation to generate behavioral data of the communication game "Find The Spy".

## Game Rules

ur experiments are based on a four-player version of this game. In this scenario, players are divided into two groups: "folk" and "spies". Three players belong to the "folk" group, while one player is designated as a "spy". At the beginning of the game, "folk" players receive the same word, while the "spy" receives a different word. These two words, while distinct, share some commonalities (e.g. two words, "apple" and "pineapple", are both fruits). Each player only knows his own word and remains ignorant of both his identity ("ordinary people" belongs to "folk" or "spy" belongs to "spies") and the identities of the other players.

The objective of the game is to eliminate the opposing group of players through communication and voting. For "spies", the goal is to conceal their identity to avoid being voted out. In the four-player scenario, when only two players remain and one of them is the "spy", the referee announces the "spies" as the winners. Conversely, the "folk" players aim to identify the "spies" based on the communication and vote them out through the voting process. If all remaining players are "folk", the referee announces the "folk" as the winners.

### Game processes

- At the beginning of a game, the referee distributes words to all players, and then several processes loop until one side wins. A loop of 2, 3, 4 is called one round, and a game may consist of one or more rounds;
- Description process: Players take turns to deliver statements, using a word (cannot be their own word or include all their fellow players who used their words before) to describe their received word;
- Discussion process: After all players give descriptions of received words, they take turns to express opinions on other players' identities.
- Voting process: Each player chooses a possible "spies" player to vote for him/her. The Player getting the most amounts of votes is eliminated from the game (If several players get the same number of votes, the player whose first vote is got earlier will be eliminated.). The referee will announce which player is eliminated by voting and reveal his/her identity. The eliminated player will be excluded from the rest of the game.

Other rules: (a) In each player's utterances at any time it is not allowed to use the word he/she receives. (b) In each player's description of words in each round they must not use words used by other players before.

#### Instructions

1.  Run `python install -r requirements.txt` first.
2.  Run `python main.py [top_p] [temperature]` to start simulation
    - Before running, you have to prepare `gameConfig.json`
      - For example:
```json
[
    [
        [
            "Alice",
            "Bob",
            "Carol",
            "Daniel"
        ],
        [
            "Alice",
            "汉堡包",
            "agents/logs/Alice_qianfan_chinese_topp095_tem08",
            1
        ],
        [
            "Bob",
            "比萨饼",
            "agents/logs/Bob_qianfan_chinese_topp095_tem08",
            0
        ],
        [
            "Carol",
            "比萨饼",
            "agents/logs/Carol_qianfan_chinese_topp095_tem08",
            0
        ],
        [
            "Daniel",
            "比萨饼",
            "agents/logs/Daniel_qianfan_chinese_topp095_tem08",
            0
        ],
        "一种快餐食品"
    ]
]
```
3.  Data process and model training
    - see `model-help.md`
4. Write `judgeConfig.json`
   - For example,
```python
[
    {
        "players": [
            "Alice",
            "Bob",
            "Carol",
            "Daniel"
        ],
        "words": [
            "番茄",
            "番茄",
            "柿子",
            "番茄"
        ],
        "hints": "一种可食用植物",
        "agent": 2,  # undercover agent id (starts from 0)
        "logFile": "judge_0.log", # game log file path judge_x.log
        "resultFile": "uttr_0.json" # judge result file path uttr_x.log
    }
]
```
-
    - Judge configs are generated by `python judge_configs_generate.py`
5. Run judge agent `python judge.py`
    - The default parameters is temperature=0.9 temperature=0.85
    - Above parameters and file path are only allowed to be modified in code
6. Result Evaluation: run `python metrics.py`
   - '0'-judge_zero.log
     - print accuracy, wa-recall, wa-f1
   - 'CoT'-judge_CoT.log-CoT
     - print accuracy, wa-recall, wa-f1
   - 'expert'-judge_expert.log-expert model
     - print accuracy, wa-recall, wa-f1
   - 'expert_CoT'-judge_expert_CoT.log-expert+COT
     - print accuracy1, wa-recall1, wa-f11, accuracy2, wa-recall2, wa-f12


#### Citation

```
@article{wang2024large,
  title={Large Language Models Need Consultants for Reasoning: Becoming an Expert in a Complex Human System Through Behavior Simulation},
  author={Wang, Chuwen and Zeng, Shirong and Wang, Cheng},
  journal={arXiv preprint arXiv:2403.18230},
  year={2024}
}
```
