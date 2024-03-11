# preprocess, check the validation of files
import json
import pickle

# Read Preprocessing Configuration File for the Dataset
config_path = './dataset/preprocess_config.json'
with open(config_path, 'r', encoding='utf-8') as file:
    config = json.load(file)

# Open the ErrorInfo file
if not os.path.exists(config["dataset_path"]):
    os.mkdir(config["dataset_path"])
errorInfo_path = f'{config["dataset_path"]}1ErrorInf.txt'
errf = open(errorInfo_path, 'w', encoding='utf-8')


def check_file(json_tendency: list, json_vote: list, json_player: list, file_id: int):
    '''
    Check if there are any errors in the datafile; if errors exist, write them to file errf.

    Args:
    json_tendency: [{'player1':{'for':[...],'against':[...],'no comment':[...]}}]. The tendencies of players extracted from the JSON file, ensure that the corresponding content for 'for' and 'against' is represented using lists.
    json_vote：[{'player1':'player_i','player2':'...',...}]. The votes of players ectracted from the JSON file.
    json_player: The basic information of players,[['player1',...,'player4'],['player1',description word,the log file,the identify(0/1)],...]
    file_id：The current file identifier being checked, used when outputting error messages

    Return:
    Return a tuple (a,b)
    a=1 indicates that the file has no problems, while a=0 indicates the presence of problems in one file
    b means the round of one game,if b=1 means that in the first round the spy is eliminated
    The value of 'b' is valid only when a=1
    '''
    n = 0
    personId = {}
    # Read the character information from 'json_player' and determine if there are any problems with the game character
    for i in range(len(json_player[0])):
        if json_player[i + 1][3] == 1:
            if n == 0:
                n = 1
            else:
                errf.write(f'There is more than one undercover player in players_{file_id}, indicating a design flaw in the game.\n')
                return 0, 0
        personId[json_player[i + 1][0]] = i
    if n == 0:
        errf.write(f'There is no spy in players_{file_id}, indicating a design flaw in the game.\n')
        return 0, 0
    noplayer = ''  # after the first round, the name of eliminated player
    flag = -1  # record the round of each game
    # shwo that the player has no for/against/vote tendency
    noidea = ['', '无', '无法选择', '不投票']
    # the infomation of each round
    for round_data, vote_data in zip(json_tendency, json_vote):
        flag += 1
        # Check if there are errors in the tendencies records, including unknown characters, contradictory tendencies etc.
        for person, opinions in round_data.items():
            if person not in personId:
                errf.write(f'Unknown characters appear in the tendency_{file_id}——{person}\n')
                return 0, 0
            if person == noplayer:
                errf.write(f'{noplayer} is eliminated, but it appear in the tendency_{file_id}\n')
                return 0, 0
            count = {}
            # count[player]=1 means person 'for'/'against' player, using this to avoid the contradictory tendency, i.e. a 'for' b and a 'aginst' b
            for supporter in opinions['for']:
                if supporter in personId and supporter in noidea:
                    # means that the player has no players to support
                    continue
                if supporter not in personId:
                    errf.write(f'In the tendency_{file_id}, unknown characters appear in the for tendency of {person}——{supporter}\n')
                    return 0, 0
                if supporter == noplayer:
                    errf.write(f'{noplayer} is eliminated，but it appear in the for tendency of {person} in tendency_{file_id}\n')
                    return 0, 0
                if supporter == person:
                    errf.write(f'In tendency_{file_id}, {person} supports himself, please delete it\n')
                    return 0, 0
                if supporter not in count:
                    count[supporter] = 1
                else:
                    errf.write(f'In tendency_{file_id}, please check the tendency of {person}\n')
                    return 0, 0
            for opponent in opinions['against']:
                if opponent in personId and opponent in noidea:
                    continue
                if opponent not in personId:
                    errf.write(f'Unknown characters appear in the tendency_{file_id}——{opponent}\n')
                    return 0, 0
                if opponent == noplayer:
                    errf.write(f'{noplayer} is eliminated， but it appear in the against tendency of {person} in tendency_{file_id}\n')
                    return 0, 0
                if opponent == person:
                    errf.write(f'In tendency_{file_id}, {person} againsts himself\n')
                    return 0, 0
                if opponent not in count:
                    count[opponent] = 1
                else:
                    errf.write(f'In tendency_{file_id}, please check the tendency of {person}\n')
                    return 0, 0
        # Check the voting data for occurrences of self-voting, unknown players, and vote for eliminated players, etc
        votecnt = {}  # vote[a]=the number of votes for a
        for voter, votee in vote_data.items():
            if voter in personId and voter != noplayer and votee in noidea:
                continue
            if voter not in personId:
                errf.write(f'Unknown characters appear in the vote_{file_id}——{voter}\n')
                return 0, 0
            if votee not in personId:
                errf.write(f'Unknown characters appear in the vote of {voter} in vote_{file_id}——{votee}\n')
                return 0, 0
            if votee == noplayer:
                errf.write(f'{noplayer} is eliminated，but {voter} votes him in vote_{file_id}\n')
                return 0, 0
            if voter == votee:
                errf.write(f'In vote_{file_id}, {voter} votes himself\n')
                return 0, 0

            if votee not in votecnt:
                votecnt[votee] = 1
            else:
                votecnt[votee] += 1
        # In the case of a tie, the person who received the first vote earlier will be eliminated, the order of keywords in a set is consistent with the order of receiving votes
        rec = 0
        for per, val in votecnt.items():
            if val > rec:
                noplayer = per
                rec = val

    return 1, flag


def process():
    '''
    The preprocessing function, it will generate valid file '1Avalid.pkl', an invalid file list '1InAvaild.pkl', and a two-round game list '1TwoRound.pkl'.
    '''
    valid = []
    tworound = []
    invalid = config['original_invalid']
    # the original invalid files, for example, the simulator encounters issues during the simulation of a certain game, leading to incomplete file generation
    path = config['dataset_path']
    for i in range(config['total_num']):
        try:
            if i in invalid:
                continue
            with open(f'{path}tendency_{i}.json', 'r', encoding='utf-8') as file:
                json_tendency = json.load(file)
            with open(f'{path}vote_{i}.json', 'r', encoding='utf-8') as file:
                json_vote = json.load(file)
            with open(f'{path}players_{i}.json', 'r', encoding='utf-8') as file:
                json_player = json.load(file)
        except IOError as e:
            print(f'There is an issue with a file indexed by {i}:{e}')
            exit(0)
        v, r = check_file(json_tendency, json_vote, json_player, i)
        if v == 1:
            valid.append(i)
            if r == 1:
                # The game consists of two rounds. In our experiment, the game either consists of only one round or only two rounds
                tworound.append(i)
        else:
            invalid.append(i)
    errf.write(f'The effective data quantity in folder {path} is {len(valid)}, the length of games with two rounds is {len(tworound)}. The invalid data quantity is {len(invalid)}')
    errf.close()
    with open(f'{path}1Avalid.pkl', 'wb') as file:
        pickle.dump(valid, file)
    with open(f'{path}1InAvalid.pkl', 'wb') as file:
        pickle.dump(invalid, file)
    with open(f'{path}1Tworound.pkl', 'wb') as file:
        pickle.dump(tworound, file)
    print(f'The effective data quantity in folder {path} is {len(valid)}, the length of games with two rounds is {len(tworound)}. The invalid data quantity is {len(invalid)}')
    print(f'Please open the 1ErrorInf.txt file under {path} to view the problematic files and make the necessary modifications.')


if __name__ == '__main__':
    process()
