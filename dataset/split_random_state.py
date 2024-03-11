# we split the total dataset to ten different training-test sets
import os
import pickle
import shutil

from sklearn.model_selection import train_test_split

filePath = './dataset/meow_8k_095_080/'  # original dataset path
newPath = './dataset/'  # the root folder of split dataset


def removeToFolder(random_state: int, dataset: list, mode: str):
    '''
    Split the original dataset, and remove the training-test set to floder.

    Args:
    random_state: The split way, used in train_test_split ( When using the same value for the random_state parameter in the train_test_split function, the partitioning results will be same). In our experiments, the random_state is set from 1 to 10.
    dataset: Index list of the original dataset
    mode: In our experiments, mode='OneRound' means all valid dataset but we only use the first round of each game,mode='TwoRound' means the set of games with two rounds.
    '''
    # 8:2,training : test
    trainval_dataset, test_dataset = train_test_split(
        dataset, train_size=0.8, random_state=random_state
    )
    path = f'{newPath}{mode}-{random_state}'
    if not os.path.exists(path):
        os.makedirs(path)
    path1 = f'{path}/TrainVal_set/'
    if not os.path.exists(path1):
        os.makedirs(path1)
    # get the index of the file and remove it
    for i in trainval_dataset:
        shutil.copy(f'{filePath}uttr_{i}.json', f'{path1}uttr_{i}.json')
        shutil.copy(f'{filePath}players_{i}.json', f'{path1}players_{i}.json')
        shutil.copy(f'{filePath}tendency_{i}.json', f'{path1}tendency_{i}.json')
        shutil.copy(f'{filePath}vote_{i}.json', f'{path1}vote_{i}.json')
    path2 = f'{path}/Test_set/'
    if not os.path.exists(path2):
        os.makedirs(path2)
    for i in test_dataset:
        shutil.copy(f'{filePath}uttr_{i}.json', f'{path2}uttr_{i}.json')
        shutil.copy(f'{filePath}players_{i}.json', f'{path2}players_{i}.json')
        shutil.copy(f'{filePath}tendency_{i}.json', f'{path2}tendency_{i}.json')
        shutil.copy(f'{filePath}vote_{i}.json', f'{path2}vote_{i}.json')


def split_it():
    '''
    split the total dataset to ten different training-test sets
    '''
    try:
        with open(f'{filePath}1Avalid.pkl', 'rb') as file:
            dataset_One = pickle.load(file)  # Corresponding file indices
    except FileNotFoundError:
        print(
            f'{filePath}1Avalid.pkl does not exist. Please perform preprocessing operations first by running preprocess.py'
        )
    try:
        with open(f'{filePath}1Tworound.pkl', 'rb') as file:
            dataset_Two = pickle.load(file)  # Corresponding file indices
    except FileNotFoundError:
        print(
            f'{filePath}1Tworound.pkl does not exist. Please perform preprocessing operations first by running preprocess.py'
        )
    for i in range(1, 11):
        removeToFolder(i, dataset_One, 'OneRound')
        removeToFolder(i, dataset_Two, 'TwoRound')


if __name__ == '__main__':
    split_it()
