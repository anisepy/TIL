import pandas as pd
# from sklearn.datasets import load_iris


def load_train():
    data = pd.read_csv('ml-project/data/train_V2.csv')
    target = 'winPlacePerc'  
    df = data.copy()
    exclude_cols = ['Id','matchId','matchType', 'groupId','rankPoints','winPoints','killPoints']
    df = df.drop(exclude_cols, axis=1)
    df = df.dropna()

    return df, target

def load_test():
    data = pd.read_csv('ml-project/data/test_V2.csv')
    test = data.copy()
    test = test.dropna()
    # exclude_cols = ['Id','matchId','groupId','rankPoints','winPoints','killPoints']
    # test = test.drop(exclude_cols, axis=1)

    return test