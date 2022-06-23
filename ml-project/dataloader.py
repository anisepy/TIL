import pandas as pd
# from sklearn.datasets import load_iris


def load_train():
    data = pd.read_csv('./data/train_V2.csv')
    target = 'winPlacePerc'  
    df = data.copy()
    # df = df.dropna()
    # df = df.drop(['matchType','Id','matchId','groupId','killPlace'],axis=1)

    return df, target

def load_test():
    test = pd.read_csv('./data/test_V2.csv')

    submission = pd.read_csv("./data/sample_submission_V2.csv")

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
    return test, submission