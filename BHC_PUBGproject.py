# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataframe setting
print("pandas version: ", pd.__version__)
pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 100)

# data load
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/미니프로젝트_pubg/train_V2.csv")
df = data.copy()

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
reduce_mem_usage(df)

# 결측치 제거
df = df.dropna()
# 같은 게임에 참여한 사람 수 컬럼 수 만듦
df.loc[:,'num']=df.groupby('matchId')['Id'].transform('count')
# 한 게임에서 최대 킬수 컬럼
df.loc[:,'max']=df.groupby('matchId')['kills'].transform('max')

# df.loc[df['num']<=df['max'],['num','max']] 
# 최대 킬수가 한 게임 사람 수 보다 많을 수 없음 -> 2124 case 확인 -> drop
df=df[df['num']>df['max']]

# Decompose matchType
df.loc[df.matchType.str.contains('solo'),'solo'] = 1
df.loc[~df.matchType.str.contains('solo'),'solo'] = 0
df.loc[df.matchType.str.contains('duo'),'duo'] = 1
df.loc[~df.matchType.str.contains('duo'),'duo'] = 0
df.loc[df.matchType.str.contains('squad'),'squad'] = 1
df.loc[~df.matchType.str.contains('squad'),'squad'] = 0
df.loc[(df.matchType.str.contains('normal'))|
       (df.matchType.str.contains('crash'))|
       (df.matchType.str.contains('flare')),'event'] = 1
df.loc[(~df.matchType.str.contains('normal'))&
       (~df.matchType.str.contains('crash'))&
       (~df.matchType.str.contains('flare')),'event'] = 0
df.isna().sum()

## 참가 인원에 비해 팀 수가 너무 적은 경우 조정
# 팀 수 의 이상치 열 생성
df.loc[:,'numDuo']=df['num']//2
df.loc[:,'numSquad']=df['num']//4

df.loc[df.duo == 1.0,'Gap'] = abs(df['numDuo']-df['numGroups'])
df.loc[df.squad == 1.0,'Gap'] = abs(df['numSquad']-df['numGroups'])
df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap']=abs(df['num']-df['numGroups'])

# df['Gap'].value_counts().to_frame().sort_index()

# solo나 event에서 numGroups 이상치 탐색
# df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap'].value_counts()
# len(df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0)&(df.Gap>9),'Gap']) # 24907
# print("%.2f" % ((24907/663429)*100)) # 3.75%
# solo나 event인데 Gap이 9보다 큰 경우들 평균값(반올림해서 3)되도록 maxPlace와 numGroup조정
df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0) & (df.Gap > 9),['maxPlace','numGroups']] = df['num']-3
df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap']=abs(df['num']-df['numGroups'])
df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap'].value_counts()

# duo에서 numGroups 이상치 탐색
# df.loc[(df.matchType.str.contains('duo')),['Gap']].value_counts()
# Gap이 5이상 차이 나면 groupId에 문제가 있는 것으로 판단, 사실 그 이하도 문제 존재
# df.loc[(df.matchType.str.contains('duo'))&(df.Gap==5.0),['matchId']].value_counts()
# df.loc[df.matchId== '35c26cc0a5212a','groupId'].value_counts() # 한팀에 2명 이상인 팀 들 5팀 이상
# 보통 API가 꼬인 경우로 보인다. -> Gap이 5이상인 경우 가장 많은 값인 1이 되도록 조정한다.
df.loc[(df.matchType.str.contains('duo'))&(df.Gap>=5),['maxPlace','numGroups']] = df['numDuo']-1
df.loc[df.duo == 1.0,'Gap']=abs(df['numDuo']-df['numGroups'])
df.loc[(df.matchType.str.contains('duo')),'Gap'].value_counts()

#squad에서 numgroups이상치 탐색 
# df.loc[(df.matchType.str.contains('squad')),['Gap']].value_counts()
# Gap이 8이상 차이 나는 것은 groupId에 이상이 있는 것으로 판단, 사실 그 이하도 문제 존재
# Gap이 8이상인 경우 가장 많은 값인 4가 되도록 조정한다.
df.loc[(df.matchType.str.contains('squad'))&(df.Gap>=8),['maxPlace','numGroups']] = df['numSquad']-4
df.loc[df.squad == 1.0,'Gap']=abs(df['numSquad']-df['numGroups'])
df.loc[(df.matchType.str.contains('squad')),'Gap'].value_counts()


## maxPlace가 numGroups와 많이 차이나는 것 조정
df.loc[:,'GrpError']=df['maxPlace']-df['numGroups']
# df.loc[df['GrpError'] > df['GrpError'].quantile(0.99),'GrpError'].value_counts() # 7, 8, 9
# df['GrpError'].mean() # 1.445498e+00

# maxPlace와 numGroups가 많이 차이나는 것(7 이상)은 matchId의 오류로 보고 평균값(반올림해서 2)만큼의 차이를 둬서(maxPlace값 - 2) 조정한다.
df.loc[df.GrpError>=7,'numGroups'] = df['maxPlace']-2
df.loc[:,'GrpError']=df['maxPlace']-df['numGroups']
df['GrpError'].value_counts()

# 마지막 확인
df.loc[df.duo == 1.0,'Gap'] = abs(df['numDuo']-df['numGroups'])
df.loc[df.squad == 1.0,'Gap'] = abs(df['numSquad']-df['numGroups'])
df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap']=abs(df['num']-df['numGroups'])

df.loc[df.event ==0.0,'Gap'].value_counts().to_frame().sort_index()
# Gap = 8.0 에서 20명이 존재하는데 이는 matchType이 crash 여서 무시한다.

# 만든열 드랍
df=df.drop(['num','numDuo','numSquad','max','Gap','GrpError'],axis=1)

# kills, headshotKills, killStreaks, longestKill 이상치 제거 (각 column 별 약 1000개 정도 없어지도록 값을 잡음)

df=df.drop(index=df[ (df['kills']>15) | (df['headshotKills']>7) | (df['killStreaks']>5) | (df['longestKill']>500)  ].index)

# rideDistance = 0 인데 roadKill 이 존재하는 것 -> 이상치
df=df.drop(index=df[ (df['rideDistance']==0) & (df['roadKills']>0)].index)
​
# damageDealt = 0 인데 kills가 존재하는 것 -> 이상치
df=df.drop(index=df[ (df['damageDealt']==0) & (df['kills']>0)].index)

# maxGrp_killPlace : 그룹 별로 능력치는 다르지만 같은 그룹이면 같은 등수를 나타내는 것이기 때문에 가장 잘하는 사람 기준으로 해서 보기 위한 column
df['maxGrp_killPlace'] = df.groupby('groupId')['killPlace'].transform('max')
df['maxGrp_kills'] = df.groupby('groupId')['kills'].transform('max')
df['maxGrp_totalDistance'] = df.groupby('groupId')['totalDistance'].transform('max')

# sniper : 얻은 무기 수 대비 kill 거리의 비율
df['sniper'] = df['longestKill']/100*df['weaponsAcquired']

# kill 기준
# 킬 대비 헤드샷 비율
df['headshotRate'] = df['headshotKills'] / df['kills']
# 킬이 0인 경우 결측치 발생 -> 0 넣어줌
df['headshotRate'] = df['headshotRate'].fillna(0) 
# 킬 대비 연속킬 비율
df['killStreakRate'] = df['killStreaks'] / df['kills']

df['killsPerDistance'] = df['totalDistance'] / df['kills']

# 킬수/매치인원 비율 피쳐
df['kill_per']= df['kills']  / df['num']

# 이들의 상관관계 확인
derived_corr = df[['maxGrp_killPlace','maxGrp_kills','maxGrp_totalDistance', 'sniper', 'headshotRate', 'killStreakRate', 'killsPerDistance','winPlacePerc']].corr()
derived_corr
# maxGrp_killPlace가 0.8이상으로 높게 나옴

# 파생 변수 생성
df['healthitems'] = df['heals'] + df['boosts']
df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
df["skill"] = df["headshotKills"] + df["roadKills"]

# 상위 10% 고수 열 생성
# df.loc[df['winPlacePerc'] > df['winPlacePerc'].quantile(0.9),'winPlacePerc'].min()
# >>> 0.9143
df.loc[df.winPlacePerc>0.9143,'superior'] = 1
df.loc[df.winPlacePerc<=0.9143,'superior'] = 0

# 하위 10% 초보 열 생성
# df.loc[df['winPlacePerc'] < df['winPlacePerc'].quantile(0.1),'winPlacePerc'].max()
# >>> 0.0635
df.loc[df.winPlacePerc<0.0635,'beginner'] = 1
df.loc[df.winPlacePerc>=0.0635,'beginner'] = 0